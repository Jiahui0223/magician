import os
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
from seen2scene.tools.log_utils import get_logger
from seen2scene.configs.opt import Base
from seen2scene.tools import metrics
from seen2scene.tools.gradient_monitor import create_gradient_distribution_plot

logger = get_logger(file_name=__file__, debug="generator")


class Net(pl.LightningModule):
    def __init__(self, args: Base):
        super().__init__()
        self.learning_rate = args.learning_rate
        self.latent_key = args.latent_key
        self.mask_unknown = args.mask_unknown
        self.lr_scheduler = args.lr_scheduler
        self.ckpt_path = args.ckpt_path
        self.task = args.task
        self.acc_grad_batches = args.accumulate_grad_batches

        # Gradient monitoring
        self.monitor_gradients = args.monitor_gradients
        self.monitor_grad_freq = args.monitor_grad_freq

        # Semantic distribution monitoring
        self.monitor_semantic_distribution = args.monitor_semantic_distribution
        self.monitor_semantic_freq = args.monitor_semantic_freq
        if self.monitor_semantic_distribution:
            from seen2scene.tools.latent_monitor import SemanticDistributionTracker

            self.semantic_tracker = SemanticDistributionTracker(
                top_k=args.monitor_semantic_top_k
            )
            logger.info(
                f"Semantic monitoring enabled (frequency={self.monitor_semantic_freq})"
            )

        # Metrics
        self.metrics = {"model": metrics.Metrics(args.task.metrics)}
        self.metrics["model"].requires_grad_(False)
        self.metrics["model"].eval()

    def setup(self, stage: str) -> None:
        """Setup models and move to correct devices for training/validation/testing.

        Called by PyTorch Lightning after model initialization. Moves all submodules
        to the appropriate device and sets up distributed training if needed.

        Args:
            stage: Training stage - "fit", "validate", "test", or "predict".

        Notes:
            - Synchronizes log directories across distributed processes
        """
        self.log_dir = self.logger.save_dir
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            log_dir_ls = [self.log_dir]
            torch.distributed.broadcast_object_list(log_dir_ls, src=0)
            self.log_dir = log_dir_ls[0]
        os.makedirs(self.log_dir, exist_ok=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        lr = self.lr_scheduler.lr
        min_lr = self.lr_scheduler.min_lr
        self.param_groups = [{"params": params, "initial_lr": lr, "lr": lr}]
        opt = torch.optim.AdamW(self.param_groups)

        warmup_steps = self.lr_scheduler.num_warmup_steps
        total_steps = self.trainer.estimated_stepping_batches
        logger.info(
            f"Setting up scheduler: lr={lr:.2e}, min_lr={min_lr:.2e}, warmup_steps={warmup_steps}, total_steps={total_steps}"
        )

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup: ramp from min_lr to lr
                alpha = step / max(1, warmup_steps)
                return min_lr / lr + alpha * (1.0 - min_lr / lr)
            else:
                # Cosine decay: lr -> min_lr
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr / lr + cosine * (1.0 - min_lr / lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        try:
            loss_dict, stats_dict = self.losses(batch)
        except torch.OutOfMemoryError:
            logger.warning(f"OOM, skipping scenes: {batch['scene_names']}")
            loss_dict = {"loss": 0.0}
            stats_dict = {}

        for k, v in loss_dict.items():
            assert not torch.isnan(v), f"NaN loss detected for {k}"
        loss = sum(loss_dict.values())

        kwargs = {
            "logger": True,
            "on_step": True,
            "on_epoch": True,
            "batch_size": len(batch["scene_names"]),
            "sync_dist": True,
            "rank_zero_only": True,
        }
        self.log_dict(loss_dict, prog_bar=True, **kwargs)
        self.log_dict(stats_dict, **kwargs)

        return loss

    def on_before_optimizer_step(self, optimizer):
        if not (
            self.monitor_gradients and self.global_step % self.monitor_grad_freq == 0
        ):
            return
        for lg in self.loggers:
            if isinstance(lg, WandbLogger):
                fig = create_gradient_distribution_plot(
                    self,
                    gradient_clip_val=self.trainer.gradient_clip_val,
                    step=self.global_step,
                )
                lg.experiment.log(
                    {
                        "charts/gradient_distribution": wandb.Plotly(fig),
                        "trainer/global_step": self.global_step,
                    }
                )
            elif isinstance(lg, TensorBoardLogger):
                grads = [p.grad.detach().flatten() for p in self.parameters() if p.grad is not None]
                if not grads:
                    continue
                writer = lg.experiment
                writer.add_scalar(
                    "gradients/global_norm",
                    torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item(),
                    self.global_step,
                )
                values = torch.cat(grads)
                if values.numel() > 1_000_000:  # subsample: histograms of 300M values are slow
                    values = values[torch.randint(0, values.numel(), (1_000_000,), device=values.device)]
                writer.add_histogram("gradients/values", values.float().cpu(), self.global_step)

    def log_semantic_distribution(self, ins_mask, object_names, latent_tgt):
        """Log semantic class distribution (W&B: Plotly chart, TensorBoard: scalars).

        Args:
            ins_mask: Instance mask [N, M] where N is voxels, M is objects
            object_names: List of object name lists per batch item
            latent_tgt: VDBTensor with grid information
        """
        if (
            not self.monitor_semantic_distribution
            or self.global_step % self.monitor_semantic_freq != 0
        ):
            return

        self.semantic_tracker.update(ins_mask, object_names, latent_tgt.grid.jidx)
        for lg in self.loggers:
            if isinstance(lg, WandbLogger):
                fig = self.semantic_tracker.create_plot(self.global_step)
                lg.experiment.log(
                    {
                        "charts/semantic_distribution": wandb.Plotly(fig),
                        "trainer/global_step": self.global_step,
                    }
                )
            elif isinstance(lg, TensorBoardLogger):
                # Accumulated voxel share of the top-k classes (incl. "empty")
                counts = self.semantic_tracker.class_counts
                total = max(sum(counts.values()), 1)
                top = sorted(counts.items(), key=lambda kv: -kv[1])
                for name, count in top[: self.semantic_tracker.top_k]:
                    lg.experiment.add_scalar(
                        f"semantic_distribution/{name}", count / total, self.global_step
                    )

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if not hasattr(self.task, "num_samples"):
            return
        batch_size = len(batch["scene_names"])
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        global_sample_idx = (batch_idx * world_size + rank) * batch_size
        if global_sample_idx >= self.task.num_samples:
            self.trainer.should_stop = True

    def on_validation_epoch_end(self):
        self.log_dict(
            self.metrics["model"].compute(prefix="val/"),
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metrics["model"].reset()

    def on_test_epoch_end(self):
        self.log_dict(
            self.metrics["model"].compute(prefix="test/"),
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metrics["model"].reset()
