"""EXP6 V3: scratch-trained 3D VAE partial-TSDF encoder (+ one camera-ray cross-attention block) feeding the V2
ray encoder / rectified-flow future-view generator.

    partial_tsdf.py     state-aware 4x aggregation of the fused P1 (sum / cnt / empty / domain) into the VAE input
                        [sdf, band fraction, free fraction] + domain mask; blob (de)compression; reconstruction targets
    vae.py              simple 3D convolutional VAE (encoder -> mu / logvar, train-time decoder), KL per valid unit
    cross_attention.py  scene-memory tokens from the latent, frustum token selection, the single cross-attention block
    flow_unet.py        the V2 2D rectified-flow U-Net with an optional cross-attention at the bottleneck
    model.py            V3Model: VAE + ray encoder + flow U-Net (T0 without / T1 with cross-attention)
    data.py             training data: E2 D100 view shards (ray channels, labels) + the 4x TSDF cache (VAE input)
    losses.py           L = L_FM + lambda_rec * L_recon + beta(step) * KL, every term normalised as frozen in protocol.md
    trainer.py          DDP training loop (AdamW, warmup + cosine, bf16, EMA, atomic checkpoints with RNG, resume)
    predictor.py        V3Predictor: encode a live fusion, condition candidate views, sample K futures (planner side)
    eval_dev.py         development-set FM loss and sample metrics (the V2 definitions) for checkpoint curves

Experiment entry points, configs, tests and reports: experiments/exp6_v3/ (protocol.md is the frozen rule set).
"""
