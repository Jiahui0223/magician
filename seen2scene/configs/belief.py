import dataclasses
from typing import *


@dataclasses.dataclass(kw_only=True)
class DepthLikelihood:
    """Observation model of the belief update: how well a world explains the new depth rays.

    Every ray has an observed range t_obs (inf = no return). Inside the world grid each ray is one of
    - blocked: t_obs ends before the ray enters the grid (an occluder outside the grid) -> no information;
    - surface: t_obs ends inside the grid -> the world's first surface hit t_hat should be at t_obs. Probability of the
      observed range bin (width bin_vox voxels) given the world: (1 - outlier_prob) * [Phi((t_obs + b/2 - t_hat) / sigma)
      - Phi((t_obs - b/2 - t_hat) / sigma)] + outlier_prob * b / L, with L the ray's grid segment and
      sigma = sigma0_vox * voxel + sigma_rel * t_obs. A world that misses the surface or has one in front of it
      (free-space violation) only gets the outlier part;
    - pass: t_obs ends beyond the grid or there is no return -> a world hit inside the grid has probability
      free_violation_prob, no hit 1 - free_violation_prob.
    Both kinds are probabilities, so they can be averaged. With novel_only, only rays whose observed segment crosses a
    voxel that was UNKNOWN before the new observation are scored (rays through already observed space cannot tell the
    particles apart and would only dilute the average).
    Tempering: neighbouring rays are strongly correlated, so a frame counts as `effective_rays_per_frame` observations:
    log-likelihood = sum over frames of effective_rays_per_frame * mean per-ray log-probability of that frame.

    Attributes:
        sigma0_vox: Range-independent depth noise in voxels (EXP4 architecture §3.3: 2 voxels).
        sigma_rel: Range-proportional depth noise (fraction of the range).
        bin_vox: Width of the observed range bin in voxels.
        outlier_prob: Mixture weight of the uniform outlier term of surface rays.
        free_violation_prob: Probability that a free ray hits a world surface (pass rays).
        margin_vox: Tolerance in voxels when a range is compared with the grid entry / exit.
        novel_only: Score only rays that cross previously UNKNOWN voxels (needs the previous observation state).
        effective_rays_per_frame: Tempering (see above). Not calibrated: the EXP4 plan selects it on training scenes.
        inlier_sigmas: A surface ray is an inlier if |t_hat - t_obs| <= inlier_sigmas * sigma.
        step_vox: Ray-marching step in voxels (as active.raycast.PatchRaycaster).
    """

    sigma0_vox: float = 2.0
    sigma_rel: float = 0.0
    bin_vox: float = 1.0
    outlier_prob: float = 0.05
    free_violation_prob: float = 0.05
    margin_vox: float = 1.0
    novel_only: bool = True
    effective_rays_per_frame: float = 1.0
    inlier_sigmas: float = 2.0
    step_vox: float = 0.5


@dataclasses.dataclass(kw_only=True)
class ObservationConstraint:
    """Known-observation constraint of the updated worlds (current fused observation).

    Attributes:
        mode: "check" (default) = report violations only, the generated worlds are kept unchanged; "paste" = observed
            BAND voxels take the observed TSDF and observed EMPTY voxels become +truncation; "carve" = only the EMPTY
            part; "none" = skip. A projected copy changes Q1 / Q2 by construction, so evaluations compare unprojected
            worlds (or project every method the same way).
        band_dilation: A BAND voxel counts as explained if a world surface voxel is within this many voxels.
    """

    mode: Literal["check", "paste", "carve", "none"] = "check"
    band_dilation: int = 1


@dataclasses.dataclass(kw_only=True)
class C0Particle:
    """C0: particle belief update without learning (reweight -> ESS -> resample / refresh -> constraint check).

    Attributes:
        likelihood: Depth observation model.
        constraint: Observation constraint check / projection of the updated worlds.
        ess_threshold: Resample when ESS < ess_threshold * K.
        resampling: Resampling scheme ("systematic" has the lowest variance of the three).
        refresh: After resampling, "duplicates" regenerates every copy beyond the first of a particle with the frozen
            generator conditioned on the current observation; "none" keeps the copies.
        refresh_seeds: Seeds of refreshed duplicates: "new" = a fresh world seed (new particle id), "persist" = the
            seed key of the particle's lineage at seed step 0 (the same generator noise as its first generation).
        collapse_inlier_rate: The set has collapsed (all worlds inconsistent with the new rays) when no particle reaches
            this inlier rate or every likelihood is -inf / NaN.
        on_collapse: "refresh_all" regenerates all K particles from the generator; "keep" only reweights.
        collapse_seeds: Seeds of a collapse refresh ("persist" = EXP4 architecture §3.3 recovery rule, FR-persist).
        seed: Base seed of the ray subset / resampling generators.
        raise_on_refresh_failure: Raise instead of keeping the old particles when the generator fails.
    """

    likelihood: DepthLikelihood = dataclasses.field(default_factory=DepthLikelihood)
    constraint: ObservationConstraint = dataclasses.field(default_factory=ObservationConstraint)
    ess_threshold: float = 0.5
    resampling: Literal["systematic", "residual", "multinomial"] = "systematic"
    refresh: Literal["duplicates", "none"] = "duplicates"
    refresh_seeds: Literal["new", "persist"] = "new"
    collapse_inlier_rate: float = 0.3
    on_collapse: Literal["refresh_all", "keep"] = "refresh_all"
    collapse_seeds: Literal["new", "persist"] = "persist"
    seed: int = 2027
    raise_on_refresh_failure: bool = False


@dataclasses.dataclass(kw_only=True)
class C1Network:
    """C1 stochastic latent update adapter (seen2scene.belief.c1.network.C1LatentUpdateNet).

    The network maps (old world latent, observation condition, noise) to an update of the old latent on the dense
    latent grid of the generator (factor 8, `latent_channels` channels). Zero-initialised output: at initialisation the
    update is exactly the identity. No training objective is defined here.

    Attributes:
        latent_channels: Channels of the generator latent (SparseVAE.channels).
        cond_channels: Channels of the observation condition (c1.conditioning.LatentConditionBuilder: 6 + C + 1).
        model_channels: Token width.
        num_blocks: Transformer blocks (windowed self-attention + cross-attention to observation tokens + MLP).
        num_heads: Attention heads.
        window_size: Side of the cubic attention window in latent cells; odd blocks shift it by half a window.
        mlp_ratio: MLP hidden width / model_channels.
        noise_channels: Channels of the per-cell Gaussian noise field (stochastic update).
        global_noise_dim: Dimension of the per-particle Gaussian code (adaLN modulation).
        obs_grid: Side of the coarse grid of observation tokens for cross-attention (0 = no cross-attention).
        qk_norm: RMS-normalise queries and keys (stability).
        use_checkpoint: Activation checkpointing of the blocks during training.
    """

    target: str = "seen2scene.belief.c1.network.C1LatentUpdateNet"
    latent_channels: int = 8
    cond_channels: int = 15
    model_channels: int = 256
    num_blocks: int = 4
    num_heads: int = 8
    window_size: int = 4
    mlp_ratio: float = 4.0
    noise_channels: int = 8
    global_noise_dim: int = 64
    obs_grid: int = 8
    qk_norm: bool = True
    use_checkpoint: bool = False


@dataclasses.dataclass(kw_only=True)
class C1Latent:
    """C1 updater: one stochastic latent update per particle, decoded with the frozen VAE decoder.

    Attributes:
        network: C1 network configuration.
        constraint: Observation constraint check / projection of the decoded worlds.
        reweight: Also reweight the particles with the depth likelihood of the decoded worlds (as C0).
        likelihood: Depth observation model used when `reweight` is set.
        seed: Base seed of the per-particle noise (keyed by scene, episode, step, particle id).
        noise_scale: Multiplier of the noise field and code (0 = deterministic update).
    """

    network: C1Network = dataclasses.field(default_factory=C1Network)
    constraint: ObservationConstraint = dataclasses.field(default_factory=ObservationConstraint)
    reweight: bool = False
    likelihood: DepthLikelihood = dataclasses.field(default_factory=DepthLikelihood)
    seed: int = 2028
    noise_scale: float = 1.0
