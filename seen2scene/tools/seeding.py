"""Reproducible seeds for world hypotheses (exp1-task5).

`seed_world(scene, episode, step, k)` gives every generated world its own seed.
S1, S2 and VHD therefore share exactly the same K worlds for the same state, and S1
uses world k=0 of S2's set. `world_rng(seed)` runs a block with that seed and restores
the global RNG state afterwards, so seeding one world does not disturb any other random
stream. It covers both the flow noise and the stochastic VAE posterior sample.
"""
import contextlib
import hashlib

import torch

GLOBAL_SEED = 2027


def seed_world(scene: str, episode: int, step: int, k: int, base: int = GLOBAL_SEED) -> int:
    """Deterministic 63-bit seed for world k of (scene, episode, step)."""
    h = hashlib.sha256(f"{base}|{scene}|{episode}|{step}|{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


@contextlib.contextmanager
def world_rng(seed: int):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)  # seeds CPU and all CUDA generators inside the fork
        yield
