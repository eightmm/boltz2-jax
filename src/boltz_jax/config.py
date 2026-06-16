"""Runtime configuration for the experimental JAX port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Static inference runtime settings."""

    backend: str = "gpu"
    dtype: str = "float32"
    seed: int = 42


@dataclass(frozen=True)
class ShapeBuckets:
    """Static shape buckets for JAX compilation."""

    residues: tuple[int, ...] = (128, 256, 512)
    msa_rows: int = 1024
    diffusion_steps: int = 200


@dataclass(frozen=True)
class ProjectPaths:
    """Input/output paths shared by probes."""

    pytorch_checkpoint: Path = Path("../boltz/.cache/boltz/boltz2_conf.ckpt")
    processed_dir: Path = Path(
        "../boltz/benchmark_results/prep/boltz_results_prot_no_msa/processed"
    )
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class BoltzJaxConfig:
    """Top-level project configuration."""

    runtime: RuntimeConfig = RuntimeConfig()
    shape_buckets: ShapeBuckets = ShapeBuckets()
    paths: ProjectPaths = ProjectPaths()
