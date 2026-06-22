"""JAX/XLA inference port of Boltz-2.

Public API (lazily imported so ``import boltz_jax`` stays cheap and torch-free):

- ``predict`` / ``featurize`` — high-level end-to-end inference (api.py).
- ``build_job_yaml`` — build a job YAML from bare sequences/ligands.
- ``boltz2_predict`` — low-level pure-JAX model fn (compose with other JAX code).
- ``load_params`` — load native (safetensors) weights into a JAX pytree.
"""

from typing import TYPE_CHECKING

__all__ = [
    "__version__",
    "predict",
    "featurize",
    "build_job_yaml",
    "boltz2_predict",
    "load_params",
]

__version__ = "0.1.0"

_LAZY = {
    "predict": ("boltz_jax.api", "predict"),
    "featurize": ("boltz_jax.api", "featurize"),
    "build_job_yaml": ("boltz_jax.data.job_yaml", "build_job_yaml"),
    "boltz2_predict": ("boltz_jax.models.predict", "boltz2_predict"),
    "load_params": ("boltz_jax.bridge.native", "load_params"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'boltz_jax' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static-analysis hints only; not executed at runtime
    from boltz_jax.api import featurize, predict
    from boltz_jax.bridge.native import load_params
    from boltz_jax.data.job_yaml import build_job_yaml
    from boltz_jax.models.predict import boltz2_predict
