"""Model registry: maps WASTECAP_BACKEND → CaptionModel instance.

Lazy-loaded: weights are never imported at module load time.
The registry reads environment variables at call time, not import time.
"""

from __future__ import annotations

import functools
import logging
import os

from wastecap.models.base import CaptionModel

logger = logging.getLogger(__name__)

_BACKEND_CLASSES: dict[str, str] = {
    "mock": "wastecap.models.mock.MockModel",
    "blip2": "wastecap.models.blip2.Blip2Model",
    "qwen_vl": "wastecap.models.qwen_vl.QwenVLModel",
    "llava": "wastecap.models.llava.LlavaModel",
}


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@functools.lru_cache(maxsize=4)
def _load_model(backend: str) -> CaptionModel:
    """Load and cache a model instance by backend name."""
    if backend not in _BACKEND_CLASSES:
        raise ValueError(f"Unknown backend {backend!r}. Valid options: {sorted(_BACKEND_CLASSES)}")
    cls = _import_class(_BACKEND_CLASSES[backend])
    logger.info("Loading model backend: %s", backend)
    return cls()  # type: ignore[return-value]


def get_model(backend: str | None = None) -> CaptionModel:
    """Return the active CaptionModel instance.

    Args:
        backend: Override the backend name. Defaults to WASTECAP_BACKEND env var.
                 Falls back to 'mock' if the env var is not set.
    """
    name = backend or os.environ.get("WASTECAP_BACKEND", "mock")
    return _load_model(name)


def clear_cache() -> None:
    """Evict all cached model instances (useful in tests)."""
    _load_model.cache_clear()
