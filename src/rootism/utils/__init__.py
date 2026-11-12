"""Small, dependency-light shared helpers."""

from .hashing import json_load, json_save, stable_hash
from .seed import resolve_dtype, set_seed

__all__ = ["json_load", "json_save", "stable_hash", "set_seed", "resolve_dtype"]
