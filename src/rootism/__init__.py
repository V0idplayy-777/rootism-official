"""Rootism — an independently implemented AI coding model and platform.

Stage 1 exposes the model, tokenizer, configuration/versioning, checkpoint I/O
and token accounting. Later stages add data, training, inference, agent, serving,
evaluation and the web platform. See docs/STATUS.md for the authoritative
implementation state of every subsystem.
"""

from .version import (
    SCHEMA_VERSION,
    MODEL_ARCHITECTURE_ID,
    ROOTISM_VERSION,
    __version__,
)

__all__ = [
    "SCHEMA_VERSION",
    "MODEL_ARCHITECTURE_ID",
    "ROOTISM_VERSION",
    "__version__",
]
