"""Training the semantic-VAD model: Whisper encoder → linear adapter → Qwen3 → EOT marker.

Only the **pure** helpers (:mod:`~.prompt`, :mod:`~.packing`) are re-exported here so that
``import semantic_vad.training`` stays torch/transformers-free. The heavy pieces are
imported explicitly by callers::

    from semantic_vad.training.modeling import WhisperQwen3
    from semantic_vad.training.data import SemanticVADDataset, collate_packed
    from semantic_vad.training.train import main   # `python -m semantic_vad.training.train`

Requires the ``train`` extra (``uv pip install -e ".[train]"``) plus a flash-attention
build — see README's "Training the model" section. Runs on a **GPU** pod, not the CPU
dataset-build pod.
"""

from .packing import build_cu_seqlens
from .prompt import (
    DEFAULT_SYSTEM,
    IGNORE_INDEX,
    SPECIAL_TOKENS,
    SpecialIds,
    build_example,
    num_audio_tokens,
)

__all__ = [
    "DEFAULT_SYSTEM",
    "IGNORE_INDEX",
    "SPECIAL_TOKENS",
    "SpecialIds",
    "build_example",
    "num_audio_tokens",
    "build_cu_seqlens",
]
