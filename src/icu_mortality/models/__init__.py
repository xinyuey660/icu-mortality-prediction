"""Model registry with optional heavy dependencies imported lazily."""

from .torch_models import (
    BiGRUAttentionClassifier,
    GRUAttentionClassifier,
    GRUDClassifier,
    TripletTransformerClassifier,
)

__all__ = [
    "BiGRUAttentionClassifier",
    "GRUAttentionClassifier",
    "GRUDClassifier",
    "TripletTransformerClassifier",
]

