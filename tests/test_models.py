from __future__ import annotations

import unittest

try:
    import torch

    from icu_mortality.models import (
        BiGRUAttentionClassifier,
        GRUAttentionClassifier,
        GRUDClassifier,
        TripletTransformerClassifier,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch training extra is not installed")
class ModelShapeTests(unittest.TestCase):
    def test_sequence_models(self):
        batch, time, features, static = 4, 48, 34, 8
        values = torch.randn(batch, time, features)
        mask = torch.randint(0, 2, (batch, time, features)).float()
        delta = torch.rand(batch, time, features)
        static_values = torch.randn(batch, static)
        for model in (
            GRUDClassifier(features, static),
            GRUAttentionClassifier(features, static),
            BiGRUAttentionClassifier(features, static),
        ):
            self.assertEqual(model(values, mask, delta, static_values).shape, (batch,))

    def test_triplet_transformer(self):
        batch, tokens, features, static = 4, 32, 34, 8
        feature_ids = torch.randint(0, features, (batch, tokens))
        values = torch.randn(batch, tokens)
        times = torch.rand(batch, tokens)
        padding = torch.zeros(batch, tokens, dtype=torch.bool)
        logits = TripletTransformerClassifier(features, static)(
            feature_ids, values, times, padding, torch.randn(batch, static)
        )
        self.assertEqual(logits.shape, (batch,))
        feature_ids[0] = features + 1
        feature_ids[0, 0] = features
        padding[0] = True
        padding[0, 0] = False
        logits = TripletTransformerClassifier(features, static)(
            feature_ids, values, times, padding, torch.randn(batch, static)
        )
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
