import json
import unittest

import numpy as np
import pandas as pd

from ddamodels.adaptation import OnlineTemperatureCalibrator
from ddamodels.canonical import CANONICAL_FEATURES, CausalSessionNormalizer
from ddamodels.controller import ControllerState, decide
from ddamodels.data import causal_normalize_canonical_frame, build_tokenized_window_rows
from ddamodels.models import TokenizedFiLMGRU, torch


class HardeningV3Tests(unittest.TestCase):
    def test_affine_calibrator_can_correct_bias(self):
        c = OnlineTemperatureCalibrator(batch_size=20, prior_strength=0.1, lr=.03, bias_lr=.03)
        for _ in range(20):
            c.observe(.7, 1)
        self.assertEqual(c.updates, 1)
        self.assertTrue(-1.0 <= c.bias <= 1.0)
        self.assertTrue(.7 <= c.temperature <= 2.0)
        self.assertGreater(c.transform(.7), .7)

    def test_controller_uses_score_direction_and_bounds(self):
        state = ControllerState()
        kwargs = dict(probabilities={1: .3, 2: .81, 3: .75}, uncertainties={1: .05, 2: .05, 3: .05},
                      current=2, state=state, progress_ratio=.5, min_confidence=.8,
                      increase_threshold=.8, max_difficulty=2)
        d1 = decide(**kwargs)
        d2 = decide(**kwargs)
        self.assertEqual(d1.applied_difficulty, 2)
        self.assertEqual(d2.applied_difficulty, 2)  # upper bound blocks increase

    def test_normalizer_train_serve_order_matches(self):
        game_mean = {"reaction_latency": 1.0}
        game_scale = {"reaction_latency": .2}
        stream = [{"reaction_latency": 1.2}, {"reaction_latency": 1.4}, {"reaction_latency": 1.1}]
        n = CausalSessionNormalizer(game_mean=game_mean, game_scale=game_scale)
        expected = []
        for row in stream:
            out, mask = n.transform(row); expected.append((out["reaction_latency"], mask["reaction_latency"])); n.update(row)
        rows = []
        for i, row in enumerate(stream):
            item = {name: np.nan for name in CANONICAL_FEATURES}
            item.update(row, session_id="s", elapsed_seconds=float(i), segment_success=1,
                        difficulty_vector_json=json.dumps([0, 0, 0, 0, 0]),
                        difficulty_mask_json=json.dumps([1, 1, 1, 1, 0]))
            rows.append(item)
        frame = causal_normalize_canonical_frame(pd.DataFrame(rows), game_mean, game_scale)
        self.assertAlmostEqual(float(frame.iloc[0]["reaction_latency"]), expected[0][0])
        self.assertAlmostEqual(float(frame.iloc[1]["reaction_latency"]), expected[1][0])

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_film_identity_initialization_and_static_forward(self):
        model = TokenizedFiLMGRU(21, 3, 10)
        values = torch.zeros(2, 12, 21)
        masks = torch.ones(2, 12, 21)
        context = torch.zeros(2, 3)
        difficulty = torch.zeros(2, 10)
        output = model(values, masks, context, difficulty)
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.allclose(model.gamma_delta.weight, torch.zeros_like(model.gamma_delta.weight)))

    def test_v2_builder_keeps_explicit_masks(self):
        rows = []
        for i in range(12):
            row = {name: float(i) for name in CANONICAL_FEATURES}
            row.update(session_id="s", elapsed_seconds=float(i), segment_success=1,
                       difficulty_vector_json=json.dumps([0, 0, 0, 0, 0]),
                       difficulty_mask_json=json.dumps([1, 1, 1, 1, 0]))
            if i == 0:
                row["reaction_latency"] = np.nan
            rows.append(row)
        out = build_tokenized_window_rows(pd.DataFrame(rows))
        mask = np.asarray(json.loads(out.iloc[0].mask_json))
        self.assertEqual(mask.shape, (12, 21))
        self.assertEqual(mask[0, CANONICAL_FEATURES.index("reaction_latency")], 0.0)


if __name__ == "__main__":
    unittest.main()
