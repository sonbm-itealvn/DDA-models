import unittest
import numpy as np

from ddamodels.canonical import (CausalSessionNormalizer, build_difficulty_vector,
                                 normalized_log_ratio_delta)
from ddamodels.adaptation import OnlineTemperatureCalibrator
from ddamodels.segments import (SegmentContract, SegmentMode, close_fixed_horizon_segment,
                                segment_boundary)
from ddamodels.controller import ControllerState, decide


class V2HardeningTests(unittest.TestCase):
    def test_bounded_ratio_is_not_session_z_scored(self):
        n = CausalSessionNormalizer(game_mean={'tempo_rate': 1}, game_scale={'tempo_rate': 1})
        out, mask = n.transform({'vitality_ratio': .95, 'time_budget_ratio': 3.0})
        self.assertAlmostEqual(out['vitality_ratio'], .95)
        self.assertTrue(np.isfinite(out['time_budget_ratio']))
        self.assertEqual(mask['vitality_ratio'], 1.0)

    def test_unbounded_shrinkage_has_scale_floor(self):
        n = CausalSessionNormalizer(game_mean={'reaction_latency': 1}, game_scale={'reaction_latency': 0})
        out, _ = n.transform({'reaction_latency': 1.2})
        self.assertTrue(np.isfinite(out['reaction_latency']))
        self.assertLessEqual(abs(out['reaction_latency']), 8)

    def test_prior_is_masked_without_fixed_randomized_source(self):
        values, mask = build_difficulty_vector(1, .2, 0, -.1, empirical_prior=.8, prior_source='dda')
        self.assertEqual(values[-1], 0.0)
        self.assertEqual(mask[-1], 0.0)

    def test_delta_contract(self):
        self.assertAlmostEqual(normalized_log_ratio_delta(1.1, 1.0, 1.25), np.log(1.1) / np.log(1.25))
        self.assertEqual(normalized_log_ratio_delta(2.0, 1.0, 1.25), 1.0)
        self.assertAlmostEqual(normalized_log_ratio_delta(0.0, 0.0, 1.25), 0.0)

    def test_batch_calibrator_does_not_update_per_wave(self):
        c = OnlineTemperatureCalibrator(batch_size=3)
        for i in range(2):
            self.assertFalse(c.observe(.9, 0))
        self.assertEqual(c.updates, 0)
        self.assertTrue(c.observe(.9, 0))
        self.assertEqual(c.updates, 1)
        self.assertGreaterEqual(c.temperature, .7)
        self.assertLessEqual(c.temperature, 2.0)

    def test_segment_contracts(self):
        SegmentContract(SegmentMode.EVENT_BOUNDED, 45).validate()
        SegmentContract(SegmentMode.FIXED_HORIZON, 30).validate()
        self.assertEqual(segment_boundary(SegmentMode.FIXED_HORIZON, 10, 30), (10, 40))
        self.assertEqual(close_fixed_horizon_segment(10, 30, 20), (10, 20, True))

    def test_controller_cooldown_is_two_horizons(self):
        state = ControllerState()
        kwargs = dict(probabilities={1:.9, 2:.9, 3:.9}, uncertainties={1:0, 2:0, 3:0},
                      current=2, state=state, struggle_probability=1.0,
                      min_confidence=0.0, cooldown_seconds=10, horizon_seconds=30)
        decide(**kwargs); decision = decide(**kwargs)
        self.assertEqual(decision.applied_difficulty, 1)
        self.assertGreaterEqual(state.cooldown_seconds, 60)


if __name__ == '__main__':
    unittest.main()
