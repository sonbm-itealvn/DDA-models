import unittest
import numpy as np

from ddamodels.canonical import canonicalize, session_relative_normalize
from ddamodels.adaptation import OnlineTemperatureCalibrator
from ddamodels.controller import ControllerState, decide


class AdaptiveComponentTests(unittest.TestCase):
    def test_missing_is_masked_not_zero_semantics(self):
        s = canonicalize({'progress_ratio': .4, 'vitality_ratio': None})
        self.assertEqual(s.mask[0], 1.0)
        self.assertEqual(s.mask[4], 0.0)
        self.assertEqual(s.values[4], 0.0)

    def test_relative_normalization_keeps_mask(self):
        values, mask = session_relative_normalize(np.array([2., 0.]), np.array([1., 0.]), np.array([2., 2.]))
        self.assertAlmostEqual(values[0], 1.0)
        self.assertEqual(values[1], 0.0)
        self.assertEqual(mask[1], 0.0)

    def test_calibrator_updates_only_when_called(self):
        c = OnlineTemperatureCalibrator(batch_size=1)
        self.assertEqual(c.updates, 0)
        c.update(.9, 0)
        self.assertEqual(c.updates, 1)

    def test_controller_holds_on_uncertainty(self):
        d = decide({1:.9, 2:.8, 3:.7}, {1:.5, 2:.5, 3:.5}, 2, ControllerState())
        self.assertEqual(d.applied_difficulty, 2)
        self.assertIn('CONFIDENCE', d.reason)


if __name__ == '__main__':
    unittest.main()
