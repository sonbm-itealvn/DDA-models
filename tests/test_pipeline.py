import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ddamodels.data import load_window_dataset
from ddamodels.models import MajorityModel, NumpyLogisticRegression
from ddamodels.train import group_split, metrics


class PipelineTests(unittest.TestCase):
    def test_group_split_has_no_overlap(self):
        groups = np.array(['a'] * 4 + ['b'] * 4 + ['c'] * 4 + ['d'] * 4 + ['e'] * 4 + ['f'] * 4)
        tr, va, te, manifest = group_split(groups, seed=1)
        self.assertEqual(set(groups[tr]) & set(groups[va]), set())
        self.assertEqual(set(groups[tr]) & set(groups[te]), set())
        self.assertEqual(set(groups[va]) & set(groups[te]), set())

    def test_numpy_logistic_learns_simple_signal(self):
        x = np.array([[-2], [-1], [1], [2]], dtype=float)
        y = np.array([0, 0, 1, 1], dtype=float)
        model = NumpyLogisticRegression(epochs=1000, lr=.1).fit(x, y)
        self.assertGreater(model.predict_proba(np.array([[2.0]]))[0], .8)

    def test_window_dataset_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'windows.jsonl'
            rows = []
            for i in range(4):
                rows.append({'session_id': f's{i}', 'player_id_hash': f'p{i}',
                             'history_json': json.dumps([[0.0, 1.0]] * 12),
                             'context_json': json.dumps([0.0, 1.0]),
                             'candidate_difficulty': 2, 'segment_success': i % 2})
            p.write_text('\n'.join(json.dumps(x) for x in rows), encoding='utf-8')
            ds = load_window_dataset(p)
            self.assertEqual(ds.history.shape, (4, 12, 2))
            self.assertEqual(ds.context.shape, (4, 2))

    def test_metrics_are_finite(self):
        result = metrics(np.array([0, 1]), np.array([.2, .8]))
        self.assertTrue(all(np.isfinite(v) for v in result.values()))


if __name__ == '__main__':
    unittest.main()
