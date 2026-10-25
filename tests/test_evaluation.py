from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.evaluate import score_cache


class PredictionEvaluationTests(unittest.TestCase):
    def test_equal_frame_weighting_and_exact_manifest_order(self):
        root = Path(__file__).resolve().parents[1] / 'build'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder = Path(folder)
            np.savez(folder / 'a.npz', gt=np.zeros((2, 2)), mask=np.array([[1, 0], [0, 0]], bool))
            np.savez(folder / 'b.npz', gt=np.zeros((2, 2)), mask=np.ones((2, 2), bool))
            np.savez(folder / 'p.npz', sample_ids=np.array(['a', 'b']),
                     predictions=np.array([np.ones((2, 2)), np.full((2, 2), 3.)]))
            samples = [dict(sample_id=sid, gt_path=str(folder / f'{sid}.npz')) for sid in ('a', 'b')]
            rows = score_cache(folder / 'p.npz', samples, 20)
            self.assertEqual(np.mean([r['MAE_cm'] for r in rows]), 2.)
            with self.assertRaises(ValueError):
                score_cache(folder / 'p.npz', samples[::-1], 20)


if __name__ == '__main__':
    unittest.main()
