import unittest

import numpy as np

from scripts.analysis import frame_metrics, paired_seed_summary, shape_metrics


class MetricContractTests(unittest.TestCase):
    def test_strict_threshold_and_missing_cells(self):
        gt = np.array([[0., 0., np.nan], [0., 0., 99.]])
        pred = np.array([[.5, 1., 0.], [0., -.5, 0.]])
        mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
        result = frame_metrics(gt, pred, mask)
        self.assertEqual(result['valid_cells'], 4)
        self.assertEqual(result['MAE_cm'], .5)
        self.assertEqual(result['GT0p5_pct'], 25.)
        self.assertAlmostEqual(result['RMSE_cm'], np.sqrt(.375))

    def test_invalid_prediction_fails_instead_of_dropping_cell(self):
        with self.assertRaises(ValueError):
            frame_metrics(np.zeros((2, 2)), np.full((2, 2), np.nan), np.ones((2, 2), bool))

    def test_average_conditions_within_each_seed_before_sd(self):
        rows = [dict(seed=s, condition=c, value=v) for s, c, v in
                [(1, '-1', 0), (1, '+1', 2), (2, '-1', 2), (2, '+1', 4),
                 (3, '-1', 4), (3, '+1', 6)]]
        mean, sd, per_seed = paired_seed_summary(rows, 'value')
        self.assertEqual(mean, 3.)
        self.assertEqual(sd, 2.)
        self.assertEqual(per_seed, {1: 1., 2: 3., 3: 5.})

    def test_missing_sign_cannot_silently_change_condition_weights(self):
        with self.assertRaises(ValueError):
            paired_seed_summary([dict(seed=1, condition='a', value=1),
                                 dict(seed=1, condition='b', value=2),
                                 dict(seed=2, condition='a', value=3)], 'value')

    def test_plane_offset_is_removed_but_relief_loss_is_detected(self):
        z, x = np.indices((8, 7))
        relief = np.zeros((8, 7)); relief[3:5, 2:5] = 3
        gt = x * .2 + z * .4 + relief
        mask = np.ones(gt.shape, bool)
        exact = shape_metrics(gt, gt + 5, mask)
        flat = shape_metrics(gt, x * .2 + z * .4, mask)
        self.assertAlmostEqual(exact['deplaned_MAE_cm'], 0.)
        self.assertAlmostEqual(exact['relief_recovery_ratio'], 1.)
        self.assertGreater(flat['deplaned_MAE_cm'], .1)
        self.assertAlmostEqual(flat['relief_recovery_ratio'], 0.)


if __name__ == '__main__':
    unittest.main()
