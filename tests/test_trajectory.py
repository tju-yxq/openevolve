import unittest

from equivariant_nas.trajectory import response_fingerprint


class TrajectoryTests(unittest.TestCase):
    def test_lr_shock_is_detected(self):
        result = response_fingerprint({300: 2.0, 1000: 4.0, 5000: 1.0})
        self.assertEqual(result.lr_shock_ratio, 2.0)
        self.assertEqual(result.warmup_recovery_ratio, 0.25)
        self.assertGreater(result.early_to_warmup_rank_risk, 0.0)


if __name__ == "__main__":
    unittest.main()

