import tempfile
import unittest
from pathlib import Path

from equivariant_nas.fidelity import FidelityObservation, SuccessiveHalvingScheduler


class FidelityTests(unittest.TestCase):
    def test_same_fidelity_ranking_and_promotion(self):
        scheduler = SuccessiveHalvingScheduler()
        for index, mae in enumerate((0.8, 0.4, 0.6, 0.2, 0.9, 0.5, 0.7, 0.3)):
            scheduler.record(
                FidelityObservation(
                    architecture_id="a{}".format(index),
                    level="micro",
                    validation_alpha_mae=mae,
                    seed=0,
                )
            )
        self.assertEqual(scheduler.promotion_candidates("micro"), ("a3", "a7"))

    def test_roundtrip(self):
        scheduler = SuccessiveHalvingScheduler()
        scheduler.record(FidelityObservation("a", "micro", 1.0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            scheduler.save(str(path))
            restored = SuccessiveHalvingScheduler.load(str(path))
            self.assertEqual(restored.observations[0].architecture_id, "a")


if __name__ == "__main__":
    unittest.main()

