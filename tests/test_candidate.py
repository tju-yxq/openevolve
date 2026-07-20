import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from equivariant_nas.candidate import (
    apply_factor_patch,
    extract_literal_spec,
    render_candidate,
)
from equivariant_nas.spec import EvolutionFactor, SpecValidationError, baseline_spec


class CandidateSafetyTests(unittest.TestCase):
    def test_render_and_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(render_candidate(baseline_spec()), encoding="utf-8")
            self.assertEqual(extract_literal_spec(str(path)), baseline_spec())

    def test_executable_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text("import os\nARCHITECTURE_SPEC = {}\n", encoding="utf-8")
            with self.assertRaises(SpecValidationError):
                extract_literal_spec(str(path))

    def test_factor_patch(self):
        parent = baseline_spec()
        patch = parent.to_dict()["macro"]
        patch["num_layers"] = 5
        child = apply_factor_patch(parent, EvolutionFactor.MACRO, patch)
        self.assertEqual(child.macro.num_layers, 5)
        self.assertEqual(parent.changed_factors(child), (EvolutionFactor.MACRO,))


if __name__ == "__main__":
    unittest.main()

