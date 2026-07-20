import json
import unittest
from dataclasses import replace

from equivariant_nas.spec import (
    ArchitectureSpec,
    EvolutionFactor,
    SpecValidationError,
    baseline_spec,
)


class ArchitectureSpecTests(unittest.TestCase):
    def test_baseline_matches_official_irreps(self):
        spec = baseline_spec()
        self.assertEqual(spec.representation.embedding_irreps(), "128x0e+64x1e+32x2e")
        self.assertEqual(spec.representation.head_irreps(), "32x0e+16x1e+8x2e")
        self.assertEqual(spec.representation.mlp_irreps(), "384x0e+192x1e+96x2e")

    def test_json_roundtrip_and_hash(self):
        spec = baseline_spec()
        restored = ArchitectureSpec.from_json(spec.canonical_json())
        self.assertEqual(spec, restored)
        self.assertEqual(spec.architecture_id(), restored.architecture_id())

    def test_factor_local_change(self):
        parent = baseline_spec()
        child = replace(parent, macro=replace(parent.macro, num_layers=5))
        parent.assert_factor_local_change(child, EvolutionFactor.MACRO)
        with self.assertRaises(SpecValidationError):
            parent.assert_factor_local_change(child, EvolutionFactor.OPERATOR)

    def test_unknown_fields_are_rejected(self):
        data = baseline_spec().to_dict()
        data["training_steps"] = 1
        with self.assertRaises(SpecValidationError):
            ArchitectureSpec.from_dict(data)

    def test_channels_above_lmax_are_rejected(self):
        data = baseline_spec().to_dict()
        data["representation"]["lmax"] = 1
        with self.assertRaises(SpecValidationError):
            ArchitectureSpec.from_dict(data)


if __name__ == "__main__":
    unittest.main()

