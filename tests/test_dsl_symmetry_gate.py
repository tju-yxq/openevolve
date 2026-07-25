from equivariant_nas.dsl.pipeline import _symmetry_gate_failed


def test_symmetry_gate_tolerates_float_noise_when_reference_is_near_zero():
    report = {"maximum": 0.05, "maximum_absolute": 2.0e-7}
    assert not _symmetry_gate_failed(report, 1.0e-4, 1.0e-5)


def test_symmetry_gate_rejects_material_absolute_and_relative_violation():
    report = {"maximum": 0.05, "maximum_absolute": 2.0e-4}
    assert _symmetry_gate_failed(report, 1.0e-4, 1.0e-5)
