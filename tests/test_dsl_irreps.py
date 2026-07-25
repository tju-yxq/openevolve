import pytest

from equivariant_nas.dsl import DSLValidationError, Irrep, Irreps


def test_o3_irreps_are_parsed_and_canonicalized():
    irreps = Irreps.parse("2x1o + 3x0e + 1x1o", "O3")
    assert str(irreps) == "3x0e+3x1o"
    assert irreps.dimension == 3 + 3 * 3


def test_o3_tensor_product_obeys_degree_and_parity_rules():
    outputs = Irrep(1, -1, "O3").tensor_product(Irrep(2, 1, "O3"))
    assert tuple(str(item) for item in outputs) == ("1o", "2o", "3o")


def test_so3_rejects_parity_suffix():
    with pytest.raises(DSLValidationError) as error:
        Irreps.parse("1x1o", "SO3")
    assert error.value.diagnostics[0].code == "E_IRREP_009"


def test_so2_frequency_product_is_supported():
    outputs = Irrep(2, 1, "SO2").tensor_product(Irrep(1, 1, "SO2"))
    assert tuple(str(item) for item in outputs) == ("m1", "m3")
