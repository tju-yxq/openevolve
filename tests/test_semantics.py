import pytest

from equivariant_nas.semantics import ScientificSemanticsError, validate_qm9_alpha_reasoning


@pytest.mark.parametrize(
    "text",
    [
        "QM9 alpha is a rank-2 tensor target.",
        "head_tensor_channels is an output head.",
        "Higher-order irreps cannot contribute to the scalar target.",
    ],
)
def test_physical_misconceptions_are_rejected(text):
    with pytest.raises(ScientificSemanticsError):
        validate_qm9_alpha_reasoning(text)


def test_correct_scalar_and_hidden_irrep_statement_passes():
    validate_qm9_alpha_reasoning(
        "Alpha is not a rank-2 tensor target; hidden l>0 irreps may contribute through coupling into l=0."
    )
