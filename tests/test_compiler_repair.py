import asyncio
import json

from equivariant_nas.spec import EvolutionFactor, baseline_spec
from scripts.run_factorized_evolution import generate_compilable_patch


class FakeEnsemble:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate_with_context(self, **_kwargs):
        return self.responses.pop(0)


def response(factor, patch):
    return json.dumps(
        {"selected_factor": factor.value, "reasoning": "test", "patch": patch}
    )


def response_with_reason(factor, patch, reasoning):
    return json.dumps(
        {"selected_factor": factor.value, "reasoning": reasoning, "patch": patch}
    )


def test_schema_failure_is_repaired_within_same_factor():
    parent = baseline_spec()
    invalid = parent.to_dict()["representation"]
    invalid["lmax"] = 1  # tensor channels remain nonzero and must be rejected.
    valid = parent.to_dict()["representation"]
    valid["scalar_channels"] = 96
    ensemble = FakeEnsemble(
        [
            response(EvolutionFactor.REPRESENTATION, invalid),
            response(EvolutionFactor.REPRESENTATION, valid),
        ]
    )
    child, _, _, history = asyncio.run(
        generate_compilable_patch(
            ensemble,
            {"system": "system", "user": "user"},
            parent,
            EvolutionFactor.REPRESENTATION,
            {parent.architecture_id()},
            repair_attempts=1,
        )
    )
    assert child.representation.scalar_channels == 96
    assert len(history) == 1
    assert history[0]["stage"] == "compiler"


def test_scientifically_invalid_reasoning_is_repaired():
    parent = baseline_spec()
    patch = parent.to_dict()["macro"]
    patch["radius"] = 6.0
    ensemble = FakeEnsemble(
        [
            response_with_reason(
                EvolutionFactor.MACRO, patch, "QM9 alpha is a rank-2 tensor target."
            ),
            response_with_reason(
                EvolutionFactor.MACRO,
                patch,
                "Alpha is scalar; the radius hypothesis has no accuracy evidence yet.",
            ),
        ]
    )
    child, reasoning, _, history = asyncio.run(
        generate_compilable_patch(
            ensemble,
            {"system": "system", "user": "user"},
            parent,
            EvolutionFactor.MACRO,
            {parent.architecture_id()},
            repair_attempts=1,
        )
    )
    assert child.macro.radius == 6.0
    assert reasoning.startswith("Alpha is scalar")
    assert len(history) == 1
