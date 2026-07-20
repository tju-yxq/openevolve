import unittest

from equivariant_nas.router import EvidenceCalibratedRouter, prompt_for_factor
from equivariant_nas.spec import EvolutionFactor


class RouterTests(unittest.TestCase):
    def test_untried_factors_are_explored(self):
        router = EvidenceCalibratedRouter(seed=1)
        selected = {router.select() for _ in range(20)}
        # Selection alone does not update attempts, so every factor remains tied.
        self.assertGreaterEqual(len(selected), 3)

    def test_invalid_factor_is_penalized(self):
        router = EvidenceCalibratedRouter(seed=2, exploration=0.0)
        for factor in EvolutionFactor:
            router.update(factor, valid=True, mae_gain=0.0)
        for _ in range(5):
            router.update(EvolutionFactor.ACTION, valid=False)
        self.assertNotEqual(router.select(), EvolutionFactor.ACTION)

    def test_runtime_context_routes_toward_cost_factors_after_initialization(self):
        router = EvidenceCalibratedRouter(seed=1, exploration=0.0)
        for factor in EvolutionFactor:
            router.update(factor, valid=True)
        selected = router.select(
            {"relative_step_time": 1.2, "parameter_ratio": 1.0}
        )
        self.assertIn(selected, {EvolutionFactor.OPERATOR, EvolutionFactor.MACRO})

    def test_prompt_states_scalar_physics_of_qm9_alpha(self):
        prompt = prompt_for_factor(
            EvolutionFactor.MACRO, "{}", {}, [], {}, reflection={}
        )
        self.assertIn("scalar isotropic polarizability", prompt["system"])
        self.assertIn("not a rank-2 tensor target", prompt["system"])
        self.assertIn("scalar output does not imply l>0", prompt["system"])
        self.assertIn("not the output head", prompt["system"])


if __name__ == "__main__":
    unittest.main()
