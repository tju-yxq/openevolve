# Trusted evaluator parses this literal without executing code.
ARCHITECTURE_SPEC = {
    "schema_version": 1,
    "representation": {
        "lmax": 2,
        "scalar_channels": 128,
        "vector_channels": 64,
        "tensor_channels": 32,
        "l3_channels": 0,
        "head_scalar_channels": 32,
        "head_vector_channels": 16,
        "head_tensor_channels": 8,
        "head_l3_channels": 0,
        "mlp_multiplier": 3,
        "feature_channels": 512,
    },
    "operator": {
        "basis_type": "bessel",
        "num_basis": 64,
        "radial_hidden": [64, 64],
        "nonlinear_message": True,
        "num_heads": 4,
    },
    "action": {
        "norm_layer": "layer",
        "rescale_degree": False,
        "alpha_drop": 0.2,
        "projection_drop": 0.0,
        "output_drop": 0.0,
        "drop_path": 0.0,
    },
    "macro": {"num_layers": 6, "radius": 5.0},
}
