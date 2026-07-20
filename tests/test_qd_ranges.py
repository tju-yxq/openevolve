from types import SimpleNamespace

from scripts.run_factorized_evolution import initialize_equivariant_feature_ranges


def test_equivariant_map_ranges_are_fixed_before_insertion():
    database = SimpleNamespace(feature_stats={"old": {}})
    initialize_equivariant_feature_ranges(database)
    assert database.feature_stats["lmax"]["min"] == 1.0
    assert database.feature_stats["lmax"]["max"] == 3.0
    assert database.feature_stats["parameter_ratio"]["max"] == 1.2
    assert set(database.feature_stats) == {
        "lmax",
        "higher_order_fraction",
        "parameter_ratio",
        "num_layers",
    }
