from types import SimpleNamespace

from equivariant_nas.search_memory import compact_metrics, summarize_lineage


def program(identifier, parent_id, mae, factor=""):
    metrics = {} if mae is None else {"validation_alpha_mae": mae}
    metadata = {} if not factor else {"selected_factor": factor}
    return SimpleNamespace(
        id=identifier,
        parent_id=parent_id,
        metrics=metrics,
        metadata=metadata,
    )


def test_lineage_memory_detects_measured_plateau():
    nodes = {
        "a": program("a", None, 0.80),
        "b": program("b", "a", 0.71, "OPERATOR"),
        "c": program("c", "b", 0.705, "ACTION"),
    }
    memory = summarize_lineage(nodes["c"], nodes.get, plateau_absolute_gain=0.02)
    assert memory.plateau_status == "yes"
    assert memory.mutation_regime == "exploratory_within_factor"
    assert memory.validation_mae_history == [0.80, 0.71, 0.705]


def test_zero_step_lineage_never_fabricates_plateau():
    node = program("a", None, None)
    memory = summarize_lineage(node, lambda _: None)
    assert memory.plateau_status == "unknown"
    assert memory.best_validation_mae is None


def test_prompt_metrics_drop_large_reports_and_keep_scientific_evidence():
    compact = compact_metrics(
        {
            "validation_alpha_mae": 0.7,
            "parameter_ratio": 0.9,
            "symmetry_report": {"large": [1, 2, 3]},
        }
    )
    assert compact == {"validation_alpha_mae": 0.7, "parameter_ratio": 0.9}
