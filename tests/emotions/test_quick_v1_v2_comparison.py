from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import yaml

from emotions.gnn_improvement_experiments import run_quick_v1_v2_comparison as quick_comparison
from emotions.gnn_improvement_experiments.run_conv_type_comparison import run_conv_type_comparison
from emotions.gnn_improvement_experiments.run_quick_v1_v2_comparison import (
    AROUSAL_EXPERIMENT_ID,
    VALENCE_EXPERIMENT_ID,
    build_fixed_overrides,
    build_quick_runs,
    build_signal_set_variant,
    build_variant,
    _build_payload,
    _model_names_for_signal_set,
    _ordered_models,
    _parse_cv_strategies,
    _parse_models,
    _parse_signal_sets,
    _resolve_report_profile,
    _resolve_requested_models,
    _resolve_requested_signal_sets,
    _save_thesis_metric_heatmaps,
    _save_thesis_metric_table,
    _save_combined_confusion_matrices,
    _save_confusion_matrix_table,
    _save_fold_metric_outputs,
    _save_group_model_ranking,
    _save_label_distribution_outputs,
    _save_model_benchmark_outputs,
    _save_training_history_outputs,
    _should_save_fold_loss_plots,
    _thesis_model_display_name,
    _plot_test_loss_summary,
    parse_args,
    run_quick_comparison,
)


def _args() -> Namespace:
    return Namespace(
        output_root="results/test_quick",
        seed=None,
        cv_strategy=None,
        n_splits=None,
        val_size=None,
        num_epochs=None,
        use_torch_compile=False,
        signal_sets=None,
        report_profile="thesis",
        save_fold_loss_plots=False,
    )


def test_quick_variants_set_expected_gnn_versions() -> None:
    basic = build_variant("BasicGCN")
    hetero_mean = build_variant("HeteroGCNMean")
    hetero_mlp = build_variant("HeteroGCNMLP")
    hetero_weighted = build_variant("HeteroGCNMLPWeights")

    assert basic.overrides["global_overrides"]["dataset"]["graph_version"] == "v2"
    assert basic.overrides["global_overrides"]["dataset"]["edge_weight_mode"] == "learned_signed"
    assert basic.overrides["global_overrides"]["dataset"]["use_edge_weights"] is True
    assert basic.overrides["global_overrides"]["gnn"]["model"]["model_version"] == "BasicGCN"
    assert hetero_mean.overrides["global_overrides"]["gnn"]["model"]["model_version"] == "HeteroGCNMean"
    assert hetero_mlp.overrides["global_overrides"]["gnn"]["model"]["model_version"] == "HeteroGCNMLP"
    assert hetero_weighted.overrides["global_overrides"]["gnn"]["model"]["model_version"] == "HeteroGCNMLPWeights"


def test_quick_gnn_variants_do_not_override_shared_architecture_knobs() -> None:
    shared_knobs = {
        "conv_type",
        "hidden_channels",
        "num_layers",
        "pooling",
        "head_pooling",
        "graph_pooling",
        "relation_pooling",
    }

    for model_name in ["BasicGCN", "HeteroGCNMean", "HeteroGCNMLP", "HeteroGCNMLPWeights"]:
        model_cfg = build_variant(model_name).overrides["global_overrides"]["gnn"]["model"]

        assert not (shared_knobs & set(model_cfg))


def test_quick_payload_defaults_to_table6_valence_only() -> None:
    base_cfg = {
        "experiments": {
            AROUSAL_EXPERIMENT_ID: {"enabled": True},
            VALENCE_EXPERIMENT_ID: {"enabled": False},
        }
    }
    payload = _build_payload(
        base_cfg=base_cfg,
        fixed_overrides=build_fixed_overrides(_args()),
        variant=build_variant("LightGBM"),
    )

    assert payload["experiments"][AROUSAL_EXPERIMENT_ID]["enabled"] is False
    assert payload["experiments"][VALENCE_EXPERIMENT_ID]["enabled"] is True
    assert payload["global_overrides"]["baselines"]["models"] == ["LightGBM"]
    assert payload["global_overrides"]["run_experiments"] == {"baselines": True, "gnn": False}


def test_quick_variants_support_random_and_majority() -> None:
    random_variant = build_variant("Random")
    majority_variant = build_variant("Majority")

    assert random_variant.overrides["global_overrides"]["baselines"]["models"] == ["Random"]
    assert majority_variant.overrides["global_overrides"]["baselines"]["models"] == ["Majority"]
    assert random_variant.summary_model_name == "Random"
    assert majority_variant.summary_model_name == "Majority"


def test_quick_model_parser_accepts_common_aliases() -> None:
    parsed = _parse_models("random,majority,basicgcn,hetero_gcn_mean,hetero-gcn-mlp,heterogcn_mlp_weights,lgbm")

    assert parsed == [
        "Random",
        "Majority",
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
        "LightGBM",
    ]


def test_quick_model_parser_accepts_yaml_list() -> None:
    parsed = _parse_models(["lightgbm", "svm", "mlp", "gazemae_mlp", "basicgcn", "heterogcnmlpweights"])

    assert parsed == ["LightGBM", "SVM", "MLP", "GazeMAE_MLP", "BasicGCN", "HeteroGCNMLPWeights"]


def test_quick_signal_set_parser_accepts_aliases_and_deduplicates() -> None:
    parsed = _parse_signal_sets("gaze,gaze-only,pupil,gaze+pupil,all,full")

    assert parsed == ["gaze_only", "pupil_only", "gaze_pupil", "all_signals"]


def test_quick_signal_set_parser_rejects_invalid_names() -> None:
    with pytest.raises(ValueError, match="Unknown signal set"):
        _parse_signal_sets("gaze,blink_only")


def test_quick_signal_sets_default_to_yaml_then_all_groups() -> None:
    wrapper_cfg = {"quick_comparison": {"signal_sets": ["gaze", "pupil-only"]}}

    assert _resolve_requested_signal_sets(wrapper_cfg, cli_signal_sets=None) == ["gaze_only", "pupil_only"]
    assert _resolve_requested_signal_sets(wrapper_cfg, cli_signal_sets="all") == ["all_signals"]
    assert _resolve_requested_signal_sets({}, cli_signal_sets=None) == [
        "gaze_only",
        "pupil_only",
        "gaze_pupil",
        "all_signals",
    ]


def test_report_profile_parser_and_fold_loss_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_quick_v1_v2_comparison.py",
            "--report-profile",
            "full",
            "--save-fold-loss-plots",
        ],
    )
    args = parse_args()

    assert args.report_profile == "full"
    assert args.save_fold_loss_plots is True
    assert _resolve_report_profile(args) == "full"
    assert _should_save_fold_loss_plots(args, "thesis") is True
    assert _should_save_fold_loss_plots(Namespace(save_fold_loss_plots=False), "full") is True
    assert _should_save_fold_loss_plots(Namespace(save_fold_loss_plots=False), "thesis") is False


def test_signal_set_overrides_force_temporal_flags_and_active_sources() -> None:
    pupil = build_signal_set_variant("pupil_only")
    dataset_cfg = pupil.overrides["global_overrides"]["dataset"]
    model_cfg = pupil.overrides["global_overrides"]["gnn"]["model"]

    assert dataset_cfg["use_relative_time"] is True
    assert dataset_cfg["use_temporal_node_feature"] is True
    assert dataset_cfg["use_temporal_edge_features"] is True
    assert dataset_cfg["use_temporal_edges"] is True
    assert dataset_cfg["use_gaze_node_features"] is False
    assert dataset_cfg["use_pupil_node_features"] is True
    assert dataset_cfg["use_pupil_edge_features"] is True
    assert dataset_cfg["use_spatial_edges"] is False
    assert dataset_cfg["use_fixation_edges"] is False
    assert dataset_cfg["dropna_columns"] == [
        "time-rel-seconds",
        "subject",
        "recording",
        "pupil-size-left-avg",
        "pupil-size-right-avg",
    ]
    assert dataset_cfg["signal_outlier_filter"]["columns"] == [
        "pupil-size-left-avg",
        "pupil-size-right-avg",
    ]
    assert model_cfg["use_spatial_edges"] is False
    assert model_cfg["use_fixation_edges"] is False


def test_signal_aware_embedding_mapping_emits_one_embedding_baseline() -> None:
    requested = ["LightGBM", "GazeMAE_MLP", "MOMENT_pupil", "BasicGCN"]

    assert _model_names_for_signal_set(requested, "gaze_only") == ["LightGBM", "GazeMAE_MLP", "BasicGCN"]
    assert _model_names_for_signal_set(requested, "pupil_only") == ["LightGBM", "MOMENT_pupil", "BasicGCN"]
    assert _model_names_for_signal_set(requested, "gaze_pupil") == [
        "LightGBM",
        "MOMENT_GazeMAE_gaze_pupil",
        "BasicGCN",
    ]
    assert _model_names_for_signal_set(requested, "all_signals") == [
        "LightGBM",
        "MOMENT_GazeMAE_all_signals",
        "BasicGCN",
    ]


def test_quick_dry_run_writes_signal_set_rows_and_nested_configs(tmp_path: Path) -> None:
    base_config = tmp_path / "wrapper.yaml"
    output_dir = tmp_path / "quick_output"
    payload = {
        "suite": {},
        "global_overrides": {
            "cross_validation": {"strategies": ["subject_kfold"]},
        },
        "experiments": {
            AROUSAL_EXPERIMENT_ID: {"enabled": False},
            VALENCE_EXPERIMENT_ID: {"enabled": True},
        },
        "quick_comparison": {
            "models": ["LightGBM", "GazeMAE_MLP", "BasicGCN"],
            "signal_sets": ["pupil_only", "all_signals"],
        },
    }
    base_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    args = Namespace(
        base_config=str(base_config),
        output_root=str(tmp_path / "unused"),
        models=None,
        signal_sets=None,
        seed=None,
        cv_strategy=None,
        n_splits=None,
        val_size=None,
        num_epochs=None,
        use_torch_compile=False,
        report_profile="thesis",
        save_fold_loss_plots=False,
        dry_run=True,
    )

    run_quick_comparison(args, output_dir=output_dir)

    summary = pd.read_csv(output_dir / "quick_comparison_summary.csv")
    assert set(summary["signal_set"]) == {"pupil_only", "all_signals"}
    assert "signal_set_description" in summary.columns
    assert (output_dir / "generated_wrapper_configs" / "pupil_only" / "Models.yaml").exists()
    assert (output_dir / "generated_wrapper_configs" / "all_signals" / "Models.yaml").exists()

    pupil_cfg = yaml.safe_load(
        (output_dir / "generated_wrapper_configs" / "pupil_only" / "Models.yaml").read_text(
            encoding="utf-8"
        )
    )
    all_cfg = yaml.safe_load(
        (output_dir / "generated_wrapper_configs" / "all_signals" / "Models.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert pupil_cfg["suite"]["results_dir"].endswith("model_runs/pupil_only")
    assert all_cfg["suite"]["results_dir"].endswith("model_runs/all_signals")
    assert pupil_cfg["global_overrides"]["baselines"]["models"] == ["LightGBM", "MOMENT_pupil"]
    assert all_cfg["global_overrides"]["baselines"]["models"] == ["LightGBM", "MOMENT_GazeMAE_all_signals"]
    assert pupil_cfg["global_overrides"]["gnn"]["models"] == ["BasicGCN"]
    assert all_cfg["global_overrides"]["gnn"]["models"] == ["BasicGCN"]


def test_conv_type_comparison_dry_run_writes_variant_metadata(tmp_path: Path) -> None:
    base_config = tmp_path / "wrapper.yaml"
    payload = {
        "suite": {},
        "global_overrides": {
            "cross_validation": {"strategies": ["subject_kfold"]},
        },
        "experiments": {
            AROUSAL_EXPERIMENT_ID: {"enabled": False},
            VALENCE_EXPERIMENT_ID: {"enabled": True},
        },
        "quick_comparison": {
            "models": ["BasicGCN", "HeteroGCNMLPWeights"],
            "signal_sets": ["gaze_pupil"],
            "table6_tasks": ["valence"],
        },
    }
    base_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    args = Namespace(
        base_config=str(base_config),
        output_root=str(tmp_path / "conv_type_output"),
        seed=None,
        n_splits=7,
        val_size=None,
        num_epochs=None,
        cv_strategy="subject_kfold",
        only_variant="HeteroGCNMLPWeights_GINEConv",
        use_torch_compile=False,
        enable_benchmarking=False,
        dry_run=True,
    )

    output_dir = run_conv_type_comparison(args)

    summary = pd.read_csv(output_dir / "conv_type_comparison_summary.csv")
    assert summary.loc[0, "variant_id"] == "HeteroGCNMLPWeights_GINEConv"
    assert summary.loc[0, "architecture"] == "HeteroGCNMLPWeights"
    assert summary.loc[0, "conv_type"] == "GINEConv"
    assert summary.loc[0, "edge_info_mode"] == "edge_attr_plus_relation_mlp_scalar_weight"
    assert summary.loc[0, "signal_set"] == "gaze_pupil"
    assert summary.loc[0, "status"] == "dry_run"

    config_path = output_dir / "generated_wrapper_configs" / "gaze_pupil" / "HeteroGCNMLPWeights_GINEConv.yaml"
    generated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert generated["global_overrides"]["gnn"]["models"] == ["HeteroGCNMLPWeights"]
    assert generated["global_overrides"]["gnn"]["model"]["model_version"] == "HeteroGCNMLPWeights"
    assert generated["global_overrides"]["gnn"]["model"]["conv_type"] == "GINEConv"
    assert generated["global_overrides"]["dataset"]["use_gaze_node_features"] is True
    assert generated["global_overrides"]["dataset"]["use_pupil_node_features"] is True
    assert generated["global_overrides"]["benchmarking"]["enabled"] is False
    assert (output_dir / "variant_manifest.csv").exists()


def test_quick_model_parser_accepts_moment_embedding_aliases() -> None:
    parsed = _parse_models(
        "moment_gaze,moment_pupil,moment_gaze_pupil,moment_all_signals,"
        "moment_gazemae_gaze_pupil,moment_gazemae_all_signals"
    )

    assert parsed == [
        "MOMENT_gaze",
        "MOMENT_pupil",
        "MOMENT_gaze_pupil",
        "MOMENT_all_signals",
        "MOMENT_GazeMAE_gaze_pupil",
        "MOMENT_GazeMAE_all_signals",
    ]


def test_quick_models_default_to_yaml_and_allow_cli_override() -> None:
    wrapper_cfg = {
        "quick_comparison": {
            "models": [
                "LightGBM",
                "SVM",
                "MLP",
                "GazeMAE_MLP",
                "BasicGCN",
                "HeteroGCNMean",
                "HeteroGCNMLP",
                "HeteroGCNMLPWeights",
            ],
        }
    }

    assert _resolve_requested_models(wrapper_cfg, cli_models=None) == [
        "LightGBM",
        "SVM",
        "MLP",
        "GazeMAE_MLP",
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
    ]
    assert _resolve_requested_models(wrapper_cfg, cli_models="random,lgbm") == ["Random", "LightGBM"]


def test_quick_cv_parser_accepts_multiple_strategies() -> None:
    parsed = _parse_cv_strategies("subject_loo,recording_loo,subject_kfold,recording_kfold")

    assert parsed == ["subject_loo", "recording_loo", "subject_kfold", "recording_kfold"]


def test_quick_plot_model_order_keeps_sanity_baselines_first() -> None:
    ordered = _ordered_models(
        [
            "LightGBM",
            "HeteroGCNMLPWeights",
            "Random",
            "MLP",
            "Majority",
            "SVM",
            "HeteroGCNMLP",
            "BasicGCN",
            "HeteroGCNMean",
        ]
    )

    assert ordered == [
        "Random",
        "Majority",
        "SVM",
        "LightGBM",
        "MLP",
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
    ]


def test_thesis_model_display_names_are_plot_friendly() -> None:
    assert _thesis_model_display_name("Random") == "Naključni"
    assert _thesis_model_display_name("Majority") == "Večinski"
    assert _thesis_model_display_name("BasicGCN") == "GCN"
    assert _thesis_model_display_name("HeteroGCNMean") == "HeteroGCN-mean"
    assert _thesis_model_display_name("HeteroGCNMLP") == "HeteroGCN-MLP"
    assert _thesis_model_display_name("HeteroGCNMLPWeights") == "HeteroGCN-MLP-w"
    assert _thesis_model_display_name("GazeMAE_MLP") == "GazeMAE+MLP"


def test_quick_runs_group_baselines_into_one_suite_invocation() -> None:
    runs = build_quick_runs(
        ["Random", "Majority", "BasicGCN", "HeteroGCNMean", "HeteroGCNMLP", "HeteroGCNMLPWeights", "LightGBM"]
    )

    assert [run.run_name for run in runs] == ["Models"]
    assert runs[0].model_names == [
        "Random",
        "Majority",
        "LightGBM",
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
    ]
    assert runs[0].overrides["global_overrides"]["baselines"]["models"] == [
        "Random",
        "Majority",
        "LightGBM",
    ]
    assert runs[0].overrides["global_overrides"]["gnn"]["models"] == [
        "BasicGCN",
        "HeteroGCNMean",
        "HeteroGCNMLP",
        "HeteroGCNMLPWeights",
    ]
    assert runs[0].summary_model_names["HeteroGCNMLPWeights"] == "HeteroGCNMLPWeights"


def test_quick_runs_group_basic_gcn_with_baselines_when_it_is_the_only_gnn() -> None:
    runs = build_quick_runs(["Random", "BasicGCN", "LightGBM"])

    assert [run.run_name for run in runs] == ["Models"]
    assert runs[0].model_names == ["Random", "LightGBM", "BasicGCN"]
    assert runs[0].summary_model_names["BasicGCN"] == "BasicGCN"
    assert runs[0].overrides["global_overrides"]["gnn"]["models"] == ["BasicGCN"]
    assert runs[0].overrides["global_overrides"]["gnn"]["model"]["model_version"] == "BasicGCN"


def test_fixed_overrides_can_target_command_output_dir() -> None:
    output_dir = Path("results/test_quick/command_timestamp")
    overrides = build_fixed_overrides(_args(), run_output_dir=output_dir)

    assert overrides["suite"]["results_dir"] == str(output_dir / "model_runs")
    assert overrides["global_overrides"]["gnn"]["training"]["use_torch_compile"] is False


def test_fixed_overrides_only_include_explicit_cli_overrides() -> None:
    args = Namespace(
        output_root="results/test_quick",
        seed=7,
        cv_strategy="subject_kfold,recording_kfold",
        n_splits=2,
        val_size=1,
        num_epochs=3,
        use_torch_compile=False,
    )
    overrides = build_fixed_overrides(args)

    assert overrides["suite"]["seed"] == 7
    assert overrides["global_overrides"]["cross_validation"] == {
        "random_state": 7,
        "strategies": ["subject_kfold", "recording_kfold"],
        "n_splits": 2,
        "val_size": 1,
    }
    assert overrides["global_overrides"]["gnn"]["training"]["num_epochs"] == 3
    assert overrides["global_overrides"]["gnn"]["training"]["use_torch_compile"] is False


def test_fixed_overrides_can_explicitly_enable_torch_compile() -> None:
    args = _args()
    args.use_torch_compile = True

    overrides = build_fixed_overrides(args)

    assert overrides["global_overrides"]["gnn"]["training"]["use_torch_compile"] is True


def test_group_model_ranking_plot_is_written(tmp_path: Path, monkeypatch) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "HeteroGCNMLPWeights",
                "status": "success",
                "accuracy": 0.39,
                "balanced_accuracy": 0.41,
                "macro_f1": 0.37,
                "weighted_f1": 0.38,
                "auc": 0.52,
            },
            {
                "model": "Random",
                "status": "success",
                "accuracy": 0.27,
                "balanced_accuracy": 0.29,
                "macro_f1": 0.26,
                "weighted_f1": 0.27,
                "auc": 0.50,
            },
            {
                "model": "Majority",
                "status": "success",
                "accuracy": 0.31,
                "balanced_accuracy": 0.33,
                "macro_f1": 0.25,
                "weighted_f1": 0.29,
                "auc": 0.50,
            },
            {
                "model": "HeteroGCNMean",
                "status": "success",
                "accuracy": 0.34,
                "balanced_accuracy": 0.35,
                "macro_f1": 0.31,
                "weighted_f1": 0.32,
                "auc": 0.49,
            },
            {
                "model": "LightGBM",
                "status": "success",
                "accuracy": 0.37,
                "balanced_accuracy": 0.38,
                "macro_f1": 0.35,
                "weighted_f1": 0.36,
                "auc": 0.51,
            },
        ]
    )

    barplot_calls = []
    original_barplot = quick_comparison.sns.barplot

    def recording_barplot(*args, **kwargs):
        barplot_calls.append(kwargs.copy())
        return original_barplot(*args, **kwargs)

    monkeypatch.setattr(quick_comparison.sns, "barplot", recording_barplot)

    output_path = _save_group_model_ranking(summary=summary, output_dir=tmp_path)

    assert output_path == tmp_path / "plots" / "classification_group_model_ranking.png"
    assert output_path.exists()
    assert barplot_calls
    assert barplot_calls[0]["x"] == "metric_display"
    assert barplot_calls[0]["y"] == "value"
    assert barplot_calls[0]["hue"] == "model_display"
    assert barplot_calls[0]["order"] == ["točnost", "uravnotežena točnost", "makro F1", "utežen F1", "AUC"]
    assert barplot_calls[0]["hue_order"] == ["Naključni", "Večinski", "LightGBM", "HeteroGCN-mean", "HeteroGCN-MLP-w"]
    assert barplot_calls[0]["palette"] == {
        "Naključni": quick_comparison.MODEL_COLOR_PALETTE["Random"],
        "Večinski": quick_comparison.MODEL_COLOR_PALETTE["Majority"],
        "LightGBM": quick_comparison.MODEL_COLOR_PALETTE["LightGBM"],
        "HeteroGCN-mean": quick_comparison.MODEL_COLOR_PALETTE["HeteroGCNMean"],
        "HeteroGCN-MLP-w": quick_comparison.MODEL_COLOR_PALETTE["HeteroGCNMLPWeights"],
    }
    assert quick_comparison._group_model_ranking_title(
        signal_set="gaze_only",
        experiment_id=VALENCE_EXPERIMENT_ID,
        cv_strategy="subject_kfold",
        has_signal_set=True,
    ) == "Primerjava modelov za prepoznavanje valence iz signalov pogleda"


def test_thesis_metric_table_uses_fixed_model_order(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "signal_set": "gaze_only",
                "signal_set_description": "gaze",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "ModelB",
                "status": "success",
                "accuracy": 0.70,
                "macro_f1": 0.50,
                "balanced_accuracy": 0.60,
                "weighted_f1": 0.55,
                "loss": 0.9,
            },
            {
                "signal_set": "gaze_only",
                "signal_set_description": "gaze",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "ModelA",
                "status": "success",
                "accuracy": 0.70,
                "macro_f1": 0.55,
                "balanced_accuracy": 0.61,
                "weighted_f1": 0.56,
                "loss": 0.8,
            },
            {
                "signal_set": "gaze_only",
                "signal_set_description": "gaze",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "ModelC",
                "status": "success",
                "accuracy": 0.65,
                "macro_f1": 0.60,
                "balanced_accuracy": 0.62,
                "weighted_f1": 0.57,
                "loss": 0.7,
            },
        ]
    )

    output_path = _save_thesis_metric_table(summary=summary, output_dir=tmp_path)
    assert output_path == tmp_path / "tables" / "thesis_signal_set_model_metrics.csv"

    table = pd.read_csv(output_path)
    assert table["model"].tolist() == ["ModelA", "ModelB", "ModelC"]
    assert table["model_order"].tolist() == [1, 2, 3]
    assert "rank" not in table.columns
    assert not any(column.endswith("_mean") or column.endswith("_std") for column in table.columns)


def test_thesis_metric_heatmaps_are_written(tmp_path: Path) -> None:
    thesis_metrics = pd.DataFrame(
        [
            {
                "signal_set": "gaze_only",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "LightGBM",
                "accuracy": 0.66,
                "macro_f1": 0.61,
            },
            {
                "signal_set": "pupil_only",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "LightGBM",
                "accuracy": 0.58,
                "macro_f1": 0.52,
            },
            {
                "signal_set": "gaze_only",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "HeteroGCNMLPWeights",
                "accuracy": 0.63,
                "macro_f1": 0.59,
            },
            {
                "signal_set": "pupil_only",
                "experiment_id": VALENCE_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 Valence",
                "cv_strategy": "subject_kfold",
                "model": "HeteroGCNMLPWeights",
                "accuracy": 0.55,
                "macro_f1": 0.50,
            },
        ]
    )

    paths = _save_thesis_metric_heatmaps(thesis_metrics=thesis_metrics, output_dir=tmp_path)

    expected_paths = {
        tmp_path / "plots" / f"thesis_accuracy_heatmap_{VALENCE_EXPERIMENT_ID}_subject_kfold.png",
        tmp_path / "plots" / f"thesis_macro_f1_heatmap_{VALENCE_EXPERIMENT_ID}_subject_kfold.png",
    }
    assert set(paths) == expected_paths
    for path in expected_paths:
        assert path.exists()


def test_training_history_outputs_include_loss_and_validation_plots(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    fold_dir = trainer_run_dir / "subject_kfold" / "fold_0"
    fold_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "model": "HeteroGCNMLPWeights",
                "train_loss": 1.1,
                "val_loss": 1.0,
                "val_balanced_accuracy": 0.4,
                "val_macro_f1": 0.35,
                "best_epoch": 2,
            },
            {
                "epoch": 2,
                "model": "HeteroGCNMLPWeights",
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_balanced_accuracy": 0.5,
                "val_macro_f1": 0.45,
                "best_epoch": 2,
            },
        ]
    ).to_csv(fold_dir / "gnn_training_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "val",
                "graph_embedding_variance": 0.12,
                "graph_embedding_mean_pairwise_cosine_similarity": 0.34,
                "logit_mean": 0.1,
                "prediction_entropy_mean": 0.9,
                "temporal_forward_edge_weight_mean": 0.01,
            }
        ]
    ).to_csv(fold_dir / "gnn_fold_diagnostics.csv", index=False)

    rows = [
        {
            "model": "HeteroGCNMLPWeights",
            "experiment_id": AROUSAL_EXPERIMENT_ID,
            "experiment_display_name": "Table-6 arousal 3-class",
            "cv_strategy": "subject_kfold",
            "status": "success",
            "suite_run_dir": str(suite_run_dir),
            "summary_model_name": "GNN",
        }
    ]
    paths = _save_training_history_outputs(rows=rows, output_dir=tmp_path)

    expected_paths = {
        tmp_path / "tables" / "training_history.csv",
        tmp_path / "plots" / "training_progress_loss.png",
        tmp_path / "plots" / "training_progress_validation_metrics.png",
        tmp_path / "plots" / "best_epoch_distribution.png",
        tmp_path / "tables" / "training_diagnostics.csv",
        tmp_path / "tables" / "training_diagnostics_summary.csv",
    }
    assert expected_paths.issubset(set(paths))
    for path in expected_paths:
        assert path.exists()
    assert not (tmp_path / "plots" / "losses").exists()
    history = pd.read_csv(tmp_path / "tables" / "training_history.csv")
    assert history["model"].tolist() == ["HeteroGCNMLPWeights", "HeteroGCNMLPWeights"]
    assert "history_source_model" not in history.columns

    full_paths = _save_training_history_outputs(rows=rows, output_dir=tmp_path, save_fold_loss_plots=True)
    loss_plot_paths = [path for path in full_paths if path.parent.name == "losses"]
    assert loss_plot_paths
    assert all(path.exists() for path in loss_plot_paths)


def test_test_loss_summary_plot_is_written(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "HeteroGCNMLPWeights",
                "status": "success",
                "loss": 0.83,
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 arousal 3-class",
                "cv_strategy": "subject_kfold",
            },
            {
                "model": "LightGBM",
                "status": "success",
                "loss": np.nan,
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "experiment_display_name": "Table-6 arousal 3-class",
                "cv_strategy": "subject_kfold",
            },
        ]
    )

    output_path = _plot_test_loss_summary(summary=summary, output_dir=tmp_path)

    assert output_path == tmp_path / "plots" / "test_loss_by_model.png"
    assert output_path.exists()


def test_fold_metric_outputs_include_top_level_metrics_and_std(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_loo"
    strategy_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "GNN",
                "fold_id": "s_P1",
                "metric_type": "aggregated",
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "loss": 1.1,
            },
            {
                "model": "GNN",
                "fold_id": "s_P2",
                "metric_type": "aggregated",
                "accuracy": 0.7,
                "macro_f1": 0.6,
                "loss": 0.9,
            },
        ]
    ).to_csv(strategy_dir / "fold_metrics.csv", index=False)

    paths = _save_fold_metric_outputs(
        rows=[
            {
                "model": "HeteroGCNMLPWeights",
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "cv_strategy": "subject_loo",
                "status": "success",
                "suite_run_dir": str(suite_run_dir),
                "summary_model_name": "GNN",
            }
        ],
        output_dir=tmp_path,
    )

    assert {
        tmp_path / "tables" / "fold_metrics.csv",
        tmp_path / "tables" / "metric_summary_with_std.csv",
    }.issubset(set(paths))

    fold_metrics = pd.read_csv(tmp_path / "tables" / "fold_metrics.csv")
    assert fold_metrics["model"].tolist() == ["HeteroGCNMLPWeights", "HeteroGCNMLPWeights"]
    assert fold_metrics["metric_source_model"].tolist() == ["GNN", "GNN"]
    assert fold_metrics["fold_id"].tolist() == ["s_P1", "s_P2"]

    metric_summary = pd.read_csv(tmp_path / "tables" / "metric_summary_with_std.csv")
    accuracy_row = metric_summary[
        (metric_summary["model"] == "HeteroGCNMLPWeights")
        & (metric_summary["metric_type"] == "aggregated")
        & (metric_summary["metric"] == "accuracy")
    ].iloc[0]
    assert accuracy_row["n_folds"] == 2
    assert np.isclose(accuracy_row["mean"], 0.6)
    assert np.isclose(accuracy_row["std"], np.std([0.5, 0.7], ddof=1))


def test_named_gnn_artifact_paths_are_collected(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_kfold"
    fold_dir = strategy_dir / "fold_0"
    model_name = "HeteroGCNMLPWeights"
    model_dir = fold_dir / "gnn" / model_name
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "fold_id": "fold_0",
                "metric_type": "aggregated",
                "accuracy": 0.75,
                "macro_f1": 0.70,
                "loss": 0.6,
            }
        ]
    ).to_csv(strategy_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_balanced_accuracy": 0.55,
                "val_macro_f1": 0.52,
                "best_epoch": 1,
            }
        ]
    ).to_csv(model_dir / "gnn_training_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "graph_embedding_variance": 0.12,
                "prediction_entropy_mean": 0.7,
            }
        ]
    ).to_csv(model_dir / "gnn_fold_diagnostics.csv", index=False)
    np.save(model_dir / "test_targets.npy", np.asarray([0, 1, 1]))
    np.save(
        model_dir / "test_predictions.npy",
        np.asarray(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        ),
    )
    (model_dir / "model_benchmark.json").write_text(
        json.dumps(
                {
                    "fold_id": "fold_0",
                    "model": model_name,
                    "model_family": "gnn",
                    "fit_seconds": 3.0,
                    "train_windows": 10,
                    "trainable_parameters": 123,
                    "total_parameters": 123,
                    "accuracy": 0.75,
                    "macro_f1": 0.70,
                }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "model": model_name,
            "experiment_id": AROUSAL_EXPERIMENT_ID,
            "experiment_display_name": "Table-6 arousal 3-class",
            "cv_strategy": "subject_kfold",
            "status": "success",
            "suite_run_dir": str(suite_run_dir),
            "summary_model_name": model_name,
            "signal_set": "all_signals",
            "signal_set_description": "all",
        }
    ]

    fold_paths = _save_fold_metric_outputs(rows=rows, output_dir=tmp_path)
    history_paths = _save_training_history_outputs(rows=rows, output_dir=tmp_path, save_plots=False)
    benchmark_paths = _save_model_benchmark_outputs(
        rows=rows,
        quick_summary=pd.DataFrame(rows),
        output_dir=tmp_path,
    )
    confusion_path = _save_confusion_matrix_table(rows=rows, output_dir=tmp_path)

    assert tmp_path / "tables" / "fold_metrics.csv" in fold_paths
    assert tmp_path / "tables" / "training_history.csv" in history_paths
    assert tmp_path / "tables" / "training_diagnostics.csv" in history_paths
    assert tmp_path / "tables" / "model_benchmark_raw.csv" in benchmark_paths
    assert confusion_path == tmp_path / "tables" / "confusion_matrices.csv"

    history = pd.read_csv(tmp_path / "tables" / "training_history.csv")
    diagnostics = pd.read_csv(tmp_path / "tables" / "training_diagnostics.csv")
    benchmarks = pd.read_csv(tmp_path / "tables" / "model_benchmark_raw.csv")
    confusion = pd.read_csv(tmp_path / "tables" / "confusion_matrices.csv")

    assert history["summary_model_name"].tolist() == [model_name]
    assert history["history_path"].str.contains(f"gnn/{model_name}/gnn_training_history.csv", regex=False).all()
    assert diagnostics["diagnostics_path"].str.contains(f"gnn/{model_name}/gnn_fold_diagnostics.csv", regex=False).all()
    assert benchmarks["benchmark_path"].str.contains(f"gnn/{model_name}/model_benchmark.json", regex=False).all()
    assert set(confusion["model"]) == {model_name}
    assert confusion["count"].sum() == 3


def test_combined_confusion_matrices_include_random_and_majority(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_loo" / "fold_0"
    (strategy_dir / "baselines" / "Random").mkdir(parents=True)
    (strategy_dir / "baselines" / "Majority").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)

    targets = np.asarray([0, 1, 2, 0])
    random_predictions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    majority_predictions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    for model_name, predictions in {
        "Random": random_predictions,
        "Majority": majority_predictions,
    }.items():
        model_dir = strategy_dir / "baselines" / model_name
        np.save(model_dir / "test_targets.npy", targets)
        np.save(model_dir / "test_predictions.npy", predictions)

    rows = [
        {
            "model": "Random",
            "experiment_id": AROUSAL_EXPERIMENT_ID,
            "cv_strategy": "subject_loo",
            "status": "success",
            "suite_run_dir": str(suite_run_dir),
            "summary_model_name": "Random",
        },
        {
            "model": "Majority",
            "experiment_id": AROUSAL_EXPERIMENT_ID,
            "cv_strategy": "subject_loo",
            "status": "success",
            "suite_run_dir": str(suite_run_dir),
            "summary_model_name": "Majority",
        },
    ]

    output_path = _save_combined_confusion_matrices(
        rows=rows,
        output_dir=tmp_path,
        experiment_id=AROUSAL_EXPERIMENT_ID,
        cv_strategy="subject_loo",
        use_strategy_suffix=False,
    )

    assert output_path == tmp_path / "figures" / "confusion_matrices_table6_arousal.png"
    assert output_path.exists()


def test_combined_confusion_matrices_can_use_strategy_suffix(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "recording_loo" / "fold_0"
    model_dir = strategy_dir / "baselines" / "Random"
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    np.save(model_dir / "test_targets.npy", np.asarray([0, 1]))
    np.save(model_dir / "test_predictions.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]]))

    output_path = _save_combined_confusion_matrices(
        rows=[
            {
                "model": "Random",
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "cv_strategy": "recording_loo",
                "status": "success",
                "suite_run_dir": str(suite_run_dir),
                "summary_model_name": "Random",
            }
        ],
        output_dir=tmp_path,
        experiment_id=AROUSAL_EXPERIMENT_ID,
        cv_strategy="recording_loo",
        use_strategy_suffix=True,
    )

    assert output_path == tmp_path / "figures" / "confusion_matrices_table6_arousal_recording_loo.png"
    assert output_path.exists()


def test_confusion_matrix_table_contains_counts_and_row_normalized_values(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_loo" / "fold_0"
    model_dir = strategy_dir / "baselines" / "Random"
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    (trainer_run_dir / "class_metadata.yaml").write_text(
        "index_to_name:\n  0: Low\n  1: Medium\n  2: High\n",
        encoding="utf-8",
    )
    np.save(model_dir / "test_targets.npy", np.asarray([0, 1, 1, 2]))
    np.save(
        model_dir / "test_predictions.npy",
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    output_path = _save_confusion_matrix_table(
        rows=[
            {
                "model": "Random",
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "cv_strategy": "subject_loo",
                "status": "success",
                "suite_run_dir": str(suite_run_dir),
                "summary_model_name": "Random",
            }
        ],
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "tables" / "confusion_matrices.csv"
    table = pd.read_csv(output_path)
    medium_to_low = table[
        (table["model"] == "Random")
        & (table["true_class_index"] == 1)
        & (table["pred_class_index"] == 0)
    ].iloc[0]
    assert medium_to_low["true_class_name"] == "Medium"
    assert medium_to_low["pred_class_name"] == "Low"
    assert medium_to_low["count"] == 1
    assert np.isclose(medium_to_low["row_normalized"], 0.5)


def test_label_distribution_outputs_include_tables_and_plots(tmp_path: Path) -> None:
    suite_run_dir = tmp_path / "suite"
    trainer_run_dir = suite_run_dir / "trainer"
    strategy_dir = trainer_run_dir / "subject_kfold" / "fold_0"
    model_dir = strategy_dir / "baselines" / "Random"
    model_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "status": "success",
                "trainer_run_dir": str(trainer_run_dir),
            }
        ]
    ).to_csv(suite_run_dir / "suite_experiment_registry.csv", index=False)
    np.save(model_dir / "test_targets.npy", np.asarray([0, 0, 1, 2]))

    paths = _save_label_distribution_outputs(
        rows=[
            {
                "model": "Random",
                "experiment_id": AROUSAL_EXPERIMENT_ID,
                "cv_strategy": "subject_kfold",
                "status": "success",
                "suite_run_dir": str(suite_run_dir),
                "summary_model_name": "Random",
            }
        ],
        output_dir=tmp_path,
    )

    expected_paths = {
        tmp_path / "tables" / "label_distribution_by_fold.csv",
        tmp_path / "tables" / "label_distribution_aggregate.csv",
        tmp_path / "plots" / "label_distribution_counts.png",
        tmp_path / "plots" / "label_distribution_proportions.png",
    }
    assert expected_paths.issubset(set(paths))
    for path in expected_paths:
        assert path.exists()

    aggregate = pd.read_csv(tmp_path / "tables" / "label_distribution_aggregate.csv")
    assert aggregate["count"].tolist() == [2, 1, 1]
    assert np.allclose(aggregate["proportion"].to_numpy(), np.asarray([0.5, 0.25, 0.25]))
