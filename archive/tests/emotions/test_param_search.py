from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from emotions import param_search


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_param_search_uses_config_contract_and_writes_summary(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    base_config_path = tmp_path / "base_config.yaml"
    param_grid_path = tmp_path / "grid.yaml"

    _write_yaml(
        base_config_path,
        {
            "dataset": {"window_length": 10, "kt": 2, "ks": 2},
            "cross_validation": {"strategies": ["subject_loo"]},
            "gnn": {"model": {"hidden_channels": 32, "use_preprocess_mlp": True, "add_self_loops": False}},
            "logging": {"results_dir": str(results_dir)},
        },
    )
    _write_yaml(param_grid_path, {"window_length": [5, 8]})

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], cwd: str, env: dict) -> subprocess.CompletedProcess:
        assert "--config" in cmd
        cfg_path = Path(cmd[cmd.index("--config") + 1])
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

        run_index = len(calls)
        run_dir = results_dir / f"run_{run_index:02d}"
        strategy_dir = run_dir / "subject_loo"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        metric = float(cfg["dataset"]["window_length"])
        pd.DataFrame(
            [
                {
                    "model": "GNN",
                    "metric_type": "aggregated",
                    "mse": metric,
                    "mae": metric + 0.1,
                }
            ]
        ).to_csv(strategy_dir / "summary.csv", index=False)

        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(param_search.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "param_search.py",
            "--search_type",
            "grid",
            "--base_config",
            str(base_config_path),
            "--param_grid",
            str(param_grid_path),
        ],
    )

    param_search.main()

    assert len(calls) == 2
    for cmd in calls:
        assert cmd[:3] == [sys.executable, "src/emotions/train.py", "--config"]

    summary_files = sorted(results_dir.glob("grid_search_summary_*.csv"))
    ordered_files = sorted(results_dir.glob("grid_search_summary_ordered_*.csv"))
    assert summary_files
    assert ordered_files

    summary_df = pd.read_csv(summary_files[-1])
    assert set(summary_df["status"].tolist()) == {"success"}
    assert set(summary_df["config"].tolist()) == {"config_0000", "config_0001"}
