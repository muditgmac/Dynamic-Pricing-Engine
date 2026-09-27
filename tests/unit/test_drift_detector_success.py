"""Success-path tests for drift monitoring."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import run_drift_check


def _install_fake_evidently(
    monkeypatch,
    drift_score,
):
    evidently_module = types.ModuleType("evidently")
    evidently_module.__path__ = []

    metric_module = types.ModuleType(
        "evidently.metric_preset"
    )
    report_module = types.ModuleType(
        "evidently.report"
    )

    class FakeDataDriftPreset:
        pass

    class FakeReport:
        def __init__(self, metrics):
            self.metrics = metrics

        def run(
            self,
            reference_data,
            current_data,
        ):
            assert not reference_data.empty
            assert not current_data.empty
            assert (
                reference_data.columns.tolist()
                == current_data.columns.tolist()
            )

        def save_html(self, path):
            Path(path).write_text(
                "<html>drift report</html>"
            )

        def as_dict(self):
            return {
                "metrics": [
                    {
                        "result": {
                            "share_of_drifted_columns":
                                drift_score,
                            "drift_by_columns": {
                                "price": {
                                    "drift_detected": True
                                },
                                "beds": {
                                    "drift_detected": False
                                },
                            },
                        }
                    }
                ]
            }

    metric_module.DataDriftPreset = (
        FakeDataDriftPreset
    )
    report_module.Report = FakeReport

    monkeypatch.setitem(
        sys.modules,
        "evidently",
        evidently_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "evidently.metric_preset",
        metric_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "evidently.report",
        report_module,
    )


@pytest.mark.parametrize(
    ("score", "expected_status"),
    [
        (0.10, "healthy"),
        (0.40, "degraded"),
        (0.75, "critical"),
    ],
)
def test_drift_report_success_path(
    tmp_path,
    monkeypatch,
    score,
    expected_status,
):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    reference = pd.DataFrame(
        {
            "price": np.linspace(
                100.0,
                300.0,
                40,
            ),
            "beds": np.tile(
                [1.0, 2.0, 3.0, 4.0],
                10,
            ),
            "unrelated_numeric": np.arange(40),
            "name": [f"listing-{i}" for i in range(40)],
        }
    )

    reference.to_parquet(
        processed / "listings_features.parquet",
        index=False,
    )

    _install_fake_evidently(
        monkeypatch,
        drift_score=score,
    )

    import src.monitoring.drift_detector as dd

    monkeypatch.setattr(
        dd,
        "PROJECT_ROOT",
        tmp_path,
    )

    config = {
        "data": {
            "processed_dir": "data/processed",
        },
        "monitoring": {
            "report_output_dir":
                "monitoring/reports",
            "drift_threshold_degraded": 0.30,
            "drift_threshold_critical": 0.60,
        },
    }

    result = run_drift_check(config=config)

    assert result["status"] == expected_status
    assert result["drift_score"] == score
    assert result["features_drifted"] == ["price"]
    assert (
        result["report_url"]
        == "/static/drift_report.html"
    )

    report = (
        tmp_path
        / "monitoring"
        / "reports"
        / "drift_report.html"
    )
    assert report.exists()
