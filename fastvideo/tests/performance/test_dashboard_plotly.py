# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from fastvideo.tests.performance import dashboard


def _record(recipe_fingerprint="recipe-a", software_profile_id="sw-a", **overrides):
    record = {
        "model_id": "wan-t2v-1.3b-2gpu",
        "gpu_type": "NVIDIA L40S",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": recipe_fingerprint,
        "hardware_profile_id": "hw-l40s",
        "software_profile_id": software_profile_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit_sha": "a" * 40,
        "config_id": "aaaaaaa",
        "latency": 10.0,
        "throughput": 4.5,
        "memory": 10000.0,
        "text_encoder_time_s": 2.0,
        "dit_time_s": 8.0,
        "vae_decode_time_s": 3.0,
    }
    record.update(overrides)
    return record


def test_group_data_uses_full_comparison_cohort():
    df = pd.DataFrame([
        _record(recipe_fingerprint="recipe-a", software_profile_id="sw-a"),
        _record(recipe_fingerprint="recipe-b", software_profile_id="sw-b"),
    ])

    groups = list(dashboard.group_data(df))

    assert len(groups) == 2
    assert {key[5] for key, _group in groups} == {"recipe-a", "recipe-b"}
    assert {key[7] for key, _group in groups} == {"sw-a", "sw-b"}


def test_group_data_fills_legacy_cohort_columns():
    df = pd.DataFrame([{
        "model_id": "legacy-wan",
        "gpu_type": "NVIDIA L40S",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit_sha": "a" * 40,
        "config_id": "aaaaaaa",
        "latency": 10.0,
    }])

    groups = list(dashboard.group_data(df))

    assert len(groups) == 1
    assert groups[0][0][2:] == ("", "", "", "", "", "")


def test_build_plots_labels_distinct_cohorts():
    df = pd.DataFrame([
        _record(recipe_fingerprint="recipe-a", software_profile_id="sw-a"),
        _record(recipe_fingerprint="recipe-b", software_profile_id="sw-b"),
    ])

    figs, skipped_metrics = dashboard.build_plots(df)
    titles = [fig.layout.title.text for fig in figs]

    assert len(figs) == len(dashboard.METRICS) * 2
    assert skipped_metrics == []
    assert any("recipe recipe-a | hw-l40s | sw-a" in title for title in titles)
    assert any("recipe recipe-b | hw-l40s | sw-b" in title for title in titles)


def test_skipped_metric_table_includes_cohort_identity():
    df = pd.DataFrame([{
        "model_id": "wan-t2v-1.3b-2gpu",
        "gpu_type": "NVIDIA L40S",
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "recipe-a",
        "hardware_profile_id": "hw-l40s",
        "software_profile_id": "sw-a",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit_sha": "a" * 40,
        "config_id": "aaaaaaa",
        "latency": 10.0,
    }])

    _figs, skipped_metrics = dashboard.build_plots(df)
    html = dashboard.render_skipped_metrics(skipped_metrics[:1])

    assert skipped_metrics[0]["cohort"] == "wan-t2v / 1.3b-sp2 / v2"
    assert skipped_metrics[0]["cohort_detail"] == "recipe recipe-a | hw-l40s | sw-a"
    assert "wan-t2v / 1.3b-sp2 / v2" in html
    assert "recipe recipe-a | hw-l40s | sw-a" in html
