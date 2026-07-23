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
        "result_schema_version": 2,
        "baseline_status": "compared",
        "comparison_status": "PASS",
        "comparison_status_reason": "Comparable baseline found",
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
    assert {key[6] for key, _group in groups} == {"recipe-a", "recipe-b"}
    assert {key[8] for key, _group in groups} == {"sw-a", "sw-b"}


def test_group_data_ignores_v2_display_name_changes():
    df = pd.DataFrame([
        _record(model_id="old-display", gpu_type="NVIDIA L40S old label"),
        _record(
            model_id="new-display",
            gpu_type="NVIDIA L40S",
            timestamp="2026-01-02T00:00:00+00:00",
        ),
    ])

    groups = list(dashboard.group_data(df))
    figs, _skipped_metrics = dashboard.build_plots(df)

    assert len(groups) == 1
    assert all("new-display | NVIDIA L40S" in fig.layout.title.text for fig in figs)


def test_group_data_fills_legacy_cohort_columns():
    df = pd.DataFrame([
        {
            "model_id": model_id,
            "gpu_type": "NVIDIA L40S",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "commit_sha": "a" * 40,
            "config_id": "aaaaaaa",
            "latency": 10.0,
        }
        for model_id in ("legacy-wan", "legacy-ltx")
    ])

    groups = list(dashboard.group_data(df))
    figs, _skipped_metrics = dashboard.build_plots(df)

    assert len(groups) == 2
    assert {key[1] for key, _group in groups} == {"legacy-wan", "legacy-ltx"}
    assert all(key[3:] == ("", "", "", "", "", "") for key, _group in groups)
    assert {key[0] for key, _group in groups} == {"legacy_v1"}
    assert all("Legacy v1" in fig.layout.title.text for fig in figs)
    assert all("legacy / legacy" not in fig.layout.title.text for fig in figs)


def test_group_data_treats_present_blank_or_schema_only_identity_as_invalid_v2():
    legacy_record = {
        "model_id": "wan-t2v-1.3b-2gpu",
        "gpu_type": "NVIDIA L40S",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit_sha": "a" * 40,
        "config_id": "aaaaaaa",
        "latency": 10.0,
    }
    df = pd.DataFrame([
        legacy_record,
        {
            **legacy_record,
            "timestamp": "2026-01-02T00:00:00+00:00",
            "commit_sha": "b" * 40,
            "workload_id": "",
        },
        {
            **legacy_record,
            "timestamp": "2026-01-03T00:00:00+00:00",
            "commit_sha": "c" * 40,
            "result_schema_version": 2,
        },
    ])

    groups = list(dashboard.group_data(df))

    assert len(groups) == 2
    assert {(key[0], len(group)) for key, group in groups} == {("legacy_v1", 1), ("invalid_v2", 2)}


def test_group_data_keeps_partial_v2_identity_scoped_by_display_metadata():
    partial_identity = {
        "result_schema_version": 2,
        "workload_id": "wan-t2v",
        "variant_id": "1.3b-sp2",
        "benchmark_version": 2,
        "recipe_fingerprint": "",
        "hardware_profile_id": "",
        "software_profile_id": "",
    }
    df = pd.DataFrame([
        _record(model_id="wan", gpu_type="NVIDIA L40S", **partial_identity),
        _record(model_id="ltx", gpu_type="NVIDIA H100", **partial_identity),
    ])

    groups = list(dashboard.group_data(df))
    figs, _skipped_metrics = dashboard.build_plots(df)

    assert len(groups) == 2
    assert {(key[1], key[2]) for key, _group in groups} == {
        ("wan", "NVIDIA L40S"),
        ("ltx", "NVIDIA H100"),
    }
    assert {key[0] for key, _group in groups} == {"invalid_v2"}
    assert all("Invalid v2:" in fig.layout.title.text for fig in figs)
    assert all("Legacy v1" not in fig.layout.title.text for fig in figs)


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


def test_build_plots_includes_comparison_status_hover_data():
    figs, _skipped_metrics = dashboard.build_plots(pd.DataFrame([_record()]))

    hover_template = figs[0].data[0].hovertemplate

    assert "cohort_kind=" in hover_template
    assert "comparison_status=" in hover_template
    assert "comparison_status_reason=" in hover_template
    assert "baseline_status=" in hover_template


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
