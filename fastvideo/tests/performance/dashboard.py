# SPDX-License-Identifier: Apache-2.0
import os
from datetime import datetime

import plotly.express as px
import pandas as pd

from hf_store import sync_from_hf, load_as_dataframe

# Grouping keys for time-series plots. Adding env-derived columns
# (torch_version, cuda_runtime, attention_backend) prevents pre-/post-upgrade
# records from being averaged into the same line, which would otherwise
# silently smear environment-driven shifts across the trend.
_GROUP_KEYS = (
    "model_id",
    "gpu_type",
    "torch_version",
    "cuda_runtime",
    "attention_backend",
)


def group_data(df: pd.DataFrame):
    keys = [k for k in _GROUP_KEYS if k in df.columns]
    return df.groupby(keys, dropna=False)


def _format_group_label(key_values: tuple, key_names: list[str]) -> str:
    parts = []
    for name, value in zip(key_names, key_values, strict=False):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            display = "n/a"
        else:
            display = str(value)
        parts.append(f"{name}={display}")
    return " | ".join(parts)


def build_plots(df: pd.DataFrame) -> list:
    figs = []

    grouped = group_data(df)
    key_names = list(grouped.keys) if hasattr(grouped, "keys") else list(_GROUP_KEYS)

    for key_values, g in grouped:
        if not isinstance(key_values, tuple):
            key_values = (key_values, )
        g = g.sort_values("timestamp")
        label = _format_group_label(key_values, key_names)

        # One chart per metric so the y-axes aren't on wildly different scales
        for metric in ("latency", "throughput", "memory"):
            if metric not in g.columns or g[metric].isna().all():
                continue

            fig = px.line(
                g,
                x="timestamp",
                y=metric,
                markers=True,
                hover_data=["config_id", "commit_sha"],
                title=f"{label} | {metric}",
                labels={"timestamp": "Time", metric: metric},
            )
            figs.append(fig)

    return figs

# -----------------------------
# 3. Render HTML dashboard
# -----------------------------
def render_html(figs: list, days: int) -> str:
    html_parts = [
        "<html>",
        "<head><meta charset='utf-8'>",
        "<style>body { font-family: sans-serif; margin: 2rem; }</style>",
        "</head><body>",
        f"<h2>Performance Dashboard (last {days} days)</h2>",
    ]

    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")
    return "\n".join(html_parts)

# -----------------------------
# 5. Main
# -----------------------------
def main() -> None:
    days = int(os.environ.get("DASHBOARD_DAYS", "30"))

    local_dir = sync_from_hf("/tmp/perf-tracking")
    df = load_as_dataframe(local_dir, days=days)

    if df.empty:
        print("No data found")
        return

    # Sanity-check: log what we actually loaded
    print(f"Loaded {len(df)} records across {df['model_id'].nunique()} model(s), "
          f"{df['gpu_type'].nunique()} GPU type(s), "
          f"date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    figs = build_plots(df)
    html = render_html(figs, days)

    commit_sha = os.environ.get("BUILDKITE_COMMIT", "unknown")[:7]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_dir = "/root/data/perf_reports"
    os.makedirs(report_dir, exist_ok=True)

    filename = f"dashboard_{commit_sha}_{timestamp}.html"
    output_file = os.path.join(report_dir, filename)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard generated: {output_file}")

if __name__ == "__main__":
    main()