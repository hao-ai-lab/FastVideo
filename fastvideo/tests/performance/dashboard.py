# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import subprocess
from datetime import datetime

import plotly.express as px
import pandas as pd

from hf_store import sync_from_hf, load_as_dataframe

# -----------------------------
# 1. Grouping
# -----------------------------
def group_data(df: pd.DataFrame):
    # Group only by model+GPU so each group produces a time-series line.
    # config_id (commit SHA) is carried as a column for hover/color use.
    keys = ["model_id", "gpu_type"]
    return df.groupby(keys, dropna=False)

# -----------------------------
# 2. Plot builder
# -----------------------------
def build_plots(df: pd.DataFrame) -> list:
    figs = []

    for (model_id, gpu_type), g in group_data(df):
        g = g.sort_values("timestamp")

        # One chart per metric so the y-axes aren't on wildly different scales
        for metric in ("latency", "throughput", "memory",
            "text_encoder_time_s", "dit_time_s", "vae_decode_time_s"):
            if g[metric].isna().all():
                continue

            fig = px.line(
                g,
                x="timestamp",
                y=metric,
                markers=True,
                hover_data=["config_id", "commit_sha"],
                title=f"{model_id} | {gpu_type} | {metric}",
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