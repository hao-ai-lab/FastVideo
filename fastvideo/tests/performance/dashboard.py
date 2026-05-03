# SPDX-License-Identifier: Apache-2.0
import os
from html import escape
from datetime import datetime

import plotly.express as px
import pandas as pd

from hf_store import sync_from_hf, load_as_dataframe

METRICS = (
    "latency",
    "throughput",
    "memory",
    "text_encoder_time_s",
    "dit_time_s",
    "vae_decode_time_s",
)

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
def build_plots(df: pd.DataFrame) -> tuple[list, list[dict[str, object]]]:
    figs = []
    skipped_metrics: list[dict[str, object]] = []

    for (model_id, gpu_type), g in group_data(df):
        g = g.sort_values("timestamp")

        # One chart per metric so the y-axes aren't on wildly different scales
        for metric in METRICS:
            if metric not in g.columns:
                skipped_metrics.append({
                    "model_id": model_id,
                    "gpu_type": gpu_type,
                    "metric": metric,
                    "reason": "column missing from loaded records",
                    "records": len(g),
                    "non_null": 0,
                })
                continue

            non_null = int(g[metric].notna().sum())
            if non_null == 0:
                skipped_metrics.append({
                    "model_id": model_id,
                    "gpu_type": gpu_type,
                    "metric": metric,
                    "reason": "no non-null values in loaded records",
                    "records": len(g),
                    "non_null": non_null,
                })
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

    return figs, skipped_metrics


def render_skipped_metrics(skipped_metrics: list[dict[str, object]]) -> str:
    if not skipped_metrics:
        return ""

    rows = [
        "<h3>Skipped Metric Plots</h3>",
        "<table>",
        ("<thead><tr><th>Model</th><th>GPU</th><th>Metric</th>"
         "<th>Records</th><th>Non-null</th><th>Reason</th></tr></thead>"),
        "<tbody>",
    ]
    for item in skipped_metrics:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item['model_id']))}</td>"
            f"<td>{escape(str(item['gpu_type']))}</td>"
            f"<td>{escape(str(item['metric']))}</td>"
            f"<td>{item['records']}</td>"
            f"<td>{item['non_null']}</td>"
            f"<td>{escape(str(item['reason']))}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)

# -----------------------------
# 3. Render HTML dashboard
# -----------------------------
def render_html(figs: list, skipped_metrics: list[dict[str, object]],
                days: int) -> str:
    html_parts = [
        "<html>",
        "<head><meta charset='utf-8'>",
        ("<style>"
         "body { font-family: sans-serif; margin: 2rem; }"
         "table { border-collapse: collapse; margin: 1rem 0 2rem; }"
         "th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; "
         "text-align: left; }"
         "th { background: #f5f5f5; }"
         "</style>"),
        "</head><body>",
        f"<h2>Performance Dashboard (last {days} days)</h2>",
    ]

    skipped_html = render_skipped_metrics(skipped_metrics)
    if skipped_html:
        html_parts.append(skipped_html)

    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")
    return "\n".join(html_parts)

# -----------------------------
# 5. Main
# -----------------------------
def main() -> None:
    days = int(os.environ.get("DASHBOARD_DAYS", "30"))

    local_dir = sync_from_hf("/tmp/perf-tracking", reuse_existing=True)
    df = load_as_dataframe(local_dir, days=days)

    if df.empty:
        print("No data found")
        return

    # Sanity-check: log what we actually loaded
    print(f"Loaded {len(df)} records across {df['model_id'].nunique()} model(s), "
          f"{df['gpu_type'].nunique()} GPU type(s), "
          f"date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    figs, skipped_metrics = build_plots(df)
    if skipped_metrics:
        print("Skipped metric plots:")
        for item in skipped_metrics:
            print(f"  - {item['model_id']} | {item['gpu_type']} | "
                  f"{item['metric']}: {item['reason']} "
                  f"({item['non_null']}/{item['records']} non-null)")
    print(f"Generated {len(figs)} metric plot(s)")
    html = render_html(figs, skipped_metrics, days)

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
