import os
import json
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone
import plotly.express as px
from huggingface_hub import snapshot_download


HF_REPO_ID = os.environ.get("HF_REPO_ID", "FastVideo/performance-tracking")
HF_TOKEN = os.environ.get("HF_API_KEY")


# -----------------------------
# 1. Load HF JSON artifacts
# -----------------------------
# def load_hf_dataset(days: int = 30) -> pd.DataFrame:
#     """
#     Loads only recent benchmark JSONs from HF repo.
#     Avoids full history scan when possible.
#     """
#     api = HfApi(token=HF_TOKEN)

#     files = api.list_repo_files(
#         repo_id=HF_REPO_ID,
#         repo_type="dataset",
#     )

#     cutoff = datetime.now(timezone.utc) - timedelta(days=days)
#     records = []

#     for f in files:
#         if not f.endswith(".json"):
#             continue

#         path = hf_hub_download(
#             repo_id=HF_REPO_ID,
#             repo_type="dataset",
#             filename=f,
#             token=HF_TOKEN,
#         )

#         try:
#             with open(path, "r") as fp:
#                 data = json.load(fp)

#             # fast skip: timestamp filter BEFORE dataframe creation
#             ts = pd.to_datetime(data.get("timestamp"), utc=True)
#             if ts < cutoff:
#                 continue

#             records.append(data)

#         except Exception:
#             continue

#     return pd.DataFrame(records)

def load_hf_dataset(days: int = 30) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        allow_patterns="*.json"
    )

    records = []

    for path in glob.glob(f"{local_dir}/**/*.json", recursive=True):
        try:
            with open(path, "r") as fp:
                data = json.load(fp)

            ts = pd.to_datetime(data.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts) or ts < cutoff:
                continue

            records.append(data)

        except Exception:
            continue

    return pd.DataFrame(records)

# -----------------------------
# 2. Normalize schema (your CI format)
# -----------------------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # config = commit + optional future knobs
    df["config_id"] = df["commit_sha"].fillna("unknown").str[:7]

    # ensure numeric safety
    for c in ["latency", "throughput", "memory"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# 3. Grouping (your requested design)
# -----------------------------
def group_data(df: pd.DataFrame):
    keys = ["model_id", "gpu_type", "config_id"]
    return df.groupby(keys, dropna=False)


# -----------------------------
# 4. Plot builder (clean + scalable)
# -----------------------------
def build_plots(df: pd.DataFrame):
    figs = []

    grouped = group_data(df)

    for name, g in grouped:
        g = g.sort_values("timestamp")

        model_id, gpu_type, config_id = name

        title = f"{model_id} | {gpu_type} | {config_id}"

        fig = px.line(
            g,
            x="timestamp",
            y=["latency", "throughput", "memory"],
            markers=True,
            title=title,
        )

        figs.append(fig)

    return figs


# -----------------------------
# 5. Render HTML dashboard
# -----------------------------
def render_html(figs):
    html_parts = [
        "<html><head><meta charset='utf-8'></head><body>",
        f"<h2>Performance Dashboard (last run)</h2>",
    ]

    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


# -----------------------------
# 6. Buildkite annotation (fixed)
# -----------------------------
def annotate_buildkite(html: str):
    if not os.environ.get("BUILDKITE"):
        return

    import subprocess

    subprocess.run(
        ["buildkite-agent", "annotate", "--style", "info", "--context", "perf-dashboard"],
        input=html.encode(),
        check=False,
    )

# -----------------------------
# 6. Buildkite upload
# -----------------------------

# def upload_artifact(path: str):
#     if not os.environ.get("BUILDKITE"):
#         return

#     import subprocess

#     subprocess.run(
#         ["buildkite-agent", "artifact", "upload", path],
#         check=False,
#     )

def upload_artifact(path: str):
    import shutil
    import subprocess

    if shutil.which("buildkite-agent") is None:
        print("buildkite-agent not found, skipping artifact upload")
        return

    subprocess.run(
        ["buildkite-agent", "artifact", "upload", path],
        check=True,
    )
    
# -----------------------------
# 7. Main
# -----------------------------
def main():
    days = int(os.environ.get("DASHBOARD_DAYS", "30"))

    df = load_hf_dataset(days=days)

    if df.empty:
        print("No data found")
        return

    df = normalize(df)

    figs = build_plots(df)
    html = render_html(figs)

    output_file = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(output_file, "w") as f:
        f.write(html)

    annotate_buildkite(html)
    print(f"Dashboard generated: {output_file}")

    upload_artifact(output_file)
    print(f"Dashboard uploaded as Buildkite artifact: {output_file}")

if __name__ == "__main__":
    main()