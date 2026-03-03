"""Deploy performance dashboard and results to gh-pages branch.

Copies the dashboard HTML and results from the Modal Volume to the
gh-pages branch and pushes. Requires GH_TOKEN env var.

Usage:
    python .buildkite/performance-benchmarks/scripts/deploy_dashboard.py
"""
import os
import subprocess
import sys

VOLUME_DIR = os.environ.get("PERF_RESULTS_VOLUME", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")
REPO_URL = os.environ.get("BUILDKITE_REPO", "")
DASHBOARD_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "dashboard", "index.html")


def main():
    if not GH_TOKEN:
        print("GH_TOKEN not set, skipping dashboard deploy.")
        return
    if not REPO_URL:
        print("BUILDKITE_REPO not set, skipping dashboard deploy.")
        return

    volume_index = (os.path.join(VOLUME_DIR, "index.json")
                    if VOLUME_DIR else "")
    if not volume_index or not os.path.exists(volume_index):
        print("No results index found, skipping dashboard deploy.")
        return

    auth_url = REPO_URL.replace("https://",
                                f"https://x-access-token:{GH_TOKEN}@")
    work_dir = "/tmp/gh-pages-deploy"

    cmds = f"""
set -e
rm -rf {work_dir}
git clone --branch gh-pages --single-branch --depth 1 \
    {auth_url} {work_dir} 2>/dev/null || {{
    git clone {auth_url} {work_dir}
    cd {work_dir}
    git checkout --orphan gh-pages
    git rm -rf . 2>/dev/null || true
}}
cd {work_dir}
cp {os.path.abspath(DASHBOARD_SRC)} index.html
mkdir -p performance-results
cp {VOLUME_DIR}/*.json performance-results/ 2>/dev/null || true
git add -A
git diff --cached --quiet && echo "No changes to deploy" && exit 0
git -c user.name="CI Bot" -c user.email="ci@fastvideo.dev" \
    commit -m "Update performance dashboard"
git push origin gh-pages
echo "Dashboard deployed successfully"
"""

    result = subprocess.run(["/bin/bash", "-c", cmds],
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                            check=False)
    if result.returncode != 0:
        print(f"Warning: dashboard deploy failed "
              f"(exit {result.returncode})")


if __name__ == "__main__":
    main()
