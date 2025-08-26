import ray  # type: ignore
import ray.data as rd  # type: ignore

ray.init(ignore_reinit_error=True)

jsonl_path = "/mnt/fast-disks/nfs/hao_lab/FastVideo/logs/user_prompts.jsonl"
prompts_file = "/mnt/fast-disks/nfs/hao_lab/FastVideo/prompts/prompts_final.txt"
video_outputs_dir = "/mnt/fast-disks/nfs/hao_lab/FastVideo/outputs/"

# Load blocklist of prompts (exact line match, ignoring only newline chars)
with open(prompts_file, "r", encoding="utf-8") as f:
    blocklist = {line.rstrip("\r\n") for line in f if line.rstrip("\r\n")}

ds = rd.read_json(jsonl_path)

# Filter out any rows whose 'prompt' is in the text file
# (batchwise Pandas filtering is efficient for large datasets)
ds_filtered = ds.map_batches(
    lambda df: df[~df["prompt"].isin(blocklist)],
    batch_format="pandas",
)

# Keep one row per prompt; choose deterministically by latest timestamp
ds_dedup = ds_filtered.groupby("prompt").map_groups(
    lambda df: df.sort_values("timestamp", ascending=False).head(1),
    batch_format="pandas",
)

print("before:", ds.count(), "after filter:", ds_filtered.count(), "after dedup:", ds_dedup.count())
# Write a single JSONL file (not sharded)
out_path = "/mnt/fast-disks/nfs/hao_lab/FastVideo/logs/user_prompts_dedup.jsonl"
ds_dedup.to_pandas().to_json(out_path, orient="records", lines=True, force_ascii=False)