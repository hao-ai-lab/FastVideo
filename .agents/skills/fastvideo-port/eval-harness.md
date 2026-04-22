# Eval Harness — Skill Improvement Loop

Run this to measure how well the skill performs against a known-good port.

```bash
# Step 1: create eval branch from pre-port commit
python .agents/skills/fastvideo-port/scripts/eval_harness.py \
    --ground_truth_branch feat/cosmos25-training \
    --pre_port_commit <hash_before_cosmos_work> \
    --model_name cosmos2_5

# Step 2: run the port skill on the eval branch, then diff
python .agents/skills/fastvideo-port/scripts/eval_harness.py \
    --ground_truth_branch feat/cosmos25-training \
    --agent_branch eval/cosmos25-port-test \
    --diff_only --model_name cosmos2_5 \
    --output_dir eval_results/cosmos25/

# Step 3: review lesson candidates in eval_results/cosmos25/lesson_candidates/
# Promote good ones to .agents/lessons/
```

## When to run

- After adding a new lesson: verify the lesson would have prevented the original mistake.
- After completing a port: run before reverting, capture any skill gaps discovered.
- Before opening the skill PR: confirm overall pass rate hasn't regressed.