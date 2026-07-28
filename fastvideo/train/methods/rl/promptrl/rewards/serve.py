# SPDX-License-Identifier: Apache-2.0
"""Launch the PromptRL VideoScore2 reward service.

Example (on the reward GPU host)::

    CUDA_VISIBLE_DEVICES=0 python -m \
        fastvideo.train.methods.rl.promptrl.rewards.serve \
        --model-id TIGER-Lab/VideoScore2 --port 8100
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptRL reward service")
    parser.add_argument("--model-id", default="TIGER-Lab/VideoScore2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-wait-sec", type=float, default=300.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--infer-fps", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import uvicorn

    from fastvideo.train.methods.rl.promptrl.rewards.service import (
        VideoScore2Judge,
        create_reward_app,
    )

    judge = VideoScore2Judge(
        args.model_id,
        max_new_tokens=args.max_new_tokens,
        infer_fps=args.infer_fps,
        temperature=args.temperature,
        seed=args.seed,
    )
    app = create_reward_app(judge, max_wait_sec=args.max_wait_sec)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
