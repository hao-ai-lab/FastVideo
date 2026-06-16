"""Worked examples (design_v3 §15), runnable as a demo:

    python3 -m mini_fastvideo.examples

Each prints what it demonstrates. This doubles as living documentation of the API.
"""
from __future__ import annotations

import numpy as np

from .models import build_default_engine
from .models.wan21 import build_wan21_card
from .parity import assert_interleave_parity
from .request import DiffusionParams, OutputSpec, TaskType, make_request
from .training import build_diffusion_nft


def _t2v(mid, prompt, seed, steps=4, **kw):
    return make_request(TaskType.T2V, mid, prompt,
                        diffusion=DiffusionParams(num_steps=steps, seed=seed), **kw)


def example_a_text_to_video(eng) -> None:
    print("\n(a) Text → video, one instance (Wan2.1-1.3B)")
    out = eng.run(_t2v("wan2.1-1.3b", "a cat surfing a wave", 7))
    print(f"    video {out.artifacts['video'].frames.shape}  "
          f"denoise_steps={out.metrics['denoise_steps']:.0f}  gpu_s={out.metrics['gpu_seconds']:.2e}")


def example_b_ltx2_two_stage(eng) -> None:
    print("\n(b) LTX2.3 two-stage distilled (base 8-step → upsample → refine 3-step), shared transformer")
    out = eng.run(_t2v("ltx2.3-distilled", "a neon city at night", 2))
    print(f"    video {out.artifacts['video'].frames.shape}  "
          f"base={out.metrics['base_steps']:.0f} refine={out.metrics['refine_steps']:.0f}")


def example_c_causal_streaming(eng) -> None:
    print("\n(c) Causal streaming (Wan-causal): chunk rollout + slab-KV, streamable by chunk")
    out = eng.run(_t2v("wan-causal-sf-1.3b", "a drone flight over mountains", 3,
                       outputs=OutputSpec(stream={"video": True})))
    print(f"    latents {out.artifacts['latents'].latent.shape}  chunks={out.metrics['chunks']:.0f}  "
          f"streamed_chunks={out.metrics.get('stream_chunks', 0)}")


def example_c2_interleave_gate(eng) -> None:
    print("\n(c2) Interleave parity gate — serial == interleaved, bit-identical (the §9.3 obligation)")
    reqs = [_t2v("wan2.1-1.3b", "alpha", 11), _t2v("wan2.1-1.3b", "beta", 22)]
    divs = assert_interleave_parity(eng, reqs)
    print(f"    divergences: {divs or 'NONE — gate PASSES ✓'}")


def example_d_rl_rollout() -> None:
    print("\n(d) RL rollout (DiffusionNFT): the SAME denoise loop + behavior capture (train ≡ serve)")
    nft = build_diffusion_nft(build_wan21_card(), num_video_per_prompt=4, num_inner_timesteps=2)
    loss, m = nft.managed_train_step({"prompts": ["a red car", "a blue boat"], "seeds": [1, 2]}, 0)
    fc = nft.old.caches.stats()["feature"]
    print(f"    policy_loss={loss['policy_loss']:.3f} kl={loss['kl_div_loss']:.5f} "
          f"reward_mean={m['reward_mean']:.3f}  consistency={nft.consistency_level().value} (likelihood-free)")
    print(f"    shared-prompt feature-cache reuse: {fc['hits']} hits / {fc['misses']} misses "
          f"(K samples encode the prompt once — the 24× reduction)")


def main() -> None:
    print("=" * 78)
    print("mini-fastvideo — worked examples (design_v3 §15). One runtime, many loops.")
    print("=" * 78)
    eng = build_default_engine()
    print(f"registered (recipe, runtime) cards: {list(eng._registry)}")
    example_a_text_to_video(eng)
    example_b_ltx2_two_stage(eng)
    example_c_causal_streaming(eng)
    example_c2_interleave_gate(eng)
    example_d_rl_rollout()
    print("\n" + "=" * 78)
    print("All examples ran on CPU with numpy toy components. The architecture (cards, driven")
    print("loops, scheduler, caches, parity, training-on-shared-loops) is real; the neural")
    print("forwards are toys. On a GPU box, swap ComponentSpec.factory for the torch adapters.")
    print("=" * 78)


if __name__ == "__main__":
    main()
