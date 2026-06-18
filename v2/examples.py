"""Worked examples (design_v3 §15), runnable as a demo:

    python3 -m v2.examples

Each prints what it demonstrates. This doubles as living documentation of the API.
"""
from __future__ import annotations

import numpy as np

from v2.recipes import build_default_engine, build_omni_engine
from v2.recipes.wan21 import build_wan21_card
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, OutputSpec, SamplingParams, TaskType, make_request
from v2.training import build_diffusion_nft


def _t2v(mid, prompt, seed, steps=4, **kw):
    return make_request(TaskType.T2V, mid, prompt,
                        diffusion=DiffusionParams(num_steps=steps, seed=seed), **kw)


def example_a_text_to_video(eng) -> None:
    print("\n(a) Text → video, one instance (Wan2.1-1.3B)")
    out = eng.run(_t2v("wan2.1-1.3b", "a cat surfing a wave", 7))
    print(f"    video {out.artifacts['video'].frames.shape}  "
          f"denoise_steps={out.metrics['denoise_steps']:.0f}  gpu_s={out.metrics['gpu_seconds']:.2e}")


def example_b_ltx2_two_stage(eng) -> None:
    print("\n(b) LTX-2 two-stage distilled (base 8-step → upsample → refine 3-step), shared transformer")
    out = eng.run(_t2v("ltx2-2stage-distilled", "a neon city at night", 2))
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


def example_g_omni_mot() -> None:
    print("\n(g) Omni / MoT (§16): ONE resident instance runs AR + diffusion loops on shared weights")
    eng = build_omni_engine()
    o = eng.run(make_request(TaskType.T2V, "cosmos3-vfm", "a phoenix",
                             sampling=SamplingParams(max_tokens=6, seed=1),
                             diffusion=DiffusionParams(num_steps=4, seed=1)))
    print(f"    Cosmos3 (reason→joint denoise): text={o.artifacts['text'].text} "
          f"video={o.artifacts['video'].frames.shape}")
    o2 = eng.run(make_request(TaskType.T2I, "bagel-mot", "a teapot",
                              sampling=SamplingParams(max_tokens=6, seed=2),
                              diffusion=DiffusionParams(num_steps=4, seed=2)))
    print(f"    BAGEL (generate_text→generate_image): text={o2.artifacts['text'].text} "
          f"image={o2.artifacts['image'].tensor.shape}")
    print(f"    scheduler priced BOTH WorkUnit kinds (runtime-visible, not one opaque stage): "
          f"{dict(eng.admission.metrics.by_kind)}")


async def _serving_demo() -> None:
    import asyncio

    from v2.deploy import DynamoWorkerAdapter, FakeDynamoRuntime, LocalFleet, build_deployment_card
    from v2.recipes.wan21 import build_wan21_card, build_wan_t2v_program
    from v2.runtime import AsyncEngine, PoolSet, wan_t2v_disaggregated
    from v2.serving import OmniOpenAIServer

    eng = build_default_engine()
    build_omni_engine(eng)
    ae = AsyncEngine(eng)

    # disaggregated pools: encoder → denoiser → decoder
    card = build_wan21_card()
    pools = PoolSet(wan_t2v_disaggregated(), card)
    pools.warmup()
    ae.register_disaggregated("wan-disagg", pools, build_wan_t2v_program())
    out = await ae.generate(make_request(TaskType.T2V, "wan-disagg", "a wave",
                                         diffusion=DiffusionParams(num_steps=4, seed=1)))
    print(f"    disaggregated T2V (enc→den→dec): video={out.artifacts['video'].frames.shape} "
          f"cross-pool transfers={out.metrics['transfers']:.0f}")

    # our own OpenAI server over a real socket
    server = OmniOpenAIServer(ae, engine_id="worker-0")
    host, port = await server.serve(port=0)

    async def http(method, path, body=b""):
        r, w = await asyncio.open_connection(host, port)
        w.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        await w.drain()
        data = await r.read(); w.close()
        return data.decode("utf-8", "replace")

    health = await http("GET", "/health")
    sse = await http("POST", "/v1/chat/completions",
                     b'{"model":"cosmos3-vfm","messages":[{"role":"user","content":"a comet"}],"stream":true}')
    n_chunks = sse.count("data: ")
    print(f"    OpenAI server: /health ok={'healthy' in health}; chat SSE streamed {n_chunks} chunks (omni reason→denoise)")
    await server.close()

    # our own fleet router + Dynamo adapter (frontable, not relied upon) — same DeploymentCard
    dcard = build_deployment_card("worker-0", [card])
    fleet = LocalFleet("least_loaded")
    fleet.register("worker-0", ae, dcard)
    routed = fleet.route(make_request(TaskType.T2V, "wan2.1-1.3b", "x", diffusion=DiffusionParams(num_steps=2)))
    dyn = FakeDynamoRuntime()
    dyn.register_worker(DynamoWorkerAdapter(ae, dcard))
    print(f"    LocalFleet routes to '{routed.worker_id}'; Dynamo adapter registered "
          f"({len(dyn.registry)} worker) — both consume the SAME DeploymentCard")


def example_h_serving_and_fleet() -> None:
    import asyncio
    print("\n(h) Serving + fleet (OUR OWN — Dynamo-optional): async engine, role pools, OpenAI server")
    asyncio.run(_serving_demo())


def main() -> None:
    print("=" * 78)
    print("v2 — worked examples (design_v3 §6,§13-16). One runtime, many loops.")
    print("=" * 78)
    eng = build_default_engine()
    print(f"registered (recipe, runtime) cards: {list(eng._registry)}")
    example_a_text_to_video(eng)
    example_b_ltx2_two_stage(eng)
    example_c_causal_streaming(eng)
    example_c2_interleave_gate(eng)
    example_d_rl_rollout()
    example_g_omni_mot()
    example_h_serving_and_fleet()
    print("\n" + "=" * 78)
    print("All examples ran on CPU with numpy toy components. The architecture (cards, driven")
    print("loops, scheduler, caches, parity, training-on-shared-loops) is real; the neural")
    print("forwards are toys. On a GPU box, swap ComponentSpec.factory for the torch adapters.")
    print("=" * 78)


if __name__ == "__main__":
    main()
