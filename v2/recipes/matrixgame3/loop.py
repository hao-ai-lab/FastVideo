"""MatrixGame3DenoiseLoop — the autoregressive multi-clip world-model denoise loop.

Matrix-Game 3.0 is NOT a single bidirectional flow-match pass (that is ``WanDenoiseLoop``). It is an
OUTER clip loop wrapping an INNER few-step flow-match denoise, with cross-clip frame/KV memory. Faithful
port of ``fastvideo/pipelines/stages/matrixgame3_denoising.py:MatrixGame3DenoisingStage.forward``:

  num_iterations = 1                       (for num_frames <= 57; the registered preset is num_frames=57)
                 = 1 + ceil((num_frames-57)/40)   (longer rollouts)
  for clip_idx in range(num_iterations):
    * draw fresh noise for this clip's latent frames; paste the first ``cond_frames`` (1 first clip, 4
      later) from the previous clip's last-4 latents (``img_cond``);
    * (clip>0) build the per-clip action(mouse/keyboard) + camera(Plücker) + KV-memory bundle
      (``x_memory``/``timestep_memory``/``memory_latent_idx``/``predict_latent_idx``) — host numpy work;
    * INNER loop: for each of the ``num_steps`` (distilled=3) FlowUniPC timesteps, call the DiT with a
      PER-TOKEN timestep (cond rows zeroed — built INSIDE the adapter), take a scheduler step, then
      RE-PASTE ``img_cond`` onto the ``cond_frames`` (so the known frames never drift);
    * after the clip: ``img_cond`` <- last-4 latents; append the denoised tail (full first clip,
      last-10 thereafter) to history.
  concat history on the time dim -> the final latent.

THIS PORT — degenerate single-clip path is built so it CPU-verifies on the toy backend:
  * The inner flow-match step uses the platform's numpy ``FLOW_MATCH_STEP`` kernel (the CPU-testable
    stand-in). On a GPU box the *real* ``FlowUniPCMultistepScheduler`` must drive ``step()`` — it is a
    stateful MULTISTEP solver and substituting plain Euler breaks the 3-step distilled trajectory
    (BRINGUP risk 3). The ``self.use_unipc`` hook documents where the real scheduler plugs in.
  * The dit-call is ``dit(latent, text_embed, sigma, cond=<MG3 bundle>)`` — ToyDiT accepts and ignores
    ``cond`` (image/action/camera conditioning is a GPU-path concern), so the toy denoises the noise
    channels exactly like the cosmos2 video2world degradation.
  * Action / camera / KV-memory (mouse_cond/keyboard_cond/plucker/x_memory) are BRINGUP: they need a
    request-API extension to carry action streams + camera trajectories, plus the vendored host helpers
    (``matrixgame3.utils`` + ``lingbotworld.cam_utils``). The single-clip no-action path is the t2v/i2v
    degenerate the loop CPU-verifies; the multi-clip rollout hooks are present + documented.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2._enums import ExecutionProfile, WorkUnitKind
from v2.loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepContext,
    StepResult,
    WorkPlan,
)
from v2.platform import FLOW_MATCH_STEP

# Matrix-Game 3.0 light_vae (Wan2.2-TI2V geometry): z_dim=48, 16x spatial, 4x temporal compression.
MG3_LATENT_CHANNELS = 48
MG3_SPATIAL_RATIO = 16
MG3_TEMPORAL_RATIO = 4
MG3_NUM_TRAIN_TIMESTEPS = 1000  # FlowUniPC timestep ~ sigma * num_train_timesteps

# Clip geometry (hardcoded in the fastvideo denoising stage). first_clip = 57 frames -> 15 latent frames.
MG3_CLIP_FRAME = 56
MG3_FIRST_CLIP_FRAME = MG3_CLIP_FRAME + 1  # 57
MG3_PAST_FRAME = 16
# DiT patch stride on the spatial latent dims (patch_size (1,2,2)). The 5B MatrixGame3WanModel folds
# (H/2, W/2) tokens, so the latent H/W must be patch-aligned (even) before denoise or the unpatchified
# velocity comes back one row/col short of the noise latent (faithful to MatrixGame3DenoisingStage's
# ``latent_h_aligned = (latent_h // patch_h) * patch_h`` crop). 720px -> latent_h 45 (odd) -> 44.
MG3_PATCH_H = 2
MG3_PATCH_W = 2


def _toy_latent_shape(req) -> tuple[int, int, int, int]:
    """Tiny deterministic CPU-toy latent geometry (matches the WanDenoiseLoop toy stand-in), so the loop
    control flow + per-token-timestep/cond-frame/re-paste logic exercise on numbers without a GPU."""
    d = req.diffusion
    from v2.platform.backends.toy import LATENT_CHANNELS
    t = max(2, d.num_frames // 40)
    h = max(2, d.height // 120)
    w = max(2, d.width // 120)
    return (LATENT_CHANNELS, t, h, w)


def _gpu_latent_shape(req, channels: int, spatial_ratio: int, temporal_ratio: int) -> tuple[int, int, int, int]:
    d = req.diffusion
    t = (max(1, d.num_frames) - 1) // temporal_ratio + 1
    h = max(1, d.height // spatial_ratio)
    w = max(1, d.width // spatial_ratio)
    # Patch-align the spatial latent dims to the DiT patch stride (faithful to MatrixGame3DenoisingStage):
    # an odd latent_h/w (e.g. 720px -> 45) would otherwise leave the unpatchified velocity one row short
    # of the noise latent and break the flow-match step's elementwise add.
    h = max(MG3_PATCH_H, (h // MG3_PATCH_H) * MG3_PATCH_H)
    w = max(MG3_PATCH_W, (w // MG3_PATCH_W) * MG3_PATCH_W)
    return (channels, max(1, t), h, w)


def _cond_compatible(latent: np.ndarray, img_cond: np.ndarray) -> bool:
    """The conditioning latent can be pasted onto the leading frames iff its channel + trailing spatial
    dims match the noise latent (true on the GPU 48-channel light_vae; false for the toy VAE stand-in)."""
    return (latent.shape[0] == img_cond.shape[0] and latent.shape[2:] == img_cond.shape[2:])


class MatrixGame3DenoiseLoop:
    """Autoregressive multi-clip world-model denoise. ``next``/``advance`` drive the INNER step; the OUTER
    clip advance happens in ``advance`` when the inner schedule is exhausted (the loop carries the clip
    index + history in ``LoopState.scratch``). On the registered single-clip preset there is exactly one
    clip, so the loop reduces to an N-step flow-match denoise over one latent with first-frame pinning."""

    def __init__(self,
                 *,
                 loop_id,
                 cfg,
                 flow_shift,
                 precision,
                 expert,
                 cost,
                 latent_channels: int = MG3_LATENT_CHANNELS,
                 spatial_ratio: int = MG3_SPATIAL_RATIO,
                 temporal_ratio: int = MG3_TEMPORAL_RATIO,
                 use_unipc: bool = True):
        self.loop_id = loop_id
        self.cfg = cfg  # carried for the WorkPlan op-structure key; CFG is off by default
        self.flow_shift = flow_shift
        self.precision = precision
        self.expert = expert
        self.cost = cost
        self.latent_channels = latent_channels
        self.spatial_ratio = spatial_ratio
        self.temporal_ratio = temporal_ratio
        # GPU path: drive the real FlowUniPCMultistepScheduler.step (stateful multistep) instead of the
        # numpy Euler stand-in (BRINGUP risk 3). The CPU toy always uses FLOW_MATCH_STEP.
        self.use_unipc = use_unipc

    # --- clip-count + per-clip geometry (faithful to MatrixGame3DenoisingStage) ------------------- #
    @staticmethod
    def _num_iterations(num_frames: int) -> int:
        if isinstance(num_frames, int) and num_frames > MG3_FIRST_CLIP_FRAME:
            return 1 + max(0, (num_frames - MG3_FIRST_CLIP_FRAME + 39) // 40)
        return 1

    def init(self, req, model, ctx) -> LoopState:
        seed = req.diffusion.seed if req.diffusion.seed is not None else 0
        rng = np.random.default_rng(seed)
        is_gpu = getattr(getattr(model, "platform", None), "device", "cpu") == "cuda"
        if is_gpu:
            shape = _gpu_latent_shape(req, self.latent_channels, self.spatial_ratio, self.temporal_ratio)
        else:
            shape = _toy_latent_shape(req)
        # Flow-match sigma schedule (1 -> 0) with flow_shift; the FlowUniPC scheduler uses the same shift.
        # ``req.diffusion.sigmas`` (a distilled explicit schedule) overrides when provided.
        sig = self.flow_shift.build_schedule(req.diffusion.num_steps,
                                             req.diffusion.height,
                                             req.diffusion.width,
                                             sigmas=req.diffusion.sigmas or None)
        x = (rng.standard_normal(shape) * float(sig[0])).astype("float32")

        # First-frame conditioning latent (image VAE-encoded by the program). On the degenerate path it is
        # absent -> no pinning, pure flow-match. cond_frames=1 on the first clip (and is clamped to the
        # available conditioning frames). On the GPU path the cond latent shares the noise latent's
        # [C=48, T, h, w] geometry, so the leading frames are pasted in (and re-pasted every step). The
        # CPU toy VAE is only a geometry-preserving stand-in for the channel dims, so ``_cond_compatible``
        # guards the pin: when the trailing dims don't match (toy), it stays the degenerate flow-match.
        img_cond = ctx.slots.get("image_latent")
        cond_frames = 0
        if img_cond is not None:
            img_cond = np.asarray(img_cond, dtype="float32")
            if _cond_compatible(x, img_cond):
                cond_frames = min(1, img_cond.shape[1])  # first clip pins 1 frame
                x[:, :cond_frames] = img_cond[:, :cond_frames]
            else:
                img_cond = None  # toy: incompatible geometry -> no pinning

        st = LoopState(
            loop_id=self.loop_id,
            instance_id=model.card.model_id,
            request_id=req.request_id,
            profile=ctx.profile,
            rng=rng,
            seed=seed,
            latents={"video": x},
            sigmas=[float(s) for s in sig],
            # FlowUniPC model timestep = sigma * num_train_timesteps (the adapter fills the
            # per-token tensor with this value; cond rows zeroed inside the adapter).
            timesteps=[float(s) * MG3_NUM_TRAIN_TIMESTEPS for s in sig])
        st.cond["prompt_embeds"] = ctx.slots.get("text_embeds")
        st.cond["negative_prompt_embeds"] = ctx.slots.get("neg_text_embeds")
        st.scratch["guidance_scale"] = float(req.diffusion.guidance_scale)
        # use_base_model=False + guidance_scale=1.0 (the distilled default) -> the CFG/uncond branch is
        # skipped (BRINGUP risk 8). do_cfg only fires for the base (non-distilled) model.
        st.scratch["use_base_model"] = bool(getattr(req.diffusion, "use_base_model", False))
        st.scratch["img_cond"] = img_cond
        st.scratch["cond_frames"] = cond_frames
        # Latent-frame span the model uses for rotary positions; single-clip span is the whole latent.
        st.scratch["predict_latent_idx"] = (0, int(x.shape[1]))
        # Autoregressive rollout (BRINGUP): num_iterations>1 + the action/camera/memory bundle. The
        # single-clip preset (num_frames<=57) keeps these inert.
        st.scratch["num_iterations"] = self._num_iterations(int(req.diffusion.num_frames))
        st.scratch["clip_idx"] = 0
        st.scratch["history"] = []  # appended denoised clip tails (rollout)
        # Per-clip action/camera/memory conditioning — None on the degenerate path; assembled per clip on
        # the GPU rollout path (needs a request-API extension to carry action streams).
        st.scratch["mg3_cond"] = self._build_clip_cond(st)
        st.scratch["stream_video"] = bool(req.outputs.stream.get("video"))
        st.plugin_state["cfg"] = {}
        return st

    def _build_clip_cond(self, st: LoopState) -> dict:
        """Assemble the per-step MG3 conditioning bundle the adapter consumes. On the degenerate
        single-clip / no-action path this is just ``cond_frames`` + ``predict_latent_idx`` (the action /
        camera / KV-memory slots stay None). BRINGUP: the multi-clip rollout fills ``mouse_cond`` /
        ``keyboard_cond`` / ``c2ws_plucker_emb`` / ``x_memory`` / ``timestep_memory`` /
        ``memory_latent_idx`` here from the vendored host helpers; that path needs a request-API
        extension to carry per-frame action streams + a camera trajectory."""
        return {
            "cond_frames": int(st.scratch.get("cond_frames", 0)),
            "predict_latent_idx": st.scratch.get("predict_latent_idx", (0, 0)),
            "mouse_cond": None,
            "keyboard_cond": None,
            "x_memory": None,
            "timestep_memory": None,
            "mouse_cond_memory": None,
            "keyboard_cond_memory": None,
            "c2ws_plucker_emb": None,
            "memory_latent_idx": None,
        }

    def next(self, st: LoopState):
        i = st.step_idx
        if i >= len(st.sigmas) - 1:
            return Done()  # inner schedule exhausted (single-clip preset -> done)
        sigma_t, sigma_next = st.sigmas[i], st.sigmas[i + 1]
        t = st.timesteps[i]  # FlowUniPC model timestep (~sigma*1000) -> per-token value
        expert_id = self.expert.expert_for(StepContext(i, t, sigma_t))
        x = st.latents["video"]
        pe, ne = st.cond["prompt_embeds"], st.cond["negative_prompt_embeds"]
        scale = st.scratch["guidance_scale"]
        precision = self.precision
        do_cfg = st.scratch.get("use_base_model", False) and scale != 1.0 and ne is not None
        mg3_cond = st.scratch["mg3_cond"]
        img_cond = st.scratch.get("img_cond")
        cond_frames = int(st.scratch.get("cond_frames", 0))

        def _velocity(model: Any, x_: Any, pe_: Any, ne_: Any) -> Any:
            # The conditioned forward + (optional) CFG combine. The adapter builds the per-token timestep
            # + packs the MG3 conditioning bundle internally; the loop hands the raw timestep ``t`` as the
            # ``sigma`` arg (the FlowUniPC convention) so the per-token tensor carries the right value.
            dit = model.component(expert_id)
            cond_pred = dit(x_, pe_, t, cond=mg3_cond)
            if do_cfg:  # base model only (distilled student skips this)
                uncond_pred = dit(x_, ne_, t, cond=mg3_cond)
                combined = np.asarray(uncond_pred) + scale * (np.asarray(cond_pred) - np.asarray(uncond_pred))
                return precision.cast(combined)
            return precision.cast(np.asarray(cond_pred))

        def run(model: Any, override: Any = None) -> StepResult:
            # EAGER thunk. FLOW_MATCH_STEP is the CPU stand-in for FlowUniPC.step (BRINGUP risk 3: the GPU
            # path must drive the real stateful multistep scheduler). After the step, RE-PASTE img_cond on
            # the cond_frames so the known frames never drift (BRINGUP risk 7).
            kernels = model.platform.kernels
            if override is not None and "noise_pred" in override:
                velocity = precision.cast(np.asarray(override["noise_pred"], dtype="float32"))
            else:
                velocity = _velocity(model, x, pe, ne)
            x_next = kernels.get(FLOW_MATCH_STEP)(precision.cast(x), velocity, sigma_t, sigma_next)
            x_next = np.asarray(x_next, dtype="float32")
            if img_cond is not None and cond_frames > 0:
                x_next[:, :cond_frames] = img_cond[:, :cond_frames]  # re-pin the conditioning frames
            return StepResult(output={"noise_pred": np.asarray(velocity, dtype="float32"), "latents": x_next})

        cond_bytes = sum(int(np.asarray(e).nbytes) for e in (pe, ne) if e is not None)
        res = ResourceRequest(compute_seconds=self.cost.predict(int(np.prod(x.shape)), 2.0 if do_cfg else 1.0),
                              resident_bytes=int(x.nbytes) + cond_bytes,
                              peak_activation_bytes=int(x.nbytes))
        return WorkPlan(
            loop_id=self.loop_id,
            instance_id=st.instance_id,
            kind=WorkUnitKind.DIFFUSION_STEP,
            shape_sig=ShapeSignature(WorkUnitKind.DIFFUSION_STEP,
                                     dims=tuple(x.shape),
                                     dtype=precision.compute_dtype,
                                     extra=(("cfg", type(self.cfg).__name__), ("ar_world_model", True))),
            resources=res,
            payload={
                "branch": "mg3",
                "step": i,
                "clip": st.scratch.get("clip_idx", 0)
            },
            run=run,
            label=f"mg3.denoise.c{st.scratch.get('clip_idx', 0)}.{i}",
            # per-token timestep packing + first-frame re-paste + (rollout) host-built conditioning ->
            # the eager path (no static-buffer capture; the cosmos2 video2world precedent).
            capturable=False)

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.latents["video"] = result.output["latents"]
        if st.profile == ExecutionProfile.ROLLOUT:
            st.trajectory.append({
                "step": st.step_idx,
                "sigma": st.sigmas[st.step_idx],
                "velocity": np.asarray(result.output["noise_pred"]).copy(),
                "latents": np.asarray(st.latents["video"]).copy()
            })
        st.step_idx += 1
        # OUTER clip advance (BRINGUP rollout): when the inner schedule is done and more clips remain,
        # snapshot the clip tail into history, slide img_cond -> last-4 latents, and reset the inner
        # step counter + draw fresh noise for the next clip. The single-clip preset never enters this.
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        # Single-clip preset: the final latent is the denoised clip. Multi-clip rollout would concat the
        # history tails on the time dim (BRINGUP).
        return LoopResult(outputs={"latents": st.latents["video"]},
                          metrics={
                              "denoise_steps": float(st.step_idx),
                              "clips": float(st.scratch.get("clip_idx", 0) + 1)
                          },
                          behavior=st.trajectory or None)
