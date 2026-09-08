# SPDX-License-Identifier: Apache-2.0
"""Wan training role extended for PromptRL.

Adds to :class:`~fastvideo.train.models.wan.wan.WanModel`:

* persistent UMT5 text encoder + tokenizer for online prompt encoding
  (the refiner produces prompts at train time, so preprocessed text
  embeddings cannot be used),
* stochastic rollouts: ``num_steps`` flow-matching steps of which the
  last ``sde_steps`` are stochastic SDE transitions whose states and
  behavior-policy log probabilities are stored for the joint loss,
* transition log-probability recomputation under the current adapter
  and under the adapter-disabled frozen reference policy,
* video decoding (inherited ``decode_latents``).

The VAE and text encoder stay frozen and persist for the whole run.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from collections.abc import Iterator

import torch

from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.rl.promptrl.sde import (
    SDETransition,
    sde_step_from_model_output,
    transition_log_prob,
)
from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.lora import lora_disabled
from fastvideo.train.utils.moduleloader import (
    load_module_from_path,
    make_inference_args,
)
from fastvideo.utils import maybe_download_model

logger = init_logger(__name__)


@dataclass(slots=True)
class WanRolloutResult:
    """Rollout output: final latents + stored stochastic transitions."""

    latents: torch.Tensor  # [B, T, C, H, W] final denoised latents
    transitions: list[SDETransition] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    timesteps: list[float] = field(default_factory=list)


class WanPromptRLModel(WanModel):
    """Wan per-role model with PromptRL rollout support."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_preprocessors(self, training_config) -> None:
        """Load VAE + persistent UMT5; skip the parquet dataloader.

        PromptRL prompts arrive online from the refiner, so the
        preprocessed-parquet dataloader built by the base class is not
        applicable here.
        """
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()
        # Rollouts are conditional-only (no CFG); skip the negative
        # prompt encoding the base class triggers in on_train_start.
        self.set_requires_negative_conditioning(False)
        self._load_prompt_text_encoder(training_config)

    def _load_prompt_text_encoder(self, training_config) -> None:
        from transformers import AutoTokenizer

        from fastvideo.models.loader.component_loader import TextEncoderLoader

        tc = training_config
        if tc.pipeline_config is None:
            raise ValueError("WanPromptRLModel requires training_config.pipeline_config "
                             "for text encoder settings")
        model_path = maybe_download_model(str(tc.model_path))
        inference_args = make_inference_args(tc, model_path=model_path)
        inference_args.text_encoder_cpu_offload = False
        loader = TextEncoderLoader()
        self.text_encoder = loader.load(
            os.path.join(model_path, "text_encoder"),
            inference_args,
        ).to(self.device).eval()
        self.text_encoder.requires_grad_(False)
        self.text_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        self._text_encoder_config = tc.pipeline_config.text_encoder_configs[0]
        self._postprocess_text = tc.pipeline_config.postprocess_text_funcs[0]
        preprocess_funcs = getattr(tc.pipeline_config, "preprocess_text_funcs", None)
        self._preprocess_text = (preprocess_funcs[0] if preprocess_funcs else None)
        logger.info("Loaded persistent Wan text encoder for online prompt encoding")

    # ------------------------------------------------------------------
    # Online prompt encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw prompts with the persistent UMT5 encoder.

        Returns ``(encoder_hidden_states, encoder_attention_mask)`` in
        the training dtype on this rank's device.
        """
        device = self.device
        dtype = self._get_training_dtype()
        tok_kwargs = dict(self._text_encoder_config.tokenizer_kwargs)
        texts = [self._preprocess_text(p) if self._preprocess_text is not None else p
                 for p in prompts]
        with set_forward_context(current_timestep=0, attn_metadata=None):
            text_inputs = self.text_tokenizer(texts, **tok_kwargs).to(device)
            outputs = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
            outputs.attention_mask = text_inputs["attention_mask"]
            embeds = self._postprocess_text(outputs).to(device=device, dtype=dtype)
            mask = text_inputs["attention_mask"].to(device=device, dtype=dtype)
        return embeds, mask

    # ------------------------------------------------------------------
    # Rollout batch
    # ------------------------------------------------------------------

    def latent_shape(self, batch_size: int) -> tuple[int, int, int, int, int]:
        """``[B, T, C, H, W]`` latent shape from the training config."""
        tc = self.training_config
        assert tc is not None
        vae_config = tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
        return (
            int(batch_size),
            int(tc.data.num_latent_t),
            int(vae_config.z_dim),
            int(tc.data.num_height) // int(vae_config.spatial_compression_ratio),
            int(tc.data.num_width) // int(vae_config.spatial_compression_ratio),
        )

    def build_rollout_batch(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        *,
        latent_shape: tuple[int, ...],
    ) -> TrainingBatch:
        """Minimal TrainingBatch for rollout/recompute forwards."""
        batch = TrainingBatch()
        batch.conditional_dict = {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        # raw_latent_shape follows the [B, C, T, H, W] convention used
        # by _prepare_dit_inputs before its permutation.
        batch.raw_latent_shape = (latent_shape[0], latent_shape[2], latent_shape[1],
                                  latent_shape[3], latent_shape[4])
        batch.timesteps = torch.zeros(latent_shape[0], device=self.device)
        batch = self._build_attention_metadata(batch)
        return batch

    # ------------------------------------------------------------------
    # Adapter-disabled reference context
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        """Disable the Wan LoRA adapter (frozen reference policy)."""
        with lora_disabled(self.transformer):
            yield


    # ------------------------------------------------------------------
    # Stochastic rollouts
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        batch: TrainingBatch,
        *,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
        num_steps: int = 20,
        sde_steps: int = 8,
        noise_scale: float = 0.8,
        flow_shift: float | None = None,
        store_transitions: bool = True,
    ) -> WanRolloutResult:
        """Denoise a fresh noise sample with an ODE/SDE hybrid schedule.

        Steps are deterministic Euler except the last ``sde_steps``,
        which take stochastic SDE transitions.  When
        ``store_transitions`` is set (joint mode), the stochastic
        transitions' states and behavior-policy log probs are kept for
        the flow-policy loss; prompt-only mode skips storage entirely.
        """
        import copy

        from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler, )

        device = self.device
        dtype = self._get_training_dtype()
        shape = self.latent_shape(batch_size)
        scheduler = copy.deepcopy(self.noise_scheduler)
        if flow_shift is not None:
            scheduler = FlowMatchEulerDiscreteScheduler(shift=float(flow_shift))
        scheduler.set_timesteps(num_inference_steps=int(num_steps), device=device)
        sigmas = scheduler.sigmas
        timesteps = scheduler.timesteps
        sigma_max = float(sigmas[1]) if sigmas.numel() > 1 else float(sigmas[0])

        current = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        transitions: list[SDETransition] = []
        sde_start = int(num_steps) - int(sde_steps)
        for step_idx in range(int(num_steps)):
            sigma = float(sigmas[step_idx])
            sigma_next = float(sigmas[step_idx + 1])
            timestep = timesteps[step_idx].reshape(1).to(device).expand(shape[0])
            batch.timesteps = timestep
            model_output = self.predict_noise(
                current,
                timestep,
                batch,
                conditional=True,
                attn_kind="dense",
            )
            stochastic = store_transitions and step_idx >= sde_start
            if stochastic:
                prev_sample, log_prob, _ = sde_step_from_model_output(
                    model_output,
                    current,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    noise_scale=noise_scale,
                    sigma_max=sigma_max,
                    generator=generator,
                )
                transitions.append(
                    SDETransition(
                        sample=current.detach().clone(),
                        prev_sample=prev_sample.detach().clone(),
                        timestep=float(timesteps[step_idx]),
                        sigma=sigma,
                        sigma_next=sigma_next,
                        old_log_prob=log_prob.detach(),
                    ))
                current = prev_sample.to(dtype)
            else:
                current = (current.float() +
                           (sigma_next - sigma) * model_output.float()).to(dtype)

        return WanRolloutResult(
            latents=current.detach(),
            transitions=transitions,
            sigmas=[float(s) for s in sigmas],
            timesteps=[float(t) for t in timesteps],
        )

    # ------------------------------------------------------------------
    # Transition log-probability recomputation
    # ------------------------------------------------------------------

    def transition_logprobs(
        self,
        transitions: list[SDETransition],
        batch: TrainingBatch,
        *,
        noise_scale: float,
        sigma_max: float,
        use_adapter: bool = True,
        requires_grad: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Recompute per-transition ``(log_prob, prev_mean)``.

        ``use_adapter=False`` evaluates the frozen reference policy via
        the adapter-disabled context; ``requires_grad=True`` keeps the
        computation differentiable for the flow-policy loss.
        """
        adapter_context = (contextlib.nullcontext() if use_adapter else self.adapter_disabled())
        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        results: list[tuple[torch.Tensor, torch.Tensor]] = []
        with adapter_context, grad_context:
            for transition in transitions:
                timestep = torch.full(
                    (transition.sample.shape[0], ),
                    float(transition.timestep),
                    device=self.device,
                )
                batch.timesteps = timestep
                model_output = self.predict_noise(
                    transition.sample,
                    timestep,
                    batch,
                    conditional=True,
                    attn_kind="dense",
                )
                log_prob, prev_mean = transition_log_prob(
                    model_output,
                    transition.sample,
                    transition.prev_sample,
                    sigma=transition.sigma,
                    sigma_next=transition.sigma_next,
                    noise_scale=noise_scale,
                    sigma_max=sigma_max,
                )
                results.append((log_prob, prev_mean))
        return results
