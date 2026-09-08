# SPDX-License-Identifier: Apache-2.0
"""PromptRL bundle export + PromptRefiner round-trip tests."""

from __future__ import annotations

import json
import os

import pytest
import torch

from fastvideo.train.methods.rl.promptrl.bundle import (
    BundleManifest,
    export_promptrl_bundle,
    extract_generator_lora,
    load_bundle_manifest,
    training_name_to_hf,
)
from fastvideo.train.methods.rl.promptrl.inference import PromptRefiner
from tests.local_tests.promptrl_fixtures import (
    build_tiny_refiner,
    build_tiny_tokenizer,
)


def _manifest() -> BundleManifest:
    return BundleManifest(
        base_refiner_model="Qwen/Qwen2.5-VL-3B-Instruct",
        base_generator_model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        fastvideo_version="0.2.0",
        refiner_lora={"rank": 16, "alpha": 32,
                      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        generator_lora={"rank": 16, "alpha": 32,
                        "target_modules": ["to_q", "to_k", "to_v", "to_out"]},
        prompt_template_version="v1",
        refiner_sampling={"max_new_tokens": 256, "temperature": 1.0, "top_p": 1.0},
        mode="joint",
    )


class TestManifest:
    def test_round_trip(self, tmp_path):
        manifest = _manifest()
        bundle_dir = str(tmp_path / "bundle")
        os.makedirs(bundle_dir)
        with open(os.path.join(bundle_dir, "manifest.json"), "w") as handle:
            json.dump(manifest.to_dict(), handle)
        loaded = load_bundle_manifest(bundle_dir)
        assert loaded == manifest
        assert loaded.prompt_template_version == "v1"
        assert loaded.refiner_lora["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValueError, match="Unknown manifest keys"):
            BundleManifest.from_dict({**_manifest().to_dict(), "surprise": 1})

    def test_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bundle_manifest(str(tmp_path))


class TestGeneratorLoRAExtraction:
    def _tiny_wan_like_transformer(self) -> torch.nn.Module:
        from fastvideo.layers.lora.linear import ReplicatedLinear
        from fastvideo.train.utils.lora import enable_lora_training

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.ModuleDict({
                    "to_q": ReplicatedLinear(8, 8),
                    "to_out": ReplicatedLinear(8, 8),
                })
                self.cross_attn = torch.nn.ModuleDict({
                    "to_v": ReplicatedLinear(8, 8),
                })
                self.ffn = torch.nn.ModuleDict({
                    "fc_in": ReplicatedLinear(8, 8),
                })

        class FakeWan(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(), Block()])

        transformer = FakeWan()
        enable_lora_training(
            transformer,
            lora_rank=2,
            lora_alpha=4,
            lora_target_modules=["to_q", "to_out", "to_v", "fc_in"],
        )
        return transformer

    def test_name_mapping(self):
        assert (training_name_to_hf("blocks.3.self_attn.to_q.lora_A") ==
                "blocks.3.attn1.to_q.lora_A")
        assert (training_name_to_hf("blocks.0.cross_attn.to_out.lora_B") ==
                "blocks.0.attn2.to_out.0.lora_B")
        assert (training_name_to_hf("blocks.12.ffn.fc_in.lora_A") ==
                "blocks.12.ffn.net.0.proj.lora_A")

    def test_extract_and_save(self, tmp_path):
        from safetensors.torch import load_file

        transformer = self._tiny_wan_like_transformer()
        state = extract_generator_lora(transformer)
        # 2 blocks x (to_q + to_out) + 2 cross-attn to_v + 2 ffn fc_in = 8 layers
        layer_names = {key.rsplit(".", 1)[0] for key in state}
        assert len(layer_names) == 8
        assert "blocks.0.attn1.to_q" in layer_names
        assert "blocks.1.attn2.to_v" in layer_names
        assert "blocks.1.ffn.net.0.proj" in layer_names
        for name in layer_names:
            assert f"{name}.lora_A" in state
            assert f"{name}.lora_B" in state
            assert f"{name}.lora_alpha" in state
            assert state[f"{name}.lora_alpha"].item() == 4.0

        written = export_promptrl_bundle(
            str(tmp_path / "bundle"),
            manifest=_manifest(),
            refiner_adapter_dir=self._save_tiny_adapter(tmp_path),
            generator_transformer=transformer,
        )
        assert os.path.isfile(written["generator"])
        saved = load_file(written["generator"])
        assert set(saved) == set(state)

    def _save_tiny_adapter(self, tmp_path) -> str:
        role = build_tiny_refiner()
        adapter_dir = str(tmp_path / "tiny_adapter")
        role.save_adapter(adapter_dir)
        return adapter_dir

    def test_no_lora_layers_raises(self):
        with pytest.raises(ValueError, match="no LoRA layers"):
            extract_generator_lora(torch.nn.Linear(4, 4))


class TestPromptRefinerRoundTrip:
    def test_adapter_round_trip_and_refine(self, tmp_path):
        role = build_tiny_refiner()
        # Perturb the adapter so it differs from the zero-init state.
        with torch.no_grad():
            for name, param in role.model.named_parameters():
                if "lora_B" in name:
                    param.add_(torch.randn_like(param) * 0.05)
        before = {name: param.detach().clone()
                  for name, param in role.model.named_parameters() if "lora_" in name}

        written = export_promptrl_bundle(
            str(tmp_path / "bundle"),
            manifest=_manifest(),
            refiner_role=role,
            refiner_tokenizer=role.tokenizer,
            generator_transformer=None,
        )
        assert os.path.isfile(os.path.join(written["refiner"], "adapter_model.safetensors"))
        assert os.path.isfile(os.path.join(written["refiner"], "adapter_config.json"))
        assert os.path.isfile(os.path.join(written["refiner"], "tokenizer_config.json"))

        # Load the bundle with a fresh base model; injected for tests.
        from tests.local_tests.promptrl_fixtures import build_tiny_model

        from peft import PeftModel

        tokenizer = build_tiny_tokenizer()
        base = build_tiny_model(len(tokenizer))
        base = PeftModel.from_pretrained(base, written["refiner"])
        refiner = PromptRefiner.from_bundle(
            str(tmp_path / "bundle"),
            model=base,
            tokenizer=tokenizer,
            device="cpu",
        )
        after = {name: param.detach().clone()
                 for name, param in refiner.model.named_parameters() if "lora_" in name}
        assert set(before) == set(after)
        for name in before:
            assert torch.allclose(before[name], after[name], atol=1e-6), name

        result = refiner.refine("a cat", seed=0)
        assert result.original_prompt == "a cat"
        assert isinstance(result.raw_completion, str)
        assert isinstance(result.format_valid, bool)
        # Fallback contract: invalid format returns the original prompt.
        if not result.format_valid:
            assert result.refined_prompt == "a cat"

    def test_refine_fallback_on_malformed(self, tmp_path):
        role = build_tiny_refiner()
        written = export_promptrl_bundle(
            str(tmp_path / "bundle"),
            manifest=_manifest(),
            refiner_role=role,
            refiner_tokenizer=role.tokenizer,
        )
        from tests.local_tests.promptrl_fixtures import build_tiny_model

        from peft import PeftModel

        tokenizer = build_tiny_tokenizer()
        base = PeftModel.from_pretrained(build_tiny_model(len(tokenizer)), written["refiner"])
        refiner = PromptRefiner.from_bundle(
            str(tmp_path / "bundle"),
            model=base,
            tokenizer=tokenizer,
            device="cpu",
        )

        # Force a malformed completion by stubbing generate/decode.
        refiner.model.generate = lambda **kwargs: torch.tensor([[0] * 5 + [2]])  # type: ignore[method-assign]
        refiner.tokenizer.batch_decode = lambda ids, **kwargs: ["no tags here"]  # type: ignore[method-assign]
        result = refiner.refine("a cat")
        assert not result.format_valid
        assert result.refined_prompt == "a cat"
        assert result.raw_completion == "no tags here"
