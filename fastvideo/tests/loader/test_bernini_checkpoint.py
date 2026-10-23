# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


def converter():
    path = Path(__file__).resolve().parents[3] / "scripts/checkpoint_conversion/bernini_r_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("bernini_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    root = dict(model_type="bernini_renderer", use_src_id_rotary_emb=True, skip_transformer_1=False,
                skip_transformer_2=True, switch_dit_boundary=0, use_unipc=True)
    (source / "config.json").write_text(json.dumps(root))
    for component in converter().COMPONENTS:
        (source / component).mkdir()
    arch = dict(num_attention_heads=12, num_layers=30, in_channels=16, out_channels=16,
                attention_head_dim=128, ffn_dim=8960, patch_size=[1, 2, 2], _class_name="WanTransformer3DModel")
    (source / "transformer/config.json").write_text(json.dumps(arch))
    (source / "transformer/model.safetensors").write_bytes(b"unchanged tensors")
    return source


@pytest.mark.parametrize("copy", [False, True])
def test_bernini_layout_does_not_change_weights(tmp_path, copy):
    source = make_source(tmp_path)
    output = converter().prepare_checkpoint(source, tmp_path / "converted", copy=copy)
    assert (output / "transformer/model.safetensors").read_bytes() == b"unchanged tensors"
    manifest = json.loads((output / "model_index.json").read_text())
    assert manifest["_class_name"] == "BerniniRPipeline"
    assert manifest["_fastvideo_source"]["revision"] is None
    arch = json.loads((output / "transformer/config.json").read_text())
    assert arch["_class_name"] == "BerniniRTransformer3DModel"
    assert json.loads((source / "transformer/config.json").read_text())["_class_name"] == "WanTransformer3DModel"
    assert (output / "transformer/model.safetensors").is_symlink() is not copy


def test_bernini_refuses_overwrite_and_wrong_model(tmp_path):
    source = make_source(tmp_path)
    output = converter().prepare_checkpoint(source, tmp_path / "converted")
    with pytest.raises(FileExistsError):
        converter().prepare_checkpoint(source, output)
    root = json.loads((source / "config.json").read_text())
    root["skip_transformer_2"] = False
    (source / "config.json").write_text(json.dumps(root))
    with pytest.raises(ValueError, match="single-expert"):
        converter().prepare_checkpoint(source, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_bernini_refuses_incomplete_shards(tmp_path):
    source = make_source(tmp_path)
    (source / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map":{"missing.weight":"absent.safetensors"}}))
    with pytest.raises(ValueError, match="missing weight files"):
        converter().prepare_checkpoint(source, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


@pytest.mark.parametrize("shard", ["../outside.safetensors", "/outside.safetensors", "model.bin", None])
def test_bernini_refuses_nonlocal_shard_names(tmp_path, shard):
    source = make_source(tmp_path)
    (source / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}}))
    with pytest.raises(ValueError, match="local safetensors filenames"):
        converter().prepare_checkpoint(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_bernini_refuses_incompatible_patch_geometry(tmp_path):
    source = make_source(tmp_path)
    config = source / "transformer/config.json"
    arch = json.loads(config.read_text())
    arch["patch_size"] = [1, 1, 1]
    config.write_text(json.dumps(arch))
    with pytest.raises(ValueError, match="patch_size"):
        converter().prepare_checkpoint(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_bernini_output_cannot_be_inside_source(tmp_path):
    source = make_source(tmp_path)
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        converter().prepare_checkpoint(source, source / "converted")
    assert not (source / "converted").exists()


def test_bernini_snapshot_layout_preserves_revision_provenance(tmp_path):
    module = converter()
    snapshot = tmp_path / "snapshots" / module.SOURCE_REVISION
    snapshot.parent.mkdir()
    make_source(tmp_path).rename(snapshot)
    output = module.prepare_checkpoint(snapshot, tmp_path / "converted")
    manifest = json.loads((output / "model_index.json").read_text())
    assert manifest["_fastvideo_source"]["revision"] == module.SOURCE_REVISION


def test_bernini_interrupted_copy_does_not_publish(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    module = converter()

    def fail_copy(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(module.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="full disk"):
        module.prepare_checkpoint(source, tmp_path / "failed", copy=True)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob("bernini-layout-*"))
