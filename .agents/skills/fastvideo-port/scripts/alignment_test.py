"""Numerical alignment test template for FastVideo model ports.

Usage:
    FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA python alignment_test.py \
        --component dit \
        --official_weights official_weights/<model_name>/ \
        --model_name <ModelName>

Copy this file to tests/local_tests/<model_name>/test_<component>_alignment.py
and fill in the TODOs for each component.
"""
import argparse

import torch
import torch.testing

SEED = 42
ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# TODO: replace these imports with actual classes once implemented
# ---------------------------------------------------------------------------
# from <official_repo>.<module> import OfficialModel
# from fastvideo.models.dits.<model_name> import FastVideoModel
# from fastvideo.configs.models.dits.<model_name> import param_names_mapping
# ---------------------------------------------------------------------------


def load_official(weights_path: str, device: torch.device, dtype: torch.dtype):
    """Load the official model with its original weights."""
    # TODO: instantiate the official model class
    # model = OfficialModel(...)
    # state_dict = torch.load(weights_path, map_location=device)
    # model.load_state_dict(state_dict)
    # model.to(device=device, dtype=dtype).eval()
    # return model
    raise NotImplementedError("Fill in official model loading")


def load_fastvideo(weights_path: str, device: torch.device, dtype: torch.dtype):
    """Load the FastVideo model, applying param_names_mapping."""
    # TODO: instantiate the FastVideo model class and apply weight mapping
    # model = FastVideoModel(...)
    # raw_sd = torch.load(weights_path, map_location=device)
    # mapped_sd = apply_param_names_mapping(raw_sd, param_names_mapping)
    # model.load_state_dict(mapped_sd, strict=True)
    # model.to(device=device, dtype=dtype).eval()
    # return model
    raise NotImplementedError("Fill in FastVideo model loading")


def make_inputs(batch_size: int, device: torch.device, dtype: torch.dtype) -> dict:
    """Generate fixed random inputs. Adjust shapes for the target component."""
    torch.manual_seed(SEED)
    # TODO: replace with correct shapes for the component under test
    return {
        # DiT example — adjust to actual signature
        "hidden_states": torch.randn(batch_size, 16, 8, 16, 16, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(batch_size, 77, 4096, device=device, dtype=dtype),
        "timestep": torch.tensor([500] * batch_size, device=device, dtype=torch.long),
    }


def run_official(model, inputs: dict):
    """Forward pass through the official model. Adjust to its API."""
    with torch.no_grad():
        # TODO: adapt to official model's forward signature
        return model(**inputs)


def run_fastvideo(model, inputs: dict):
    """Forward pass through the FastVideo model."""
    with torch.no_grad():
        # TODO: adapt to FastVideo model's forward signature
        return model(**inputs)


def compare(official_out, fastvideo_out, component: str) -> None:
    """Assert numerical closeness and print a summary."""
    if isinstance(official_out, (tuple, list)):
        official_out = official_out[0]
    if isinstance(fastvideo_out, (tuple, list)):
        fastvideo_out = fastvideo_out[0]

    max_diff = (official_out - fastvideo_out).abs().max().item()
    mean_diff = (official_out - fastvideo_out).abs().mean().item()
    print(f"[{component}] max_abs_diff={max_diff:.2e}  mean_abs_diff={mean_diff:.2e}")

    torch.testing.assert_close(
        fastvideo_out,
        official_out,
        atol=ATOL,
        rtol=RTOL,
        msg=f"{component} parity failed: max diff {max_diff:.2e} > atol {ATOL}",
    )
    print(f"[{component}] PASS — outputs match within atol={ATOL}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["dit", "vae_encoder", "vae_decoder", "text_encoder"],
                        default="dit")
    parser.add_argument("--official_weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print(f"Testing {args.component} alignment | device={device} dtype={dtype}")

    official = load_official(args.official_weights, device, dtype)
    fastvideo = load_fastvideo(args.official_weights, device, dtype)

    inputs = make_inputs(args.batch_size, device, dtype)
    official_out = run_official(official, inputs)
    fastvideo_out = run_fastvideo(fastvideo, inputs)

    compare(official_out, fastvideo_out, args.component)


if __name__ == "__main__":
    main()
