"""Stage every FastH3 input into the personal workspace's volume (CPU only, public repos, no token)."""
import modal

volume = modal.Volume.from_name("h3-pro6000-weights", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]>=0.34")
app = modal.App("h3-pro6000-download-personal", image=image)


@app.function(volumes={"/vol": volume}, timeout=10800, cpu=16, memory=32768)
def download() -> dict:
    import os
    from huggingface_hub import hf_hub_download, snapshot_download
    out = {}
    out["v2"] = snapshot_download("FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4", local_dir="/vol/v2-nvfp4", max_workers=16)
    volume.commit()
    out["v4"] = snapshot_download("FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4", local_dir="/vol/v4-nvfp4",
                                  allow_patterns=["transformer/*", "scheduler/*", "audio_scheduler/*", "*.json",
                                                  "tokenizer/*", "processor/*"], max_workers=16)
    volume.commit()
    out["official_vae"] = snapshot_download("MiniMaxAI/MiniMax-H3", local_dir="/vol/official",
                                            allow_patterns=["vae/*", "model_index.json", "modular_model_index.json"],
                                            max_workers=8)
    out["comfy_int8_vae"] = hf_hub_download("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors",
                                            local_dir="/vol/comfy")
    out["light_vae"] = hf_hub_download("corechan/MiniMax-H3-LightVAE", "lynnreal_light_vae_decoder_fp16.safetensors",
                                       local_dir="/vol/light-vae")
    out["kijai"] = hf_hub_download("Kijai/MiniMax-H3-experimental", "minimax_h3_lynnreal_light_vae_int8_convrot.safetensors",
                                   local_dir="/vol/kijai")
    volume.commit()
    sizes = {}
    for root in ("/vol/v2-nvfp4", "/vol/v4-nvfp4", "/vol/official", "/vol/comfy", "/vol/light-vae", "/vol/kijai"):
        sizes[root] = round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs) / 1e9, 2)
    out["sizes_gb"] = sizes
    return out


@app.local_entrypoint()
def main():
    import json
    print("DOWNLOADED", json.dumps(download.remote(), indent=1))
