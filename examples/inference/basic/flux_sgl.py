import numpy as np  
from sglang.multimodal_gen import DiffGenerator  
  
OUTPUT_PATH = "image_samples"  
  
def _print_frame_matrix(frames, label: str) -> None:  
    if not frames:  
        print(f"[{label}] No frames returned")  
        return  
    frame0 = frames[0]  
    if isinstance(frame0, np.ndarray):  
        arr = frame0  
    else:  
        arr = np.array(frame0)  
  
    print(  
        f"[{label}] frame0 shape={arr.shape} dtype={arr.dtype} "  
        f"min={arr.min()} max={arr.max()} mean={arr.mean()}"  
    )  
  
    if arr.ndim >= 2:  
        h = min(4, arr.shape[0])  
        w = min(4, arr.shape[1])  
        if arr.ndim == 3:  
            c = min(3, arr.shape[2])  
            print(f"[{label}] frame0 slice (H{h}xW{w}xC{c}):\n{arr[:h, :w, :c]}")  
        else:  
            print(f"[{label}] frame0 slice (H{h}xW{w}):\n{arr[:h, :w]}")  
  
def main():  
    # Initialize DiffGenerator for FLUX.1-dev  
    generator = DiffGenerator.from_pretrained(  
        model_path="black-forest-labs/FLUX.1-dev",  
        num_gpus=1,  
        dit_cpu_offload=False,  
        vae_cpu_offload=False,  
        text_encoder_cpu_offload=True,  
        pin_cpu_memory=True,  
    )  
  
    # First generation  
    prompt = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."  
    result = generator.generate(  
        sampling_params_kwargs=dict(  
            prompt=prompt,  
            return_frames=True,  
            save_output=False,  # Skip disk save to focus on tensors  
            output_path=OUTPUT_PATH,  
        )  
    )  
    frames = result if isinstance(result, list) else result.get("frames", [])  
    _print_frame_matrix(frames, "prompt1")  
  
    # Second generation (model stays loaded)  
    prompt2 = (  
        "A majestic lion strides across the golden savanna, its powerful frame "  
        "glistening under the warm afternoon sun. The tall grass ripples gently in "  
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "  
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "  
        "cinematic."  
    )  
    result2 = generator.generate(  
        sampling_params_kwargs=dict(  
            prompt=prompt2,  
            return_frames=True,  
            save_output=False,  
            output_path=OUTPUT_PATH,  
        )  
    )  
    frames2 = result2 if isinstance(result2, list) else result2.get("frames", [])  
    _print_frame_matrix(frames2, "prompt2")  
  
if __name__ == "__main__":  
    main()
