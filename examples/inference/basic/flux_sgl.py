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



# [prompt1] frame0 slice (H4xW4xC3):
# [[[68 62 46]
#   [65 56 40]
#   [65 56 39]
#   [65 57 40]]

#  [[65 58 40]
#   [67 59 40]
#   [63 56 38]
#   [64 56 38]]

#  [[60 53 37]
#   [63 55 38]
#   [63 56 38]
#   [64 57 38]]

#  [[62 55 38]
#   [63 56 38]
#   [64 56 38]
#   [64 56 38]]]
# [02-15 13:21:49] Processing prompt 1/1: A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afte
# [02-15 13:21:49] Sampling params:
#                        width: -1
#                       height: -1
#                   num_frames: 1
#                       prompt: A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic.
#                   neg_prompt: None
#                         seed: 42
#                  infer_steps: 50
#       num_outputs_per_prompt: 1
#               guidance_scale: 1.0
#      embedded_guidance_scale: 3.5
#                     n_tokens: None
#                   flow_shift: None
#                   image_path: None
#                  save_output: False
#             output_file_path: image_samples/A_majestic_lion_strides_across_the_golden_savanna_its_powerful_frame_glistening_under_the_warm_afte_20260215-132149_3e696704.png
        
# [02-15 13:21:49] Running pipeline stages: ['input_validation_stage', 'prompt_encoding_stage_primary', 'conditioning_stage', 'timestep_preparation_stage', 'latent_preparation_stage', 'denoising_stage', 'decoding_stage']
# [02-15 13:21:49] [InputValidationStage] started...
# [02-15 13:21:49] [InputValidationStage] finished in 0.0000 seconds
# [02-15 13:21:49] [TextEncodingStage] started...
# [02-15 13:21:50] [TextEncodingStage] finished in 0.0506 seconds
# [02-15 13:21:50] [ConditioningStage] started...
# [02-15 13:21:50] [ConditioningStage] finished in 0.0000 seconds
# [02-15 13:21:50] [TimestepPreparationStage] started...
# [02-15 13:21:50] [TimestepPreparationStage] finished in 0.0005 seconds
# [02-15 13:21:50] [LatentPreparationStage] started...
# [02-15 13:21:50] [LatentPreparationStage] finished in 0.0002 seconds
# [02-15 13:21:50] [DenoisingStage] started...
# 100%|████████████████████████████████████████████████████████████████████████████| 50/50 [00:35<00:00,  1.41it/s]
# [02-15 13:22:25] [DenoisingStage] average time per step: 0.7076 seconds
# [02-15 13:22:25] [DenoisingStage] finished in 35.3833 seconds
# [02-15 13:22:25] [DecodingStage] started...
# [02-15 13:22:25] [DecodingStage] finished in 0.1041 seconds
# [02-15 13:22:25] Peak GPU memory: 25.81 GB, Remaining GPU memory at peak: 19.18 GB. Components that can stay resident: ['text_encoder', 'text_encoder_2']
# [02-15 13:22:26] Pixel data generated successfully in 36.82 seconds
# [02-15 13:22:26] Completed batch processing. Generated 1 outputs in 36.82 seconds
# [02-15 13:22:26] Memory usage - Max peak: 26426.31 MB, Avg peak: 26426.31 MB
# [prompt2] frame0 shape=(720, 1280, 3) dtype=uint8 min=0 max=255 mean=109.69557798032407
# [prompt2] frame0 slice (H4xW4xC3):
# [[[250 248 182]
#   [252 244 174]
#   [252 243 180]
#   [252 242 179]]

#  [[250 243 168]
#   [251 244 173]
#   [250 242 180]
#   [251 242 178]]

#  [[251 243 176]
#   [251 241 178]
#   [252 241 182]
#   [253 242 180]]

#  [[251 242 179]
#   [252 241 180]
#   [253 241 181]
#   [253 241 178]]]