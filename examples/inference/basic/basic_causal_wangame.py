from fastvideo import VideoGenerator
from fastvideo.configs.pipelines import SelfForcingWanGameI2V480PConfig
from fastvideo.models.dits.matrixgame.utils import create_action_presets
import torch

BASE_MODEL_PATH = "Wan2.1-Fun-1.3B-InP-Diffusers"
WEIGHTS_PATH = "checkpoints/wangame_ode_init/checkpoint-1200/transformer"

OUTPUT_PATH = "video_samples_wangame"
IMAGE_PATH = "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png"


def main():
    generator = VideoGenerator.from_pretrained(
        BASE_MODEL_PATH,
        pipeline_config=SelfForcingWanGameI2V480PConfig(),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        override_pipeline_cls_name="WanGameCausalDMDPipeline",
        override_transformer_cls_name="CausalWanGameActionTransformer3DModel",
        init_weights_from_safetensors=WEIGHTS_PATH,
    )

    num_frames = 81
    actions = create_action_presets(num_frames, keyboard_dim=4)
    actions["keyboard"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_frames)
    actions["mouse"] = torch.tensor([[0.0, 0.0]] * num_frames)

    generator.generate_video(
        prompt="",
        image_path=IMAGE_PATH,
        mouse_cond=actions["mouse"].unsqueeze(0),
        keyboard_cond=actions["keyboard"].unsqueeze(0),
        num_frames=num_frames,
        height=352,
        width=640,
        num_inference_steps=40,
        guidance_scale=1.0,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
