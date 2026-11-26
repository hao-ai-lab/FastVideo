import json
import os

from fastvideo import VideoGenerator

# FastPAN2 is an 8-call distilled student of PAN 2 and takes PAN 2's structured captions, serialized as one-line JSON:
# the style, lighting and camera, each subject, the first frame, and timed events with their camera motion and
# actions. PAN 2's prompt rewriter (`inference/pan-prompt-rewriter` in ifm-ai/PAN2) turns a plain description into one.
# This caption describes a 5 s clip; its event times span the video.
CAPTION = json.loads("""
{
  "video_style": "Realistic nature cinematography in one uninterrupted continuous shot.",
  "lighting": "Low golden morning sun from the left, long soft shadows across fresh snow and a warm rim light on the fox's fur.",
  "perspective_view": "Low eye-level side view from the snow surface, tracking right alongside the fox at a steady distance.",
  "subjects_lifecycle": [
    {
      "subject_id": "<|Subject1|>",
      "display_name": "Red fox",
      "subject_attribute": "Adult red fox with a thick orange winter coat, white chest and tail tip, and black lower legs."
    }
  ],
  "init_frame_visual_description": "A red fox stands at the left of the frame in knee-deep powder snow at the edge of a pine forest, facing right. Dark pine trunks rise behind it and the snow ahead is smooth and untouched, glowing gold in the low sun. Small ice crystals drift in the air.",
  "events": [
    {
      "event_index": 0,
      "time_range_sec": "0 to 2.5",
      "event_description": "The fox trots to the right through the snow, lifting its legs high and leaving a line of deep prints behind it. Powder sprays from its paws with each step while the camera tracks right beside it.",
      "camera_motion": [
        {
          "time_range_sec": "0 to 2.5",
          "camera_motion": [
            "track_right"
          ]
        }
      ],
      "contained_actions": [
        {
          "time_range_sec": "0 to 2.5",
          "action": "<|Subject1|> trots forward through deep snow with high, springy steps."
        }
      ]
    },
    {
      "event_index": 1,
      "time_range_sec": "2.5 to 5",
      "event_description": "The fox slows, stops and turns its head toward the camera, ears upright, its breath visible as small clouds in the cold air. The camera eases to a stop and holds on the fox.",
      "camera_motion": [
        {
          "time_range_sec": "2.5 to 3.5",
          "camera_motion": [
            "track_right"
          ]
        },
        {
          "time_range_sec": "3.5 to 5",
          "camera_motion": [
            "static"
          ]
        }
      ],
      "contained_actions": [
        {
          "time_range_sec": "2.5 to 5",
          "action": "<|Subject1|> comes to a halt, turns its head toward the camera and breathes out visible clouds."
        }
      ]
    }
  ],
  "quality": {
    "video_quality": "Sharp fur and snow detail with natural motion blur on the spraying powder.",
    "text_logos_overlays": null,
    "artifacts": null
  }
}
""")
PROMPT = json.dumps(CAPTION, ensure_ascii=False, separators=(",", ":"))
# Frames = 4 * ceil(seconds * 24 / 4) + 1 at 24 fps: 121 for 5 s, 241 for 10 s, 481 for 20 s.
NUM_FRAMES = 121

OUTPUT_PATH = "video_samples_fastpan2/fastpan2_t2v_5s.mp4"
MODEL_PATH = os.environ.get("FASTPAN2_MODEL_PATH", "IFM/FastPAN2")

# A first frame for image-to-video; None runs text-to-video.
IMAGE_PATH = None


def main():
    # The checkpoint's fastvideo_inference.json fixes the sampler: 8 transformer calls, no classifier-free guidance
    # and VSA2 sparse attention, which needs a B200 / GB200 / B300 GPU. With more GPUs, num_gpus enables Ulysses
    # sequence parallelism (the 32 attention heads must be divisible by it).
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=int(os.environ.get("FASTPAN2_NUM_GPUS", "1")),
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    # No num_inference_steps, guidance_scale or negative prompt: the student's sampler comes from the checkpoint.
    generator.generate_video(
        PROMPT,
        image_path=IMAGE_PATH,
        height=704,
        width=1248,
        num_frames=NUM_FRAMES,
        seed=1234,
        output_path=OUTPUT_PATH,
        save_video=True,
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
