import json
import os

from fastvideo import VideoGenerator

# PAN2 is prompted with a structured caption, serialized as one-line JSON: the style, lighting and camera, each
# subject, the first frame, and timed events with their camera motion and actions. Its prompt rewriter
# (`inference/pan-prompt-rewriter` in ifm-ai/PAN2) turns a plain description into one. This is the release's own
# text-to-video example (`inference/prompts/examples.json`), a 10 s clip whose event times span the video.
CAPTION = json.loads("""
{
  "video_style": "Realistic sports cinematography in one uninterrupted continuous shot.",
  "lighting": "Direct morning sunlight from the pool's far side; bright surface reflections, natural skin color and blue-green underwater light with moving caustics.",
  "perspective_view": "Low side three-quarter view from an adjacent empty lane, tracking right alongside the swimmer. The waterproof camera descends through the surface after entry, follows underwater, then rises above water while maintaining the same side and direction.",
  "subjects_lifecycle": [
    {
      "subject_id": "<|Subject1|>",
      "display_name": "Racing swimmer",
      "subject_attribute": "Athletic man in his early 30s, clean-shaven, wearing a white silicone cap, dark racing goggles and navy knee-length racing jammers."
    }
  ],
  "init_frame_visual_description": "The swimmer crouches on a white starting block at the left of an outdoor competition pool, facing right along his lane. His fingers curl over the block's front edge, hips raised and feet staggered. The low camera sits just above the water in the neighboring lane, framing his entire starting posture and several meters of clear water ahead. Blue-and-white lane ropes converge toward the distant end wall. Sunlight defines his shoulders and the matte fabric of his jammers; small surface ripples break the reflection of the pale pool deck, with a dark floor guide visible beneath them.",
  "events": [
    {
      "event_index": 0,
      "time_range_sec": "0 to 2.2",
      "event_description": "He drives off the block, extends his arms beyond his head and enters fingertips first farther down the lane. His torso and pointed feet follow through the opening, leaving a brief upright splash behind him. He lengthens into a submerged glide as the camera follows right and descends through the rippling surface beside his entry trail.",
      "camera_motion": [
        {
          "time_range_sec": "0 to 0.3",
          "camera_motion": [
            "static"
          ]
        },
        {
          "time_range_sec": "0.3 to 1.5",
          "camera_motion": [
            "track_right"
          ]
        },
        {
          "time_range_sec": "1.5 to 2.2",
          "camera_motion": [
            "track_right",
            "dolly_down"
          ]
        }
      ],
      "contained_actions": [
        {
          "time_range_sec": "0.2 to 1.5",
          "action": "<|Subject1|> pushes from the block, dives forward with arms extended and enters the water hands first, followed by torso and feet."
        },
        {
          "time_range_sec": "1.5 to 2.2",
          "action": "<|Subject1|> glides underwater in a long streamlined shape, carrying forward momentum from the dive."
        }
      ]
    },
    {
      "event_index": 1,
      "time_range_sec": "2.2 to 5.8",
      "event_description": "Seen fully underwater, he adds two compact dolphin kicks, then angles toward the surface. One arm separates into the first pull as his legs switch to a fast flutter kick. His head breaks the surface and he establishes front crawl; from below, alternating forearms press water backward while small bubbles trail his hands and shoulders. The floor guide passes beneath him as the camera matches his forward speed.",
      "camera_motion": [
        {
          "time_range_sec": "2.2 to 5.8",
          "camera_motion": [
            "track_right"
          ]
        }
      ],
      "contained_actions": [
        {
          "time_range_sec": "2.2 to 3.5",
          "action": "<|Subject1|> performs two dolphin kicks during the glide and directs his streamlined body upward."
        },
        {
          "time_range_sec": "3.5 to 5.8",
          "action": "<|Subject1|> breaks out into front crawl, alternating underwater pulls and changing to a continuous flutter kick."
        }
      ]
    },
    {
      "event_index": 2,
      "time_range_sec": "5.8 to 10",
      "event_description": "The camera rises through the waterline, revealing his recovering arms above the surface while keeping his advancing body at useful scale. He powers onward with alternating reaches and pulls, each hand entry producing a short splash in a regular rhythm. He rolls his head toward the camera for one breath, returns his face to the water and drives into the next strokes without slowing.",
      "camera_motion": [
        {
          "time_range_sec": "5.8 to 6.6",
          "camera_motion": [
            "track_right",
            "dolly_up"
          ]
        },
        {
          "time_range_sec": "6.6 to 10",
          "camera_motion": [
            "track_right"
          ]
        }
      ],
      "contained_actions": [
        {
          "time_range_sec": "5.8 to 10",
          "action": "<|Subject1|> races forward in front crawl with alternating arm recoveries and rhythmic hand entries, takes one side breath and returns his face to the water while maintaining his kick."
        }
      ]
    }
  ],
  "quality": {
    "video_quality": "Resolved athletic form and transparent water, with brief motion blur on fast hands and splashes.",
    "text_logos_overlays": null,
    "artifacts": null
  }
}
""")
PROMPT = json.dumps(CAPTION, ensure_ascii=False, separators=(",", ":"))
NUM_FRAMES = 241  # 10 s at 24 fps, the caption's length.

OUTPUT_PATH = "video_samples_pan2"
MODEL_PATH = os.environ.get("PAN2_MODEL_PATH", "IFM/PAN2")

# Set to a first frame (e.g. "assets/images/girl.png") for image-to-video; the same pipeline runs both modes. The
# caption's `init_frame_visual_description` should then describe that frame.
IMAGE_PATH = None


def main():
    # The 29B DiT runs in bf16; with more GPUs, num_gpus enables Ulysses sequence parallelism (the 32 attention
    # heads must be divisible by it).
    generator = VideoGenerator.from_pretrained(
        MODEL_PATH,
        num_gpus=int(os.environ.get("PAN2_NUM_GPUS", "1")),
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    generator.generate_video(
        PROMPT,
        image_path=IMAGE_PATH,
        num_frames=NUM_FRAMES,
        output_path=OUTPUT_PATH,
        save_video=True,
    )


if __name__ == "__main__":
    main()
