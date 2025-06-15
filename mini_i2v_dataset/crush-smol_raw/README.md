`crush-smol` created out of https://huggingface.co/datasets/bigdata-pw/crush (`crush_smol.parquet`).

Captions were generated with Qwen2VL.

<details>
<summary>generate_captions.py</summary>

```py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from torchvision import io

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
prompt = """
You're a helpful assistant who is a part of a team working on video generation. As an assistant your task is to provide a meaningful caption for a provided video. These captions wll be used to fine-tune other video generation models.
Keeping that in mind, provide a caption for the input video. Please focus on the articulate details of the scenes presented in the video. Here are some guidelines:

* Describe the composition of the scene, how it's progressing with all the components involved.
* Provide some description of the objects involved in the scene.
* DO NOT start the caption with "In this video,".
* Try to include phrase "flattens objects as if they were under a hydraulic press." in a meaningful manner.
"""

MAX_nframes = 16

def fetch_video(ele: dict, nframe_factor=2):
    if isinstance(ele['video'], str):
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        video, _, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], nframe_factor)
        else:
            fps = ele.get("fps", 1.0)
            nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
        if nframes > MAX_nframes:
            nframes = MAX_nframes
            print(f"Setting `nframes` to {nframes=}")
        idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
        return video[idx]

video_path = "crush-smol/videos"
video_paths = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".mp4")]

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ],
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Adjust based on your GPU memory
batch_size = 1

for i in range(0, len(video_paths), batch_size):
    batch_video_paths = video_paths[i: i + batch_size]
    videos = [fetch_video({"type": "video", "video": path, "fps": 1.0}) for path in batch_video_paths]

    # Process videos and prepare inputs
    inputs = processor(text=[text_prompt] * len(videos), videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    # Inference: Generate output for the batch
    output_ids = model.generate(**inputs, max_new_tokens=256)

    # Decode the outputs
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Save each output text to corresponding file
    for video_path, output_text in zip(batch_video_paths, output_texts):
        caption_path = Path(video_path).with_suffix(".txt")
        with open(caption_path, "w") as f:
            f.write(output_text.strip())

        print(output_text.strip())
```
  
</details>