from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)
    return scores

class QwenVLScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=None,
        ).to(self.device)
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        self.task = '''
Your role is to evaluate the aesthetic quality score of given images.
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible
subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward
composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but
lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with
a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful
composition with emotional impact.

Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the evaluation process in detail here]
</Thought>
<Score>X</Score>
'''
        
    @torch.no_grad()
    def __call__(self, prompt, images):
        images_base64 = [pil_image_to_base64(image) for image in images]
        messages=[]
        for base64_qwen in images_base64:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": self.task},
                    ],
                },
            ])

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rewards = extract_scores(output_texts)
        return rewards

# Usage example
def main():
    scorer = QwenVLScorer(
        device="cuda",
        dtype=torch.bfloat16
    )
    images=[
    "nasa.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]

    print(scorer(None, pil_images))

if __name__ == "__main__":
    main()