import json
import random
import os

# 将数字映射到英文单词
NUM_TO_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
}

import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
import importlib

model_id = "black-forest-labs/FLUX.1-dev"
device = "cuda"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

def process_jsonl(input_file, output_file, image_directory):
    """
    处理输入的jsonl文件，并生成新的jsonl文件和图片。

    Args:
        input_file (str): 输入的jsonl文件名。
        output_file (str): 输出的jsonl文件名。
        image_directory (str): 保存图片的目录。
    """
    # 确保保存图片的目录存在
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            try:
                data = json.loads(line.strip())

                # 1. 从 "include" 中获取当前的 count
                original_count = data["include"][0]["count"]
                class_name = data["include"][0]["class"]

                image = pipe(
                    data["t2i_prompt"],
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                ).images[0]
                image_path = os.path.join(image_directory, f"image_{i}.jpg")
                image.save(image_path)

                # 4. 创建新的JSON对象
                change_num = set([1, 2, 3, 4]) - set([original_count])
                for num in change_num:
                    new_data = {
                        "tag": data["tag"],
                        "include": [{"class": class_name, "count": num}],
                        "exclude": [{"class": class_name, "count": num + 1}],
                        "t2i_prompt": data["t2i_prompt"],
                        "prompt": f"Change the number of {class_name} in the image to {NUM_TO_WORD[num]}.",
                        "image": image_path
                    }

                    # 5. 将新的JSON对象写入输出文件
                    outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"处理第 {i+1} 行时出错: {e}")
                continue

if __name__ == '__main__':
    # 设定输入/输出文件名和图片目录
    input_filename = "metadata.jsonl"
    output_filename = "output.jsonl"
    image_save_directory = "generated_images"

    # 执行处理函数
    process_jsonl(input_filename, output_filename, image_save_directory)

    print(f"处理完成！结果已保存到 '{output_filename}'，图片路径保存在 '{image_save_directory}' 目录。")