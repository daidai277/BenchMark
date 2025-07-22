import os
import re

import numpy as np
import pandas as pd
import torch

from transformers import (
    Gemma3ForConditionalGeneration,
)
from modelscope import AutoProcessor
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用GPU 0和1


data_name = 'JAMA 2023'
model_name = 'gemma-3-12b-it'

data_pd = f'data summary/{data_name}_summary.xlsx'
test_json_path = f'dataset/json/{data_name}_dataset.json'
result_path = f'results/{model_name} {data_name} response.xlsx'
model_path = f"/data/DuWending/LLM/{model_name}"


model = Gemma3ForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(model_path)


# ====================测试模式===================
# 读取测试数据
with open(test_json_path, 'r') as f:
    test_dataset = json.load(f)

import tqdm

if os.path.exists(result_path):
    result_pd = pd.read_excel(result_path)
else:
    result_pd = pd.read_excel(data_pd)
    result_pd["respond"] = np.nan

test_bar = tqdm.tqdm(test_dataset)
for index, item in enumerate(test_bar):
    if not pd.notna(result_pd.at[index, 'respond']):
        input_image_prompt = item["conversations"][0]["value"]
        # 去掉前后的<|vision_start|>和<|vision_end|>
        prompt = input_image_prompt.split("<|vision_start|>")[0]
        prompt += (".Based on the provided image and clinical description "
                   "please make an ophthalmic diagnosis for this patient. And output the diagnostic conclusions only.")

        # 使用正则表达式提取路径
        pattern = r'<\|vision_start\|>(.*?)<\|vision_end\|>'
        paths = re.findall(pattern, input_image_prompt)
        content = []
        if paths:
            for i in paths:
                i = i.replace('\\', '/')
                content.append({
                    "type": "image",
                    "image": f"{i}",
                    "resized_height": 224,
                    "resized_width": 224,
                })

            content.append({"type": "text", "text": prompt})
        else:
            content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            generation = generation[0][input_len:]

        response = processor.decode(generation, skip_special_tokens=True)


        # 修改第 i 行的 respond 列
        result_pd.at[index, 'respond'] = response

        result_pd.to_excel(result_path, index=False)

print('Test Complete!')