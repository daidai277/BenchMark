import os
import re

import pandas as pd
import torch

from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用GPU 0和1


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


data_name = 'JAMA 2023'
model_name = 'Qwen2.5-VL-7B-Instruct'

data_pd = f'data summary/{data_name}_summary.xlsx'
test_json_path = f'dataset/json/{data_name}_dataset.json'
result_path = f'results/{model_name} {data_name} response.xlsx'
model_path = f"/data/DuWending/LLM/{model_name}"

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                          trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)

checkpoint_path = f"./output/{model_name}"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float16,
    trust_remote_code=True,
)

# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 读取测试数据
with open(test_json_path, 'r') as f:
    test_dataset = json.load(f)

test_image_list = []
patient_result_dict = {}
import tqdm

df = pd.read_excel(data_pd)
test_bar = tqdm.tqdm(test_dataset)
for index, item in enumerate(test_bar):
    input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    prompt = input_image_prompt.split("<|vision_start|>")[0]
    prompt += ".Based on the provided image and clinical description please make an ophthalmic diagnosis for this patient. And output the diagnostic conclusions only."

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
            "role": "user",
            "content": content,
        }
    ]

    response = predict(messages, model)

    # 修改第 i 行的 respond 列
    df.at[index, 'respond'] = response

# 3. 保存到新文件（避免覆盖原始数据）
    df.to_excel(result_path, index=False)
