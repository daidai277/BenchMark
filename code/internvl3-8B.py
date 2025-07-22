from tqdm import tqdm


import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import json
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用GPU 0和1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

data_name = 'JAMA 2023'
model_name = "InternVL3-8B"

data_pd = f'data summary/{data_name}_summary.xlsx'
test_json_path = f'dataset/json/{data_name}_dataset.json'
result_path = f'results/{model_name} {data_name} response.xlsx'
model_path = f"/data/DuWending/LLM/{model_name}"


model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

with open(test_json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

if os.path.exists(result_path):
    result_pd = pd.read_excel(result_path)
else:
    result_pd = pd.read_excel(data_pd)
    result_pd["respond"] = np.nan

instruction = '. Based on the provided image and clinical description please make an ophthalmic diagnosis for this patient. And output the diagnostic conclusions only.'#. And translate it into Chinese.'
instruction2 = '. Based on the provided images and clinical description please make an ophthalmic diagnosis for this patient. And output the diagnostic conclusions only.'#. And translate it into Chinese.'
instruction3 = '. Based on the clinical description please make an ophthalmic diagnosis for this patient. And output the diagnostic conclusions only.'

test_bar = tqdm(data)
for i, d in enumerate(test_bar):
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # if i > 67:
    conversation = d['conversations'][0]['value']
    prompt = conversation.split('<|vision_start|>')[0]
    if prompt[-1] == ' ' or prompt[-1] == '.':
        prompt = prompt[:-1]
    if prompt[-1] == ' ' or prompt[-1] == '.':
        prompt = prompt[:-1]

    path = conversation.split('<|vision_start|>')[1:]
    paths = []
    for p in path:
        p = p.split('<|vision_end|>')[0]
        # p = os.path.join('yanke', p)
        paths.append(p)

    if len(paths) == 0:
        id = d['id']
        prompt = prompt + instruction3
        question = prompt
        response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    elif len(paths) == 1:
        prompt = prompt + instruction
        id = paths[0].split('\\')[-1].split(".")[0]
        pixel_values = load_image(paths[0], max_num=12).to(
            torch.bfloat16).cuda()
        # generation_config = dict(max_new_tokens=1024, do_sample=True)
        question = '<image>\n' + prompt
        response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')
        # print(paths[0], response)
    else:
        id = paths[0].split('\\')[-1].split('（')[0]
        prompt = prompt + instruction2
        pixel_values = []
        num_patches_list = []
        tmp_question = ''
        for index, pp in enumerate(paths):
            'Image-1: <image>\nImage-2: <image>\nDescribe'
            tmp_question = tmp_question + 'Image-' + str(index+1) + ': <image>\n'
            tmp_img = load_image(pp, max_num=12).to(torch.bfloat16).cuda()
            pixel_values.append(tmp_img)
            num_patches_list.append(tmp_img.size(0))

        pixel_values = torch.cat(pixel_values, dim=0)
        question = tmp_question + prompt

        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=True)

        # 修改第 i 行的 respond 列
    result_pd.at[i, 'respond'] = response

    result_pd.to_excel(result_path, index=False)
