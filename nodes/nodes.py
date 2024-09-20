import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import json
from nodes import common_ksampler, VAEDecode
import comfy
from PIL import Image, ImageDraw, ImageFont

import sys
import os
from uuid import uuid4
import re
import unicodedata

def clean_text(text):
    normalized_text = unicodedata.normalize('NFKD', text)

    cleaned_text = ''.join(c for c in normalized_text if ord(c) < 128)
    
    return cleaned_text


def create_comic_base(total_width, total_height):
    return Image.new('RGB', (total_width, total_height), color='black')

def get_text_height(text, font, max_width):
    draw = ImageDraw.Draw(Image.new('RGB', (max_width, 1)))
    lines = []
    words = text.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        line_width = draw.textbbox((0, 0), test_line, font=font)[2]  # Use textbbox to get width
        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)
    
    # Calculate total height required
    line_height = draw.textbbox((0, 0), "A", font=font)[3]  # Get height of a single line
    return len(lines) * (line_height+5), lines

def add_to_comic(comic_page, new_image, text, index, panel_width=1024, panel_height=1024, extra_displacement=0, columns=2):
    """
    This is a piece of shit no good function
    """
    draw = ImageDraw.Draw(comic_page)
    
    font_size = 30 
    font = ImageFont.truetype("arial.ttf", font_size)

    col = index % columns
    row = index // columns
    
    x = col * panel_width
    y = row * panel_height + extra_displacement

    text_height, text_lines = get_text_height(text, font, panel_width-35)
    
    image_size = (int(panel_width * 0.97), int(panel_height * 0.97))
    new_image_resized = new_image.resize(image_size, Image.LANCZOS)
    
    new_image_resized_x = x + (panel_width - image_size[0]) // 2
    new_image_resized_y = y + (panel_height - image_size[1]) // 2

    comic_page.paste(new_image_resized, (new_image_resized_x, new_image_resized_y))
    
    text_y = y + panel_height + 10 
    text_x = x + 15

    text_box_width = panel_width - 33
    text_box_height = text_height + 15 
    
    rectangle_y = text_y - 5 
    draw.rectangle([(text_x, rectangle_y), (text_x + text_box_width, rectangle_y + text_box_height)], fill="white")
    
    for line in text_lines:
        draw.text((text_x+10, text_y), line, font=font, fill="black")
        text_y += draw.textbbox((0, 0), line, font=font)[3] 

    return text_height + 35





def tensor_to_np(img_tensor, batch_index=0):
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = np.clip(i, 0, 255).astype(np.uint8).squeeze()
    return img

def llm_api_request(prompt, system, max_tokens, temp, top_p, min_p, url='http://localhost:5000/v1/chat/completions', api_key='none', model='none', instruction_template='ChatML'):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages=[]
    if system.strip() != '':
        messages.append({
            "role": "system",
            "content": system
        })
    if prompt.strip() != '':
        messages.append({
            "role": "user",
            "content": prompt
        })

    data = {
        "messages": messages,
        "mode": "instruct",
        "model": model,
        "instruction_template": instruction_template,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "min_p": min_p
    }

    response = requests.post(url, headers=headers, json=data, verify=False)

    text = response.json()['choices'][0]['message']['content'].strip()
    
    return text

class LLM_API_Request:     

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {
                    "system_prompt": ("STRING",{"multiline": True}),
                    "prompt": ("STRING",{"multiline": True}),
                    "url": ("STRING",{"default": "http://127.0.0.1:5000/v1/chat/completions"}),
                    "api_key": ("STRING",{"default": "none"}),
                    "model": ("STRING",{"default": "none"}),
                    "instruction_template": ("STRING",{"default": "ChatML"}),
                    "max_tokens": ("INT",{"default": 500, "min": 0, "max": 999999}),
                    "temperature": ("FLOAT",{"default": 1,  'step':0.01, "min": -5, "max": 5}),
                    "top_p": ("FLOAT",{"default": 0.7,  'step':0.01, "min": 0, "max": 1}),
                    "min_p": ("FLOAT",{"default": 0.1, 'step':0.01,  "min": 0, "max": 1}),
                    "repetition_penalty": ("FLOAT",{"default": 1, 'step':0.01,  "min": 0, "max": 5}),
                    "seed": ("INT", {"default": 0, 'step':0.01, "min": 0, "max": 0xffffffffffffffff}),
        }}
        
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "llm_api_request"
    CATEGORY = "LLM API"
    
    def llm_api_request(self, system_prompt, prompt, url, api_key, model, instruction_template, max_tokens, temperature, top_p, min_p, repetition_penalty, seed):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        messages=[]
        if system_prompt.strip() != '':
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        if prompt.strip() != '':
            messages.append({
                "role": "user",
                "content": prompt
            })

        data = {
            "messages": messages,
            "mode": "instruct",
            "model": model,
            "instruction_template": instruction_template,
            "max_tokens": max_tokens,
            "min_p": min_p,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty:": repetition_penalty
        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        text = response.json()['choices'][0]['message']['content'].strip()
        
        return (text,)




class GenerateComic:     

    def __init__(self):
        self.vae_decoder=VAEDecode()

    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {
                    "story": ("STRING", {"multiline": True}),
                    "add_to_positive": ("STRING", {"multiline": True}),
                    "add_to_negative": ("STRING", {"multiline": True}),
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "latent_image": ("LATENT",),
                    "num_columns": ("INT", {"default": 2, "min": 0, "max": 10}),
        }}
        

    RETURN_TYPES = ()
    FUNCTION = "generate_comic"
    CATEGORY = "Comic generation"
    OUTPUT_NODE = True

    def clip_encode(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]
    
    def get_image(self, clip, prompt,  add_to_positive, add_to_negative, model, vae, seed, steps, cfg, sampler_name, scheduler, latent_image):
        positive = self.clip_encode(clip, add_to_positive+prompt)
        negative = self.clip_encode(clip, add_to_negative)
        (sampled,)=common_ksampler(model, seed, steps, cfg, sampler_name,
                                scheduler, positive, negative, latent_image)
        (decoded,) = self.vae_decoder.decode(vae, sampled)
        return decoded
    
    def generate_visual(self, panel_prompts, add_to_positive, add_to_negative, panel_texts, clip, model, vae, seed, steps, cfg, sampler_name, scheduler, latent_image, columns):
        num_panels = len(panel_prompts)
        panel_width, panel_height = 1024, 1024

        
        comic_height = int(panel_height*((num_panels+1)/columns))*2
        comic_width = columns * panel_width
        
        comic = create_comic_base(comic_width, comic_height)
        prev_row=0
        column_text_heights = {str(i): [0] for i in range(-1, 999)}
        for idx, (prompt, text) in enumerate(zip(panel_prompts, panel_texts)):
            row = idx // columns
            prev_row=row

            new_image = self.get_image(clip, prompt,  add_to_positive, add_to_negative, model, vae, seed, steps, cfg, sampler_name, scheduler, latent_image)
            seed+=1
            new_image = tensor_to_np(new_image)

            new_image_pil = Image.fromarray(new_image.astype('uint8'))
            text_height = add_to_comic(comic, new_image_pil, text, idx, columns=columns, extra_displacement=max(column_text_heights[str(row-1)]))

    

            column_text_heights[str(row)].append(text_height+max(column_text_heights[str(row-1)]))

        comic_pixels = np.array(comic)

        for y in range(comic_height - 1, -1, -1):
            crop_height = y
            if (comic_pixels[y, :, :] != 0).any(): 
                break

        comic = comic.crop((0, 0, comic_width, crop_height))


        return comic

    def generate_comic(self, story, add_to_positive, add_to_negative, model, clip, vae, seed, steps, cfg, sampler_name, scheduler, latent_image, num_columns):
        story = clean_text(story)
        panel_prompts = re.findall(r'\{([^{}]*)\}', story)
        print("\nPanel prompts: ", panel_prompts)
        story = story.replace("}", '').replace("{", '')

        panel_texts = []
        for i, prompt in enumerate(panel_prompts):
            start_idx = story.find(panel_prompts[i-1]) + len(panel_prompts[i-1]) if i != 0 else 0
            end_idx = story.find(panel_prompts[i]) -1
            story_piece = story[start_idx : end_idx].strip().replace('\n', '')
            if story_piece.startswith('.'):
                story_piece=story_piece[1:]
            panel_texts.append(story_piece)
        
        print("\nPanel texts: ", panel_texts)

        comic = self.generate_visual(panel_prompts, add_to_positive, add_to_negative, panel_texts, clip, model, vae, seed, steps, cfg, sampler_name, scheduler, latent_image, columns=num_columns)
        
        main_script_path = os.path.abspath(sys.argv[0])
        main_dir = os.path.dirname(main_script_path)
        output_dir = os.path.join(main_dir, "output")
        path = os.path.join(output_dir, f"{uuid4()}.jpeg")

        print("PATH ",path)

        comic.save(path)
        return ()