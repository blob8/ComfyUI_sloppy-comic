from nodes import common_ksampler, VAEDecode
import comfy
import subprocess
import sys
import threading
subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import re
import unicodedata
import io
import asyncio
from playwright.async_api import async_playwright
import random
import base64
import cv2
import torch

def find_least_edge_quadrant(image) :
    img = image.convert("L")
    img = img.resize((512, 512))
    img = np.array(img)

    edge_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(edge_x, edge_y)
    h, w = edges.shape
    half_h, half_w = h // 2, w // 2
    quadrants = {
    'tl': edges[:half_h, :half_w],
    'tr': edges[:half_h, half_w:],
    'bl': edges[half_h:, :half_w],
    'br': edges[half_h:, half_w:]
    }

    avg_edges = {q: np.mean(val) for q, val in quadrants.items()}

    return min(avg_edges, key=avg_edges.get)


class RowImage:
    def __init__(self, image, w, h, x, y , is_first, is_last, diag_scale, caption_corner):
        self.image = image
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.is_first = is_first
        self.is_last = is_last
        self.diag_scale = diag_scale
        self.caption_corner = caption_corner
        self.free_space_below = False #if true caption gets put below
    
    @property
    def aspect(self):
        return self.h / self.w
    
    def displace_rescale(self, displacement, factor):
        self.x += displacement
        new_w = factor*self.w
        width_delta = new_w - self.w
        self.w = new_w
        self.h*=factor
        return width_delta


class Row:
    def __init__(self, y, gutter, diag_scale):
        self.images = []
        self.gutter = gutter
        self.max_height = 0
        self.current_x = gutter
        self.y = y
        self.diag=diag_scale*random.randint(-1,1)

    def add_image(self, image, target_w, caption_corner):
        aspect = image.height / image.width
        w = int((target_w * target_w / aspect) ** 0.5)
        h = int(w * aspect)

        if self.images:
            self.images[-1].is_last = False
        row_img = RowImage(image, w, h, self.current_x, self.y, is_first=len(self.images)==0, is_last=True, diag_scale=self.diag, caption_corner=caption_corner)
        self.images.append(row_img)

        self.current_x += w + self.gutter
        return row_img

    def finalize(self, canvas_width):

        total_image_width = sum([img.w for img in self.images])
        needed_width = (canvas_width-self.gutter*(1+len(self.images)))

        if len(self.images) > 1: #if regular row
            rescale = needed_width/total_image_width
            prev=0
        else: #if single image
            if self.images[0].aspect > 1: #if tall or square
                rescale = 1
                prev = int(needed_width/2 - self.images[0].w/2)
            else:
                rescale = needed_width/total_image_width
                prev=0

        for image in self.images:
            prev += image.displace_rescale(prev,rescale)
        
        for image in self.images:
            if self.height > image.h:
                image.free_space_below = self.height - image.h

    @property
    def height(self):
        return max([img.h for img in self.images])


def pil_image_to_data_uri(img, fmt, quality):
    bio = io.BytesIO()
    if fmt.upper() == "JPEG" and img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        background.save(bio, format=fmt, quality=quality)
    else:
        img.save(bio, format=fmt, quality=quality)
    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def comic_collage_from_pil(images,texts,width=1500,style_opts=None):
    print("Composing comic...")

    if style_opts is None:
        style_opts = {}
    
    bg_color = style_opts.get("bg_color", "#fff7ee")
    diag_scale = style_opts.get("diag", 0.1)
    base_panel_w = int(width / style_opts.get("squares_per_row"))
    default_caption_corner = style_opts.get("caption_corner", 'auto')
    gutter = int(base_panel_w/40)
    rows = []
    current_row = Row(y=gutter *3, gutter=gutter,diag_scale=diag_scale)
    height = 0
    for idx, img in enumerate(images):
        caption_corner = find_least_edge_quadrant(img) if default_caption_corner == 'auto' else default_caption_corner
        aspect = img.height / img.width
        w = int((base_panel_w * base_panel_w / aspect) ** 0.5)
        h = int(w * aspect)

        if current_row.current_x + w + gutter > width and current_row.images:
            current_row.finalize(width)
            rows.append(current_row)

            new_y = current_row.y + current_row.height + gutter *2
            height = new_y
            current_row = Row(y=new_y, gutter=gutter, diag_scale=diag_scale)

        current_row.add_image(img, base_panel_w, caption_corner)

    if current_row.images:
        current_row.finalize(width)
        rows.append(current_row)
        height += current_row.height *1.07

    positioned_images = [img for row in rows for img in row.images]

    data_uris = []
    for img in images:
        img_copy = img.copy()
        max_dim = base_panel_w * 2
        img_copy.thumbnail((max_dim, max_dim), Image.LANCZOS)
        data_uris.append(pil_image_to_data_uri(img_copy, fmt="PNG", quality=100))

    panels_html = []

    for idx, (row_img, uri) in enumerate(zip(positioned_images, data_uris)):
        diag_scale = row_img.diag_scale
        w, h, x, y = row_img.w, row_img.h, row_img.x, row_img.y

        points = f" 0 0, \
                    {w} 0, \
                    {w-h*(diag_scale if diag_scale and not row_img.is_last else 0)} {h}, \
                    {-h*(diag_scale if diag_scale and not row_img.is_first else 0)} {h}"

        corner = row_img.caption_corner
        caption_disp = int(base_panel_w/100)

        if row_img.free_space_below:
            caption_style = f"left:{caption_disp}px; bottom:-{int(row_img.free_space_below/2)}px;"
        elif corner == "tl":
            caption_style = f"left:{caption_disp}px; top:{caption_disp}px;"
        elif corner == "tr":
            caption_style = f"right:{caption_disp}px; top:{caption_disp}px;"
        elif corner == "br":
            caption_style = f"right:{caption_disp}px; bottom:{caption_disp}px;"
        else:
            caption_style = f"left:{caption_disp}px; bottom:{caption_disp}px;"

        caption_html = f'<div class="caption" style="{caption_style}">{texts[idx]}</div>'
        adjusted_width = w+abs(h*diag_scale)
        adjusted_height = adjusted_width*(h/w)
        y_disp = abs(adjusted_height-h) * (-1 if 'b' in corner else 0)
        panels_html.append(f"""
        <div class="panel" style="left:{x}px; top:{y}px; width:{w}px; height:{h}px;">
            <svg xmlns="http://www.w3.org/2000/svg"
                 width="{w}" height="{h}"
                 viewBox="0 0 {w} {h}"
                 preserveAspectRatio="none"
                 style="display:block; overflow:visible;">
                <defs>
                    <clipPath id="clip-{idx}" clipPat
                    hUnits="userSpaceOnUse">
                        <polygon points="{points}" />
                    </clipPath>
                </defs>
                <image href="{uri}"
                       x="{min(-h*diag_scale,0)}" y="{y_disp}" width="{adjusted_width}" height="{adjusted_height}"
                       preserveAspectRatio="xMidYMid meet"
                       clip-path="url(#clip-{idx})" />
                <polygon points="{points}"
                         fill="none"
                         stroke="#111"
                         stroke-width="{int(base_panel_w/100)}"
                         stroke-linejoin="round"
                         vector-effect="non-scaling-stroke"
                         style="filter: drop-shadow(0px 10px 30px rgba(0,0,0,0.25));" />
            </svg>
            {caption_html}
        </div>
        """)


    html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    html,body{{margin:0;padding:0;background:{bg_color};}}
    .page {{
        width: {width}px;
        min-height: {height}px;
        position: relative;
        box-sizing: border-box;
        font-family: "Comic Sans MS","Marker Felt","Segoe UI",sans-serif;
        overflow: visible;
        background-image: radial-gradient(circle at 10% 10%, rgba(0,0,0,0.06) 0px, rgba(0,0,0,0.06) 1px, transparent 1px), linear-gradient(45deg, rgba(0,0,0,0.02) 25%, transparent 25%, transparent 75%, rgba(0,0,0,0.02) 75%, rgba(0,0,0,0.02)), linear-gradient(-45deg, rgba(255,255,255,0.04) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.04) 75%, rgba(255,255,255,0.04)); background-size: 40px 40px, 8px 8px, 8px 8px;
    }}
    .panel {{ position:absolute; overflow:visible; }}
    .caption {{ position:absolute; padding:{int(base_panel_w/100)}px {int(base_panel_w/80)}px; font-weight:700; font-size:{int(0.04*base_panel_w)}px; background:#fff; border-radius:{int(base_panel_w/50)}px; border:{int(base_panel_w/100)}px solid #111; transform:translateZ(0); box-shadow:{int(base_panel_w/80)}px {int(base_panel_w/100)}px 0 rgba(0,0,0,0.12); z-index:3; }}
    .panel svg {{ display:block; }}
    </style>
    </head>
    <body>
    <div class="page">
        {''.join(panels_html)}
    </div>
    </body>
    </html>
    """

    def capture_screenshot_sync(html):
        result_container = {}
        def run():
            async def capture_screenshot(html):
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    page = await browser.new_page()
                    await page.set_content(html)
                    screenshot = await page.screenshot(full_page=True)
                    await browser.close()
                    return screenshot

            result_container["screenshot"]= asyncio.run(capture_screenshot(html))

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        return result_container["screenshot"]

    screenshot = capture_screenshot_sync(html)


    img = Image.open(io.BytesIO(screenshot))
    
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0
    return img_tensor



def clean_text(text):
    normalized_text = unicodedata.normalize('NFKD', text)

    cleaned_text = ''.join(c for c in normalized_text if ord(c) < 128)
    
    return cleaned_text


class LLM_API_Request:     

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {
                    "system_prompt": ("STRING",{"multiline": True}),
                    "prompt": ("STRING",{"multiline": True}),
                    "start_with": ("STRING",{"multiline": True}),
                    "url": ("STRING",{"default": "http://127.0.0.1:8080/v1/chat/completions"}),
                    "api_key": ("STRING",{"default": ""}),
                    "model": ("STRING",{"default": ""}),
                    "max_tokens": ("INT",{"default": 1500, "min": 0, "max": 999999}),
                    "temperature": ("FLOAT",{"default": 0.7, 'step':0.01, "min": 0, "max": 5}),
                    "top_p": ("FLOAT",{"default": 0.95, 'step':0.01, "min": 0, "max": 1}),
                    "min_p": ("FLOAT",{"default": 0.01, 'step':0.01,  "min": 0, "max": 1}),
                    "repetition_penalty": ("FLOAT",{"default": 1, 'step':0.01,  "min": 0, "max": 5}),
                    "seed": ("INT", {"default": 0, 'step':0.01, "min": 0, "max": 0xffffffffffffffff}),
        }}
        
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "llm_api_request"
    CATEGORY = "LLM API"
    
    def llm_api_request(self, system_prompt, prompt, start_with, url, api_key, model, max_tokens, temperature, top_p, min_p, repetition_penalty, seed):

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
        messages.append({
            "role": "user",
            "content": prompt
        })
        if start_with.strip() != '':
            messages.append({
                    "role": "assistant",
                    "content": start_with
                })
            
        data = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "min_p": min_p,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty:": repetition_penalty
        }

        response = requests.post(url, headers=headers, json=data, verify=False)

        text = start_with + response.json()['choices'][0]['message']['content'].strip()
        
        return (text,)



class GenerateComic:     

    sdxl_ratio_to_res = {
        "horizontal_wide":[1408,640],
        "horizontal":[1216,832],
        "square":[1024,1024],
        "vertical":[832,1216],
        "vertical_tall":[640,1408]

    }
    def __init__(self):
        self.vae_decoder=VAEDecode()
        self.device = comfy.model_management.intermediate_device()
        
    @classmethod
    def INPUT_TYPES(cls):
               
        return {"required": {
                    "story": ("STRING", {"multiline": True}),
                    "add_to_positive": ("STRING", {"multiline": True}),
                    "add_to_negative": ("STRING", {"multiline": True}),
                    "width": ("INT", {"default": 1800, "min": 0, "max": 99999, 'step':100}),
                    "squares_per_row": ("FLOAT", {"default": 2.6, "min": 1.5, "max": 99999, 'step':0.1}),
                    "bg_color": ("STRING", {"default": "#fff7ee"}),
                    "caption_positioning": ("STRING", {"default": "auto"}),
                    "shear": ("FLOAT", {"default": 0, "min": 0.0, "max": 0.3, 'step':0.05}),
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 15, "min": 1, "max": 10000, "step": 1}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
        }}
        

    RETURN_TYPES = ('IMAGE',)
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
    
    def generate_visual(self, panel_prompts, comic_width, style_opts, add_to_positive, add_to_negative, panel_texts, clip, model, vae, seed, steps, cfg, sampler_name, scheduler):
        
        generated_imgs = []
        for idx, (prompt, text) in enumerate(zip(panel_prompts, panel_texts)):
            
            aspect = prompt.split(',')[0].strip()
            if aspect in self.sdxl_ratio_to_res.keys():
                width, height = self.sdxl_ratio_to_res[aspect]
                prompt = prompt[prompt.find(',')+1:]
            else:
                print(f'No aspect ratio spectified as first tag of the image, must be one of {list(self.sdxl_ratio_to_res.keys())}, defaulting to square.')
                width, height = self.sdxl_ratio_to_res['square']

            latent_image = {"samples":torch.zeros([1, 4, height // 8, width // 8], device=self.device)}
            print(f'Generating panel {idx+1} of {len(panel_prompts)}')
            new_image = self.get_image(clip, prompt,  add_to_positive, add_to_negative, model, vae, seed, steps, cfg, sampler_name, scheduler, latent_image)
            numpy_array = (new_image.squeeze(0) * 255).cpu().numpy()
            generated_imgs.append(Image.fromarray(numpy_array.astype(np.uint8)))
            seed+=1

        
        comic = comic_collage_from_pil(images=generated_imgs, texts=panel_texts, width=comic_width, style_opts=style_opts)
        return comic

    def generate_comic(self, story, add_to_positive, add_to_negative, width, squares_per_row, bg_color, caption_positioning, shear, model, clip, vae, seed, steps, cfg, sampler_name, scheduler):
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

        style_opts = {'bg_color':bg_color, 'diag': shear, 'caption_corner': caption_positioning, 'squares_per_row': squares_per_row}
        comic = self.generate_visual(panel_prompts, width, style_opts, add_to_positive, add_to_negative, panel_texts, clip, model, vae, seed, steps, cfg, sampler_name, scheduler)
    
        return (comic,)
    
