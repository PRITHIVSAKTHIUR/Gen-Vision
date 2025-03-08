import os
import random
import uuid
import json
import time
import asyncio
import re
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import edge_tts

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from transformers.image_utils import load_image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

DESCRIPTION = """
# Gen Vision 🎃
"""

css = '''
h1 {
  text-align: center;
  display: block;
}

#duplicate-button {
  margin: auto;
  color: #fff;
  background: #1565c0;
  border-radius: 100vh;
}
'''

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------
# Progress Bar Helper
# -----------------------
def progress_bar_html(label: str) -> str:
    """
    Returns an HTML snippet for a thin progress bar with a label.
    The progress bar is styled as a dark red animated bar.
    """
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #DDA0DD; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #FF00FF; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

# -----------------------
# Text Generation Setup
# -----------------------
model_id = "prithivMLmods/FastThink-0.5B-Tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

TTS_VOICES = [
    "en-US-JennyNeural",  # @tts1
    "en-US-GuyNeural",    # @tts2
]

# -----------------------
# Multimodal OCR Setup
# -----------------------
MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

async def text_to_speech(text: str, voice: str, output_file="output.mp3"):
    """Convert text to speech using Edge TTS and save as MP3"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def clean_chat_history(chat_history):
    """
    Filter out any chat entries whose "content" is not a string.
    """
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

# -----------------------
# Stable Diffusion Image Generation Setup
# -----------------------

MAX_SEED = np.iinfo(np.int32).max
USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = False

if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # LoRA options with one example for each.
    LORA_OPTIONS = {
        "Realism": ("prithivMLmods/Canopus-Realism-LoRA", "Canopus-Realism-LoRA.safetensors", "rlms"),
        "Pixar": ("prithivMLmods/Canopus-Pixar-Art", "Canopus-Pixar-Art.safetensors", "pixar"),
        "Photoshoot": ("prithivMLmods/Canopus-Photo-Shoot-Mini-LoRA", "Canopus-Photo-Shoot-Mini-LoRA.safetensors", "photo"),
        "Clothing": ("prithivMLmods/Canopus-Clothing-Adp-LoRA", "Canopus-Dress-Clothing-LoRA.safetensors", "clth"),
        "Interior": ("prithivMLmods/Canopus-Interior-Architecture-0.1", "Canopus-Interior-Architecture-0.1δ.safetensors", "arch"),
        "Fashion": ("prithivMLmods/Canopus-Fashion-Product-Dilation", "Canopus-Fashion-Product-Dilation.safetensors", "fashion"),
        "Minimalistic": ("prithivMLmods/Pegasi-Minimalist-Image-Style", "Pegasi-Minimalist-Image-Style.safetensors", "minimalist"),
        "Modern": ("prithivMLmods/Canopus-Modern-Clothing-Design", "Canopus-Modern-Clothing-Design.safetensors", "mdrnclth"),
        "Animaliea": ("prithivMLmods/Canopus-Animaliea-Artism", "Canopus-Animaliea-Artism.safetensors", "Animaliea"),
        "Wallpaper": ("prithivMLmods/Canopus-Liquid-Wallpaper-Art", "Canopus-Liquid-Wallpaper-Minimalize-LoRA.safetensors", "liquid"),
        "Cars": ("prithivMLmods/Canes-Cars-Model-LoRA", "Canes-Cars-Model-LoRA.safetensors", "car"),
        "PencilArt": ("prithivMLmods/Canopus-Pencil-Art-LoRA", "Canopus-Pencil-Art-LoRA.safetensors", "Pencil Art"),
        "ArtMinimalistic": ("prithivMLmods/Canopus-Art-Medium-LoRA", "Canopus-Art-Medium-LoRA.safetensors", "mdm"),
    }

    # Load all LoRA weights
    for model_name, weight_name, adapter_name in LORA_OPTIONS.values():
        pipe.load_lora_weights(model_name, weight_name=weight_name, adapter_name=adapter_name)
    pipe.to("cuda")
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to(device)

def save_image(img: Image.Image) -> str:
    """Save a PIL image with a unique filename and return the path."""
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=180, enable_queue=True)
def generate_image(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.0,
    randomize_seed: bool = True,
    lora_model: str = "Realism",
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    effective_negative_prompt = negative_prompt  # Use provided negative prompt if any
    model_name, weight_name, adapter_name = LORA_OPTIONS[lora_model]
    pipe.set_adapters(adapter_name)
    outputs = pipe(
         prompt=prompt,
         negative_prompt=effective_negative_prompt,
         width=width,
         height=height,
         guidance_scale=guidance_scale,
         num_inference_steps=28,
         num_images_per_prompt=1,
         cross_attention_kwargs={"scale": 0.65},
         output_type="pil",
    )
    images = outputs.images
    image_paths = [save_image(img) for img in images]
    return image_paths, seed

# -----------------------
# Main Chat/Generation Function
# -----------------------
@spaces.GPU
def generate(
    input_dict: dict,
    chat_history: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generates chatbot responses with support for multimodal input, TTS, and image generation.
    Special commands:
      - "@tts1" or "@tts2": triggers text-to-speech.
      - "@<lora_command>": triggers image generation using the new LoRA pipeline.
         Available commands (case-insensitive): @realism, @pixar, @photoshoot, @clothing, @interior, @fashion, 
         @minimalistic, @modern, @animaliea, @wallpaper, @cars, @pencilart, @artminimalistic.
    """
    text = input_dict["text"]
    files = input_dict.get("files", [])
    
    # Check for image generation command based on LoRA tags.
    lora_mapping = { key.lower(): key for key in LORA_OPTIONS }
    for key_lower, key in lora_mapping.items():
        command_tag = "@" + key_lower
        if text.strip().lower().startswith(command_tag):
            prompt_text = text.strip()[len(command_tag):].strip()
            yield progress_bar_html(f"Processing Image Generation ({key} style)")
            image_paths, used_seed = generate_image(
                prompt=prompt_text,
                negative_prompt="",
                seed=1,
                width=1024,
                height=1024,
                guidance_scale=3,
                randomize_seed=True,
                lora_model=key,
            )
            yield progress_bar_html("Finalizing Image Generation")
            yield gr.Image(image_paths[0])
            return 
    
    # Check for TTS command (@tts1 or @tts2)
    tts_prefix = "@tts"
    is_tts = any(text.strip().lower().startswith(f"{tts_prefix}{i}") for i in range(1, 3))
    voice_index = next((i for i in range(1, 3) if text.strip().lower().startswith(f"{tts_prefix}{i}")), None)
    
    if is_tts and voice_index:
        voice = TTS_VOICES[voice_index - 1]
        text = text.replace(f"{tts_prefix}{voice_index}", "").strip()
        conversation = [{"role": "user", "content": text}]
    else:
        voice = None
        text = text.replace(tts_prefix, "").strip()
        conversation = clean_chat_history(chat_history)
        conversation.append({"role": "user", "content": text})
    
    if files:
        if len(files) > 1:
            images = [load_image(image) for image in files]
        elif len(files) == 1:
            images = [load_image(files[0])]
        else:
            images = []
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text},
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        yield progress_bar_html("Processing with Qwen2VL Ocr")
        for new_text in streamer:
            buffer += new_text
            buffer = buffer.replace("<|im_end|>", "")
            time.sleep(0.01)
            yield buffer
    else:
        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        }
        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        outputs = []
        for new_text in streamer:
            outputs.append(new_text)
            yield "".join(outputs)

        final_response = "".join(outputs)
        yield final_response

        if is_tts and voice:
            output_file = asyncio.run(text_to_speech(final_response, voice))
            yield gr.Audio(output_file, autoplay=True)

# -----------------------
# Gradio Chat Interface
# -----------------------
demo = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2),
    ],
    examples=[
        ['@realism Chocolate dripping from a donut against a yellow background, in the style of brocore, hyper-realistic'],
        ["@pixar A young man with light brown wavy hair and light brown eyes sitting in an armchair and looking directly at the camera, pixar style, disney pixar, office background, ultra detailed, 1 man"],
        ["@realism A futuristic cityscape with neon lights"],
        ["@photoshoot A portrait of a person with dramatic lighting"],
        [{"text": "summarize the letter", "files": ["examples/1.png"]}],
        ["Python Program for Array Rotation"],
        ["@tts1 Who is Nikola Tesla, and why did he die?"],
        ["@clothing Fashionable streetwear in an urban environment"],
        ["@interior A modern living room interior with minimalist design"],
        ["@fashion A runway model in haute couture"],
        ["@minimalistic A simple and elegant design of a serene landscape"],
        ["@modern A contemporary art piece with abstract geometric shapes"],
        ["@animaliea A cute animal portrait with vibrant colors"],
        ["@wallpaper A scenic mountain range perfect for a desktop wallpaper"],
        ["@cars A sleek sports car cruising on a city street"],
        ["@pencilart A detailed pencil sketch of a historic building"],
        ["@artminimalistic An artistic minimalist composition with subtle tones"],
        ["@tts2 What causes rainbows to form?"],
    ],
    cache_examples=False,
    type="messages",
    description=DESCRIPTION,
    css=css,
    fill_height=True,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple", placeholder="default [text, vision] , scroll down examples to explore more art styles"),
    stop_btn="Stop Generation",
    multimodal=True,
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True)
