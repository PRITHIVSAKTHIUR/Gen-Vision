import os
import random
import uuid
from typing import Tuple
import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max
USE_TORCH_COMPILE = 0
ENABLE_CPU_OFFLOAD = 0

if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0_Lightning",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    LORA_OPTIONS = {
        "Realism (face/character)👦🏻": ("prithivMLmods/Canopus-Realism-LoRA", "Canopus-Realism-LoRA.safetensors", "rlms"),
        "Pixar (art/toons)🙀": ("prithivMLmods/Canopus-Pixar-Art", "Canopus-Pixar-Art.safetensors", "pixar"),
        "Photoshoot (camera/film)📸": ("prithivMLmods/Canopus-Photo-Shoot-Mini-LoRA", "Canopus-Photo-Shoot-Mini-LoRA.safetensors", "photo"),
        "Clothing (hoodies/pant/shirts)👔": ("prithivMLmods/Canopus-Clothing-Adp-LoRA", "Canopus-Dress-Clothing-LoRA.safetensors", "clth"),
        "Interior Architecture (house/hotel)🏠": ("prithivMLmods/Canopus-Interior-Architecture-0.1", "Canopus-Interior-Architecture-0.1δ.safetensors", "arch"),
        "Fashion Product (wearing/usable)👜": ("prithivMLmods/Canopus-Fashion-Product-Dilation", "Canopus-Fashion-Product-Dilation.safetensors", "fashion"),
        "Minimalistic Image (minimal/detailed)🏞️": ("prithivMLmods/Pegasi-Minimalist-Image-Style", "Pegasi-Minimalist-Image-Style.safetensors", "minimalist"),
        "Modern Clothing (trend/new)👕": ("prithivMLmods/Canopus-Modern-Clothing-Design", "Canopus-Modern-Clothing-Design.safetensors", "mdrnclth"),
        "Animaliea (farm/wild)🫎": ("prithivMLmods/Canopus-Animaliea-Artism", "Canopus-Animaliea-Artism.safetensors", "Animaliea"),
        "Liquid Wallpaper (minimal/illustration)🖼️": ("prithivMLmods/Canopus-Liquid-Wallpaper-Art", "Canopus-Liquid-Wallpaper-Minimalize-LoRA.safetensors", "liquid"),
        "Canes Cars (realistic/futurecars)🚘": ("prithivMLmods/Canes-Cars-Model-LoRA", "Canes-Cars-Model-LoRA.safetensors", "car"),
        "Pencil Art (characteristic/creative)✏️": ("prithivMLmods/Canopus-Pencil-Art-LoRA", "Canopus-Pencil-Art-LoRA.safetensors", "Pencil Art"),
        "Art Minimalistic (paint/semireal)🎨": ("prithivMLmods/Canopus-Art-Medium-LoRA", "Canopus-Art-Medium-LoRA.safetensors", "mdm"),

    }

    for model_name, weight_name, adapter_name in LORA_OPTIONS.values():
        pipe.load_lora_weights(model_name, weight_name=weight_name, adapter_name=adapter_name)
    pipe.to("cuda")

style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}

DEFAULT_STYLE_NAME = "3840 x 2160"
STYLE_NAMES = list(styles.keys())

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    if style_name in styles:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    else:
        p, n = styles[DEFAULT_STYLE_NAME]

    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

@spaces.GPU(duration=180, enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_name: str = DEFAULT_STYLE_NAME,
    lora_model: str = "Realism (face/character)👦🏻",
    progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))

    positive_prompt, effective_negative_prompt = apply_style(style_name, prompt, negative_prompt)
    
    if not use_negative_prompt:
        effective_negative_prompt = ""  # type: ignore

    model_name, weight_name, adapter_name = LORA_OPTIONS[lora_model]
    pipe.set_adapters(adapter_name)

    images = pipe(
        prompt=positive_prompt,
        negative_prompt=effective_negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=20,
        num_images_per_prompt=1,
        cross_attention_kwargs={"scale": 0.65},
        output_type="pil",
    ).images
    image_paths = [save_image(img) for img in images]
    return image_paths, seed

examples = [
    "realism, man in the style of dark beige and brown, uhd image, youthful protagonists, nonrepresentational",
    "pixar, a young man with light brown wavy hair and light brown eyes sitting in an armchair and looking directly at the camera, pixar style, disney pixar, office background",
    "hoodie, front view, capture a urban style, superman hoodie, technical materials, fabric small point label on text blue theory, with a raised collar, fabric is a light yellow, low angle to capture the hoodies form and detailing, f/5.6 to focus on the hoodies craftsmanship, solid grey background, studio light setting, with batman logo.",
]


css = '''
.gradio-container{max-width: 888px !important}
h1{text-align:center}
.submit-btn {
    background-color: #ecde2c  !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #ffec00  !important;
}
'''

def load_predefined_images():
    predefined_images = [
        "assets/1.png",
        "assets/2.png",
        "assets/3.png",
        "assets/4.png",
        "assets/5.png",
        "assets/6.png",
        "assets/7.png",
        "assets/8.png",
        "assets/9.png",
    ]
    return predefined_images

with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt with resp. tag!",
                container=False,
            )
            run_button = gr.Button("Generate  as  (1024 x 1024)🎃", scale=0, elem_classes="submit-btn")

            with gr.Row(visible=True):
                model_choice = gr.Dropdown(
                    label="LoRA Selection",
                    choices=list(LORA_OPTIONS.keys()),
                    value="Realism (face/character)👦🏻")
            
            with gr.Accordion("Advanced options", open=True):
                use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True, visible=True)
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    lines=4,
                    max_lines=6,
                    value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                    placeholder="Enter a negative prompt",
                    visible=True,
                )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                        visible=True
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row(visible=True):
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=8,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=8,
                        value=1024,
                    )
                    
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.1,
                    maximum=20.0,
                    step=0.1,
                    value=3.0,
                )

                style_selection = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                    label="Quality Style",
                )

        with gr.Column(scale=2):
            result = gr.Gallery(label="Result", columns=1, preview=True, show_label=False)

            gr.Examples(
                examples=examples,
                inputs=prompt,
                outputs=[result, seed],
                fn=generate,
                cache_examples=False,
            )
            
            predefined_gallery = gr.Gallery(
                label="Image Gallery",
                columns=3,
                show_label=False,
                value=load_predefined_images()
            )

        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
            api_name=False,
        )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            style_selection,
            model_choice,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch()
