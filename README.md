---
title: GEN V
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.40.0
header: mini
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: 'Image Generation : Gen V'
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## GITLFS

    # Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    
    git clone https://huggingface.co/spaces/prithivMLmods/GEN-VISION
    
    # If you want to clone without large files - just their pointers
    
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/prithivMLmods/GEN-VISION

## SSH
    
    # Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    
    git clone git@hf.co:spaces/prithivMLmods/GEN-VISION
    
    # If you want to clone without large files - just their pointers
    
    GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:spaces/prithivMLmods/GEN-VISION

![alt text](assets/genv.png)

-------------------------------------------------------------------------------------------------------------

## Colab ⚡

Installing all the requirements ( requirements.txt )

![alt text](Colab/colab1.png)

Authentication & Huggingface Login ( pass your Access token from HF )

![alt text](Colab/colab2.png)

Attached Models Loaded

![alt text](Colab/colab3.png)

Loading LoRA Models

![alt text](Colab/colab4.png)

Launched in Gradio

![alt text](Colab/colab5.png)

Running Sample

![alt text](Colab/colab6.png)

Results #prompt : Hoodie: Front view, capture a urban style, Superman Hoodie, technical materials, fabric small point label on text Blue theory, the design is minimal, with a raised collar, fabric is a Light yellow, low angle to capture the Hoodies form and detailing, f/5.6 to focus on the hoodies craftsmanship, solid grey background, studio light setting, with batman logo in the chest region of the t-shirt


![alt text](Colab/colab7.png)


.

.

.

## Dependencies

| Package        | Version       |
|----------------|---------------|
| `diffusers`    | latest        |
| `torch`        | latest        |
| `torchvision`  | latest        |
| `pipeline`     | latest        |
| `transformers` | 4.43.3        |
| `accelerate`   | latest        |
| `safetensors`  | latest        |
| `spaces`       | latest        |
| `peft`         | latest        |
| `gradio`       | latest        |

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


![alt text](assets/GenVis.gif)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



