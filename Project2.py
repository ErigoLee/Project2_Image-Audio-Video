import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import imageio.v3 as iio
import os

# ✅ device setting
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ pipeline loading
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True,
).to(device)

# ✅ text → embedding function
def get_prompt_embedding(prompt):
    tokens = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    return pipe.text_encoder(tokens)[0]

# ✅ common setting
num_frames = 12
guidance = 8
steps = 30
height = width = 512
fps = 5

# ✅ style prompts setting
style_prompts = {
    "ghibli": {
        "prompt_1": "a peaceful countryside with soft wind and wildflowers, hand-painted look, warm light, studio ghibli style, whimsical and nostalgic atmosphere",
        "prompt_2": "a magical forest at twilight with glowing fireflies and soft mist, studio ghibli style, highly detailed, dreamy and mysterious feeling, cinematic lighting",
        "filename": "ghibli_interpolation.mp4"
    },
    "simpsons": {
        "prompt_1": "a quiet suburban home in Springfield, simpsons cartoon style, yellow characters, blue sky, simple 2D look",
        "prompt_2": "a donut shop on fire at night, simpsons style, comical chaos, cartoon fire, dramatic sky",
        "filename": "simpsons_interpolation.mp4"
    }
}

# ✅ repeat: two styles video generation
for style, config in style_prompts.items():
    print(f"🎨 Generating {style} style video...")

    emb1 = get_prompt_embedding(config["prompt_1"])
    emb2 = get_prompt_embedding(config["prompt_2"])

    images = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        interp_emb = (1 - t) * emb1 + t * emb2
        image = pipe(
            prompt_embeds=interp_emb,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=height,
            width=width
        ).images[0]
        images.append(np.array(image))

    # 저장
    iio.imwrite(config["filename"], images, fps=fps, codec='libx264')
    print(f"✅ {config['filename']} store successfully!")

print("🎬 All videos have been successfully generated!")
