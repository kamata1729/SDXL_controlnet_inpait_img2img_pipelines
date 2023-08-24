from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline

prompt = "pixelart on a bench"

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

canny_image = np.array(init_image)
canny_image = cv2.Canny(canny_image, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image = Image.fromarray(canny_image)

images = pipe(
    prompt, image=init_image, control_image=canny_image, mask_image=mask_image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"pixelart_canny.png")