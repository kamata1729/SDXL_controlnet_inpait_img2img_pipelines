# ControlNet Pipelines for SDXL inpaint/img2img models
This repository provides the implementation of `StableDiffusionXLControlNetInpaintPipeline` and `StableDiffusionXLControlNetImg2ImgPipeline`. These pipelines are not officially implemented in [diffusers](https://github.com/huggingface/diffusers) yet, but enable more accurate image generation/editing processes. 
## StableDiffusionXLControlNetInpaintPipeline
**SDXL + Inpainting + ControlNet pipeline**

![inpaint_teaser](https://github.com/kamata1729/SDXL_controlnet_inpait_img2img_pipelines/assets/26928144/67d4a33b-0dbd-4c1e-9240-9c75b5ebeaea)

`StableDiffusionXLControlNetInpaintPipeline` is quite simple!
Sample codes are below:
```shell
# for depth conditioned controlnet
python test_controlnet_inpaint_sd_xl_depth.py
# for canny image conditioned controlnet
python test_controlnet_inpaint_sd_xl_canny.py
```
Of course, you can also use the ControlNet provided by SDXL, such as normal map, openpose, etc.

In `test_controlnet_inpaint_sd_xl_depth.py`, `StableDiffusionXLControlNetInpaintPipeline` is used as follows. 
All you have to do is to specify `control_image` and `mask_image` as conditions.

```python

...

# construct pipeline
from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_model_cpu_offload()

...

# image generation conditioned with control_image & mask_image
images = pipe(
    prompt, image=init_image, control_image=depth_image, mask_image=mask_image, num_inference_steps=30, controlnet_conditioning_scale=controlnet_conditioning_scale,
).images

images[0].save(f"dogstatue.png")
```


## StableDiffusionXLControlNetImg2ImgPipeline
