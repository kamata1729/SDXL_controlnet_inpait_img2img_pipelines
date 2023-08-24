# ControlNet Pipelines for SDXL inpaint/img2img models
This repository provides the implementation of `StableDiffusionXLControlNetInpaintPipeline` and `StableDiffusionXLControlNetImg2ImgPipeline`. These pipelines are not officially implemented in [diffusers](https://github.com/huggingface/diffusers) yet, but enable more accurate image generation/editing processes. 
## StableDiffusionXLControlNetInpaintPipeline
**SDXL + Inpainting + ControlNet pipeline**

![inpaint_depth drawio](https://github.com/kamata1729/SDXL_controlnet_inpait_img2img_pipelines/assets/26928144/6e2c5af3-57ef-4286-af36-de6ba060119c)

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
# construct pipeline
import torch
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
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
**SDXL + Img2Img + ControlNet pipeline**
![img2img_canny drawio (1)](https://github.com/kamata1729/SDXL_controlnet_inpait_img2img_pipelines/assets/26928144/029a7cbe-3e03-4b97-80f7-f5af2d3df2e2)

Sample codes:
```shell
# for depth conditioned controlnet
python test_controlnet_img2img_sd_xl_depth.py
# for canny image conditioned controlnet
python test_controlnet_img2img_sd_xl_canny.py
```

Specific usage is as follows:
```python
# construct pipeline
import torch
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from pipeline_controlnet_img2img_sd_xl import StableDiffusionXLControlNetImg2ImgPipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_model_cpu_offload()

...

# image generation conditioned with control_image & init_image
image = pipe(
    "futuristic-looking woman",
    num_inference_steps=20,
    generator=generator,
    image=init_image,
    control_image=depth_image,
).images[0]
...
