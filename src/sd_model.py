from diffusers import DiffusionPipeline
import torch
from diffusers import DPMSolverMultistepScheduler

# https://huggingface.co/docs/diffusers/v0.31.0/stable_diffusion
# https://pillow.readthedocs.io/en/stable/reference/Image.html

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
pipeline = pipeline.to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
generator = torch.Generator("cuda").manual_seed(0)

def generate_image(prompt):
    image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
    return image


# 主程序入口
if __name__ == "__main__":
    # prompt = "portrait photo of a old warrior chief"
    # image = generate_image(prompt)
    # print(image)
    # image.save('outputs/my_image2.png')
    #
    # 无法使用中文提示词，只支持英文
    # prompts = ['股票交易市场', '科技股投资', '科技创新与市场潜力']
    # for prompt in prompts:
    #     image = generate_image(prompt)
    #     image.save(f'outputs/{prompt}.png')

    prompts = ['Stock Market', 'Technology Stock Investment', 'Technological innovation and market potential']
    for prompt in prompts:
        image = generate_image(prompt)
        image.save(f'outputs/{prompt}.png')
