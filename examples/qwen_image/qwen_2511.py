"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2509, use model_cls="qwen-image-edit-2509"
pipe = LightX2VPipeline(
    model_path="/workspace/models/Qwen-Image-Edit-2511",
    model_cls="qwen-image-edit-2511",
    task="i2i",
)

# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn3",
    auto_resize=True,
    infer_steps=40,
    guidance_scale=4,
)

# Generation parameters
seed = 42
prompt = "make the girl in Figure 1 dance with thecapybara in Fiqure 2."
negative_prompt = ""
image_path = "/workspace/TI2I_Qwen_Image_Edit_Input.jpg,/workspace/TI2I_Qwen_Image_Edit_Input2.jpg"  # or "/path/to/img_0.jpg,/path/to/img_1.jpg"
save_result_path = "/workspace/LightX2V_CFG_Qwen_Image_Edit_Output.png"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)