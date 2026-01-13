#!/bin/bash

# Base paths
export lightx2v_path="/workspace/LightX2V"
# This points to the Official Wan2.2-TI2V-5B model (expected path after `huggingface-cli download Wan-AI/Wan2.2-TI2V-5B`)
# You may need to update the snapshot hash if 'current' symlink doesn't work or use the specific hash folder.
export model_path="/workspace/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/current"

export CUDA_VISIBLE_DEVICES=0

# Load base environment settings from LightX2V (sets PYTHONPATH, DTYPE, etc.)
source ${lightx2v_path}/scripts/base/base.sh

echo "Running Wan2.2 TI2V I2V test..."
echo "Model Path: ${model_path}"

# Ensure we are using the correct config for Wan2.2 TI2V I2V
python3 -m lightx2v.infer \
    --model_cls wan2.2 \
    --task i2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/wan22/wan_ti2v_i2v.json" \
    --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
    --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "${lightx2v_path}/save_results/test_wan22_ti2v_i2v.mp4"
