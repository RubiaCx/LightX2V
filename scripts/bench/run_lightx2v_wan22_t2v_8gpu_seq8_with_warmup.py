# torchrun --nproc_per_node=8 run_lightx2v_wan22_t2v_8gpu_seq8_with_warmup.py
#
# 8 GPU = cfg_p_size(1) * seq_p_size(8)  (no CFG parallel)
# config: configs/dist_infer/wan22_moe_t2v_seq8_ulysses.json

import os
import sys
from datetime import datetime

LIGHTX2V_PATH = os.getenv("LIGHTX2V_PATH", "/path/to/LightX2V")
MODEL_PATH = os.getenv("WAN22_T2V_MODEL_PATH", "/path/to/Wan-AI/Wan2.2-T2V-A14B")

sys.path.append(LIGHTX2V_PATH)

from lightx2v import LightX2VPipeline  # noqa: E402

ts = datetime.now().strftime("%y%m%d%H%M%S")
model_cls = "wan2.2_moe"
task = "t2v"

pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls=model_cls,
    task=task,
)

pipe.create_generator(config_json=f"{LIGHTX2V_PATH}/configs/dist_infer/wan22_moe_t2v_seq8_ulysses.json")

# Generation parameters
seed = 42
prompt = (
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, "
    "while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming "
    "through the window."
)
negative_prompt = " "
target_shape = [720, 1280]

save_result_path = f"{LIGHTX2V_PATH}/save_results/{model_cls}_{task}_seq8_{ts}.mp4"

# warmup
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)

# measure run
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    return_result_tensor=True,
)

