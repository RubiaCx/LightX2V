# export WAN22_T2V_MODEL_PATH_BASE=/home/scratch.rubchen_gpu_1/hf_cache/hub/models--Wan-AI--Wan2.2-T2V-A14B
# export WAN22_T2V_MODEL_PATH="$(ls -d ${WAN22_T2V_MODEL_PATH_BASE}/snapshots/* | head -n 1)"
# torchrun --nproc_per_node=8 run_lightx2v_wan22_t2v_8gpu_cfg2_seq4_with_warmup.py
#
# 8 GPU = cfg_p_size(2) * seq_p_size(4)
# config: configs/dist_infer/wan22_moe_t2v_cfg_ulysses.json

import os
import sys
from datetime import datetime
from time import perf_counter

LIGHTX2V_PATH = os.getenv("LIGHTX2V_PATH", "/home/scratch.rubchen_gpu_1/LightX2V")
MODEL_PATH = os.getenv("WAN22_T2V_MODEL_PATH", "/home/scratch.rubchen_gpu_1/hf_cache/hub/models--Wan-AI--Wan2.2-T2V-A14B")

sys.path.append(LIGHTX2V_PATH)

from lightx2v import LightX2VPipeline  # noqa: E402
from loguru import logger  # noqa: E402

def _get_rank() -> int:
    # Works before torch.distributed is initialized (torchrun sets RANK/LOCAL_RANK).
    try:
        return int(os.getenv("RANK", "0"))
    except Exception:
        return 0


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

ts = datetime.now().strftime("%y%m%d%H%M%S")
model_cls = "wan2.2_moe"
task = "t2v"

BENCH_ROOT = os.getenv("SGLANG_BENCH_ROOT", "/home/scratch.rubchen_gpu_1/sglang")
LOG_DIR = os.path.join(BENCH_ROOT, "logs")
OUT_DIR = os.path.join(BENCH_ROOT, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

rank = _get_rank()
run_tag = f"{model_cls}_{task}_cfg2_seq4_{ts}"
log_file = os.path.join(LOG_DIR, f"{run_tag}_rank{rank}.log")
save_result_path = os.path.join(OUT_DIR, f"{run_tag}.mp4")

# Write all loguru logs to file as well.
logger.add(log_file, enqueue=True, backtrace=False, diagnose=False)
logger.info(f"rank={rank}")
logger.info(f"LIGHTX2V_PATH={LIGHTX2V_PATH}")
logger.info(f"MODEL_PATH={MODEL_PATH}")
logger.info(f"BENCH_ROOT={BENCH_ROOT}")
logger.info(f"log_file={log_file}")
logger.info(f"save_result_path={save_result_path}")

pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls=model_cls,
    task=task,
)

pipe.create_generator(config_json=f"{LIGHTX2V_PATH}/configs/dist_infer/wan22_moe_t2v_cfg_ulysses.json")

# Generation parameters
seed = 42
prompt = (
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, "
    "while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming "
    "through the window."
)
negative_prompt = " "
target_shape = [720, 1280]

# warmup
_cuda_sync()
t0 = perf_counter()
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
_cuda_sync()
t1 = perf_counter()
logger.info(f"warmup_s={t1 - t0:.6f}")

# measure run
_cuda_sync()
t2 = perf_counter()
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
    return_result_tensor=True,
)
_cuda_sync()
t3 = perf_counter()
logger.info(f"measure_s={t3 - t2:.6f}")
logger.info("done")

