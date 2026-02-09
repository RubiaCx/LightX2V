# torchrun --nproc_per_node=8 run_lightx2v_wan22_t2v_8gpu_c8s8_cfg0_with_warmup.py
#
# Goal: Make LightX2V run closer to SGLang baseline settings:
# - steps=27, 720p, frames=81, seed=42, prompt same
# - seq parallel = 8 (Ulysses), CFG0
# - Print SGLang-like summary lines (avg step / denoise / decode / peak mem / e2e)
#
# Config file (renamed only):
#   configs/dist_infer/wan22_moe_t2v_c8s8.json

import os
import sys
import re
from datetime import datetime
from time import perf_counter

LIGHTX2V_PATH = os.getenv("LIGHTX2V_PATH", "/home/scratch.rubchen_gpu_1/LightX2V")
MODEL_PATH = os.getenv(
    "WAN22_T2V_MODEL_PATH",
    "/home/scratch.rubchen_gpu_1/hf_cache/hub/models--Wan-AI--Wan2.2-T2V-A14B/snapshots/c8c270b13ee05bfa474194ac9fb07a5868a97cea",
)

sys.path.append(LIGHTX2V_PATH)

from lightx2v import LightX2VPipeline  # noqa: E402
from loguru import logger  # noqa: E402


def _get_rank() -> int:
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


def _reset_peak_mem():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _get_peak_mem_gb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        return None
    return None


def _parse_section(text: str, start_tag: str, end_tag: str) -> str:
    """Return substring between the last start_tag and the following end_tag."""
    s = text.rfind(start_tag)
    if s < 0:
        return ""
    e = text.find(end_tag, s + len(start_tag))
    if e < 0:
        return text[s:]
    return text[s : e + len(end_tag)]


def _extract_last_float(block: str, pattern: str):
    """Return last float matched by regex pattern with one capturing group."""
    vals = re.findall(pattern, block)
    if not vals:
        return None
    return float(vals[-1])


def _extract_all_floats(block: str, pattern: str):
    vals = re.findall(pattern, block)
    return [float(v) for v in vals]


def _safe_shutdown_pipe(pipe):
    """Try best-effort shutdown to reduce leaked resources warnings."""
    for name in ["shutdown", "close", "finalize"]:
        fn = getattr(pipe, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _destroy_process_group():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


# -------------------- run config --------------------
ts = datetime.now().strftime("%y%m%d%H%M%S")
model_cls = "wan2.2_moe"
task = "t2v"

BENCH_ROOT = os.getenv("SGLANG_BENCH_ROOT", "/home/scratch.rubchen_gpu_1/sglang")
LOG_DIR = os.path.join(BENCH_ROOT, "logs")
OUT_DIR = os.path.join(BENCH_ROOT, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

rank = _get_rank()
is_rank0 = rank == 0

# renamed tag to reflect c8s8 + cfg0 intent
run_tag = f"{model_cls}_{task}_c8s8_cfg0_{ts}"
log_file = os.path.join(LOG_DIR, f"{run_tag}_rank{rank}.log")
save_result_path = os.path.join(OUT_DIR, f"{run_tag}.mp4")
save_path = save_result_path if is_rank0 else None

logger.add(log_file, enqueue=True, backtrace=False, diagnose=False)
logger.info(f"rank={rank}")
logger.info(f"LIGHTX2V_PATH={LIGHTX2V_PATH}")
logger.info(f"MODEL_PATH={MODEL_PATH}")
logger.info(f"BENCH_ROOT={BENCH_ROOT}")
logger.info(f"log_file={log_file}")
logger.info(f"save_result_path(rank0_only)={save_result_path if is_rank0 else 'None'}")

pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls=model_cls,
    task=task,
)

# IMPORTANT: config renamed only (path same)
config_name = "wan22_moe_t2v_c8s8.json"
pipe.create_generator(config_json=f"{LIGHTX2V_PATH}/configs/dist_infer/{config_name}")

# Generation parameters
seed = 42
prompt = (
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, "
    "while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming "
    "through the window."
)
# prefer empty negative prompt to avoid subtle differences vs CLI default
negative_prompt = ""
target_shape = [720, 1280]

# -------------------- warmup --------------------
WARMUP_START = f"===WARMUP_START {run_tag} rank{rank}==="
WARMUP_END = f"===WARMUP_END {run_tag} rank{rank}==="

logger.info(WARMUP_START)
_cuda_sync()
_reset_peak_mem()
t0 = perf_counter()
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_path,  # rank0 only
)
_cuda_sync()
t1 = perf_counter()
logger.info(WARMUP_END)

if is_rank0:
    logger.info(f"warmup_s={t1 - t0:.6f}")
    warmup_peak_gb = _get_peak_mem_gb()
    if warmup_peak_gb is not None:
        logger.info(f"warmup_peak_gpu_mem_gb={warmup_peak_gb:.2f}")

# -------------------- measure --------------------
MEASURE_START = f"===MEASURE_START {run_tag} rank{rank}==="
MEASURE_END = f"===MEASURE_END {run_tag} rank{rank}==="

logger.info(MEASURE_START)
_cuda_sync()
_reset_peak_mem()
t2 = perf_counter()
pipe.generate(
    seed=seed,
    prompt=prompt,
    target_shape=target_shape,
    negative_prompt=negative_prompt,
    save_result_path=save_path,  # rank0 only
    return_result_tensor=is_rank0,  # rank0 only
)
_cuda_sync()
t3 = perf_counter()
logger.info(MEASURE_END)

measure_s = t3 - t2
measure_peak_gb = _get_peak_mem_gb()

if is_rank0:
    logger.info(f"measure_s={measure_s:.6f}")
    logger.info(f"Video Saved in {save_result_path}")
    logger.info("done")

# -------------------- parse stats (rank0 only) --------------------
# Parse only rank0's log and only the MEASURE section to avoid mixing warmup/other runs.
if is_rank0:
    try:
        text = open(log_file, "r", errors="ignore").read()
        block = _parse_section(text, MEASURE_START, MEASURE_END)

        # Prefer explicit profiler totals if present
        dit_total = _extract_last_float(block, r"Run DiT cost\s*([0-9.]+)\s*seconds")
        vae_total = _extract_last_float(block, r"Run VAE Decoder cost\s*([0-9.]+)\s*seconds")
        pipe_total = _extract_last_float(block, r"RUN pipeline cost\s*([0-9.]+)\s*seconds")

        # Per-step mean from rank0 infer_main, if present
        infer_main_all = _extract_all_floats(
            block, r"Rank 0 - .*?infer_main cost\s*([0-9.]+)\s*seconds"
        )
        infer_main_mean = (sum(infer_main_all) / len(infer_main_all)) if infer_main_all else None
        infer_main_steps = len(infer_main_all)

        # Choose "best available" totals for denoise/vae/total
        denoise_total = dit_total if dit_total is not None else (sum(infer_main_all) if infer_main_all else None)
        total_total = pipe_total if pipe_total is not None else measure_s

        # Clean summary
        logger.info("========== STATS (rank0, measure run) ==========")
        logger.info(
            f"TOTAL_s (pipeline): {total_total:.6f}  "
            f"(source={'RUN pipeline cost' if pipe_total is not None else 'measure_s'})"
        )
        if denoise_total is not None:
            logger.info(
                f"DENOISE_s: {denoise_total:.6f}  "
                f"(source={'Run DiT cost' if dit_total is not None else 'sum(infer_main)'})"
            )
        else:
            logger.info("DENOISE_s: N/A (no Run DiT / infer_main found in log section)")
        if vae_total is not None:
            logger.info(f"VAE_s: {vae_total:.6f}  (source=Run VAE Decoder cost)")
        else:
            logger.info("VAE_s: N/A (no Run VAE Decoder found in log section)")
        if infer_main_mean is not None:
            logger.info(
                f"infer_main_avg_s_per_step: {infer_main_mean:.6f}  steps_seen={infer_main_steps}"
            )
        else:
            logger.info("infer_main_avg_s_per_step: N/A")
        logger.info("===============================================")

        # SGLang-like lines (approximate mapping)
        if infer_main_mean is not None:
            logger.info(f"[DenoisingStage] average time per step: {infer_main_mean:.4f} seconds")
        if denoise_total is not None:
            logger.info(f"[DenoisingStage] finished in {denoise_total:.4f} seconds")

        logger.info("[DecodingStage] started...")
        if vae_total is not None:
            logger.info(f"[DecodingStage] finished in {vae_total:.4f} seconds")

        if measure_peak_gb is not None:
            logger.info(f"Peak GPU memory: {measure_peak_gb:.2f} GB")

        # Use TOTAL_s as the closest proxy of end-to-end in measure section
        logger.info(f"Pixel data generated successfully in {total_total:.2f} seconds")
        logger.info(f"Completed batch processing. Generated 1 outputs in {total_total:.2f} seconds")
        logger.info(f"Warmed-up request processed in {total_total:.2f} seconds (with warmup excluded)")

    except Exception as e:
        logger.warning(f"Failed to parse stats from log_file={log_file}: {e}")

_safe_shutdown_pipe(pipe)
_destroy_process_group()