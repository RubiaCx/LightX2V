# torchrun --nproc_per_node=8 run_lightx2v_wan22_t2v_8gpu_seq8_with_warmup.py
#
# 8 GPU = cfg_p_size(1) * seq_p_size(8)  (no CFG parallel)
# config: configs/dist_infer/wan22_moe_t2v_seq8_ulysses.json
'''
2026-02-10 08:57:02.090 | INFO     | __main__:<module>:239 - Video Saved in /home/scratch.rubchen_gpu_1/sglang/outputs/wan2.2_moe_t2v_seq8_260210085331.mp4
2026-02-10 08:57:02.090 | INFO     | __main__:<module>:240 - done
2026-02-10 08:57:02.090 | INFO     | __main__:<module>:242 - measure_peak_gpu_mem_gb_global_max: allocated=68.50 reserved=69.12
2026-02-10 08:57:02.097 | INFO     | __main__:<module>:264 - ========== STATS (rank0, measure run) ==========
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:265 - TOTAL_s (pipeline): 81.452815  (source=RUN pipeline cost)
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:270 - DENOISE_s: 81.245543  (source=Run DiT cost)
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:277 - VAE_s: 1.926902  (source=Run VAE Decoder cost)
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:281 - infer_main_avg_s_per_step: 2.911348  steps_seen=27
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:286 - ===============================================
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:289 - [DenoisingStage] average time per step: 2.9113 seconds
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:291 - [DenoisingStage] finished in 81.2455 seconds
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:293 - [DecodingStage] started...
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:295 - [DecodingStage] finished in 1.9269 seconds
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:298 - Peak GPU memory (global max over ranks): allocated=68.50 GB, reserved=69.12 GB
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:304 - Pixel data generated successfully in 81.45 seconds
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:305 - Completed batch processing. Generated 1 outputs in 81.45 seconds
2026-02-10 08:57:02.098 | INFO     | __main__:<module>:306 - Warmed-up request processed in 81.45 seconds (with warmup excluded)
'''
import os
import sys
import re
from datetime import datetime
from time import perf_counter

LIGHTX2V_PATH = os.getenv("LIGHTX2V_PATH", "/home/scratch.rubchen_gpu_1/LightX2V")
MODEL_PATH = os.getenv("WAN22_T2V_MODEL_PATH", "/home/scratch.rubchen_gpu_1/hf_cache/hub/models--Wan-AI--Wan2.2-T2V-A14B/snapshots/c8c270b13ee05bfa474194ac9fb07a5868a97cea")

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
            torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())
    except Exception:
        pass


def _get_peak_mem_bytes():
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            allocated = int(torch.cuda.max_memory_allocated(device=dev))
            reserved = int(torch.cuda.max_memory_reserved(device=dev))
            return allocated, reserved
    except Exception:
        return None
    return None


def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024**3)


def _dist_reduce_max_bytes(pair_bytes):
    """
    Reduce (allocated_bytes, reserved_bytes) across ranks with MAX.
    Returns reduced pair on all ranks if dist is initialized; otherwise returns input.
    """
    if pair_bytes is None:
        return None
    try:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return pair_bytes

        t = torch.tensor([pair_bytes[0], pair_bytes[1]], device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t[0].item()), int(t[1].item())
    except Exception:
        return pair_bytes


def _parse_section(text: str, start_tag: str, end_tag: str) -> str:
    s = text.rfind(start_tag)
    if s < 0:
        return ""
    e = text.find(end_tag, s + len(start_tag))
    if e < 0:
        return text[s:]
    return text[s : e + len(end_tag)]


def _extract_last_float(block: str, pattern: str):
    vals = re.findall(pattern, block)
    if not vals:
        return None
    return float(vals[-1])


def _extract_all_floats(block: str, pattern: str):
    vals = re.findall(pattern, block)
    return [float(v) for v in vals]


def _safe_shutdown_pipe(pipe):
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


ts = datetime.now().strftime("%y%m%d%H%M%S")
model_cls = "wan2.2_moe"
task = "t2v"

BENCH_ROOT = os.getenv("SGLANG_BENCH_ROOT", "/home/scratch.rubchen_gpu_1/sglang")
LOG_DIR = os.path.join(BENCH_ROOT, "logs")
OUT_DIR = os.path.join(BENCH_ROOT, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

rank = _get_rank()
is_rank0 = (rank == 0)

run_tag = f"{model_cls}_{task}_seq8_{ts}"
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

pipe.create_generator(config_json=f"{LIGHTX2V_PATH}/configs/dist_infer/wan22_moe_t2v_seq8_ulysses.json")

seed = 42
prompt = (
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, "
    "while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming "
    "through the window."
)
negative_prompt = " "
target_shape = [720, 1280]

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

warmup_peak_bytes = _get_peak_mem_bytes()
if warmup_peak_bytes is not None:
    logger.info(
        "warmup_peak_gpu_mem_gb: "
        f"allocated={_bytes_to_gb(warmup_peak_bytes[0]):.2f} "
        f"reserved={_bytes_to_gb(warmup_peak_bytes[1]):.2f}"
    )
warmup_peak_bytes_max = _dist_reduce_max_bytes(warmup_peak_bytes)
if is_rank0:
    logger.info(f"warmup_s={t1 - t0:.6f}")
    if warmup_peak_bytes_max is not None:
        logger.info(
            "warmup_peak_gpu_mem_gb_global_max: "
            f"allocated={_bytes_to_gb(warmup_peak_bytes_max[0]):.2f} "
            f"reserved={_bytes_to_gb(warmup_peak_bytes_max[1]):.2f}"
        )

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
measure_peak_bytes = _get_peak_mem_bytes()
if measure_peak_bytes is not None:
    logger.info(
        "measure_peak_gpu_mem_gb: "
        f"allocated={_bytes_to_gb(measure_peak_bytes[0]):.2f} "
        f"reserved={_bytes_to_gb(measure_peak_bytes[1]):.2f}"
    )
measure_peak_bytes_max = _dist_reduce_max_bytes(measure_peak_bytes)

if is_rank0:
    logger.info(f"measure_s={measure_s:.6f}")
    logger.info(f"Video Saved in {save_result_path}")
    logger.info("done")
    if measure_peak_bytes_max is not None:
        logger.info(
            "measure_peak_gpu_mem_gb_global_max: "
            f"allocated={_bytes_to_gb(measure_peak_bytes_max[0]):.2f} "
            f"reserved={_bytes_to_gb(measure_peak_bytes_max[1]):.2f}"
        )

if is_rank0:
    try:
        text = open(log_file, "r", errors="ignore").read()
        block = _parse_section(text, MEASURE_START, MEASURE_END)

        dit_total = _extract_last_float(block, r"Run DiT cost\s*([0-9.]+)\s*seconds")
        vae_total = _extract_last_float(block, r"Run VAE Decoder cost\s*([0-9.]+)\s*seconds")
        pipe_total = _extract_last_float(block, r"RUN pipeline cost\s*([0-9.]+)\s*seconds")

        infer_main_all = _extract_all_floats(block, r"Rank 0 - .*?infer_main cost\s*([0-9.]+)\s*seconds")
        infer_main_mean = (sum(infer_main_all) / len(infer_main_all)) if infer_main_all else None
        infer_main_steps = len(infer_main_all)

        denoise_total = dit_total if dit_total is not None else (sum(infer_main_all) if infer_main_all else None)
        total_total = pipe_total if pipe_total is not None else measure_s

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

        if infer_main_mean is not None:
            logger.info(f"[DenoisingStage] average time per step: {infer_main_mean:.4f} seconds")
        if denoise_total is not None:
            logger.info(f"[DenoisingStage] finished in {denoise_total:.4f} seconds")

        logger.info("[DecodingStage] started...")
        if vae_total is not None:
            logger.info(f"[DecodingStage] finished in {vae_total:.4f} seconds")

        if measure_peak_bytes_max is not None:
            logger.info(
                "Peak GPU memory (global max over ranks): "
                f"allocated={_bytes_to_gb(measure_peak_bytes_max[0]):.2f} GB, "
                f"reserved={_bytes_to_gb(measure_peak_bytes_max[1]):.2f} GB"
            )

        logger.info(f"Pixel data generated successfully in {total_total:.2f} seconds")
        logger.info(f"Completed batch processing. Generated 1 outputs in {total_total:.2f} seconds")
        logger.info(f"Warmed-up request processed in {total_total:.2f} seconds (with warmup excluded)")

    except Exception as e:
        logger.warning(f"Failed to parse stats from log_file={log_file}: {e}")

_safe_shutdown_pipe(pipe)
_destroy_process_group()
