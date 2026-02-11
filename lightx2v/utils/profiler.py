import asyncio
import os
import threading
import time
from datetime import datetime
from functools import wraps

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)
_excluded_time_local = threading.local()
_stage_records_local = threading.local()
_stage_scope_local = threading.local()


def stage_ts() -> str:
    """Match style like: [02-08 09:33:11]"""
    return datetime.now().strftime("%m-%d %H:%M:%S")


def STAGE_LOG_ENABLED() -> bool:
    """
    Enable stage-style logs like:
      [MM-DD HH:MM:SS] [TextEncodingStage] started...
      [MM-DD HH:MM:SS] [TextEncodingStage] finished in 0.1234 seconds
    """
    return os.getenv("LIGHTX2V_STAGE_LOG", "0") in ("1", "true", "True")


def STAGE_LOG_RANK0_ONLY() -> bool:
    return os.getenv("LIGHTX2V_STAGE_RANK0_ONLY", "1") in ("1", "true", "True")


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def STAGE_SUMMARY_ENABLED() -> bool:
    # Default on when stage log is enabled
    v = os.getenv("LIGHTX2V_STAGE_SUMMARY", "")
    if v == "":
        return STAGE_LOG_ENABLED()
    return v in ("1", "true", "True")


def STAGE_SUMMARY_INCLUDE_INIT() -> bool:
    # Include init/load-weights summary at the end of generate()
    return os.getenv("LIGHTX2V_STAGE_SUMMARY_INCLUDE_INIT", "1") in ("1", "true", "True")


def _get_stage_scope() -> str:
    if not hasattr(_stage_scope_local, "scope"):
        _stage_scope_local.scope = "request"
    return _stage_scope_local.scope


def _set_stage_scope(scope: str):
    _stage_scope_local.scope = scope


class StageScope:
    """Temporarily switch stage recording scope (init/request)."""

    def __init__(self, scope: str):
        self.scope = scope
        self.prev = None

    def __enter__(self):
        self.prev = _get_stage_scope()
        _set_stage_scope(self.scope)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev is not None:
            _set_stage_scope(self.prev)
        return False


def _get_stage_records():
    if not hasattr(_stage_records_local, "records"):
        # two buffers:
        # - init: weights/config init (create_generator)
        # - request: one generate/run_pipeline invocation
        _stage_records_local.records = {"init": [], "request": []}
    return _stage_records_local.records


def stage_reset(scope: str = "request"):
    rec = _get_stage_records()
    if scope == "all":
        rec["init"].clear()
        rec["request"].clear()
    else:
        rec.setdefault(scope, [])
        rec[scope].clear()


def stage_record(stage_name: str, elapsed_s: float, extra=None):
    try:
        rec = {"stage": stage_name, "elapsed_s": float(elapsed_s), "extra": extra or {}}
        scope = _get_stage_scope()
        _get_stage_records().setdefault(scope, [])
        _get_stage_records()[scope].append(rec)
    except Exception:
        # best-effort; never fail inference due to logging
        pass


def stage_attach_extra_to_last(stage_name: str, extra: dict):
    """Attach extra fields to the most recent record of a given stage (best-effort)."""
    try:
        scope = _get_stage_scope()
        records = _get_stage_records().get(scope, [])
        for i in range(len(records) - 1, -1, -1):
            if records[i].get("stage") == stage_name:
                ex = records[i].get("extra") or {}
                ex.update(extra or {})
                records[i]["extra"] = ex
                return
    except Exception:
        pass


def _stage_print_summary_from_records(records, title: str = "StageSummary"):
    if not records:
        logger.info(f"[{stage_ts()}] [{title}] (no stage records)")
        return 0.0

    # Aggregate by stage name: sum and count
    agg = {}
    extras = {}
    for r in records:
        name = r.get("stage", "UnknownStage")
        agg.setdefault(name, {"sum": 0.0, "n": 0})
        agg[name]["sum"] += float(r.get("elapsed_s", 0.0))
        agg[name]["n"] += 1
        ex = r.get("extra") or {}
        if ex:
            extras.setdefault(name, []).append(ex)

    preferred = [
        "LoadWeightsStage",
        "RequestE2EStage",
        "InputValidationStage",
        "TextEncodingStage",
        "ConditioningStage",
        "TimestepPreparationStage",
        "LatentPreparationStage",
        "DenoisingStage",
        "DecodingStage",
        "SaveOutputStage",
    ]
    ordered = [s for s in preferred if s in agg] + sorted([s for s in agg.keys() if s not in preferred])

    logger.info(f"[{stage_ts()}] [{title}] ==============================")
    total_components = 0.0
    for s in ordered:
        info = agg[s]
        line = f"[{stage_ts()}] [{title}] {s}: {info['sum']:.4f} seconds (n={info['n']})"
        if s == "DenoisingStage" and s in extras:
            last = extras[s][-1]
            if "avg_time_per_step_s" in last:
                line += f", avg_per_step={float(last['avg_time_per_step_s']):.4f}s"
        logger.info(line)
        # Avoid double counting: RequestE2EStage is a wall-time envelope over stages.
        if s != "RequestE2EStage":
            total_components += info["sum"]

    # Derived metrics to align with sglang server mode:
    # sglang `timings.total_duration_ms` excludes saving outputs, while LightX2V may include it.
    if "RequestE2EStage" in agg:
        req_e2e = float(agg["RequestE2EStage"]["sum"])
        save_s = float(agg.get("SaveOutputStage", {}).get("sum", 0.0))
        logger.info(f"[{stage_ts()}] [{title}] REQUEST_E2E(wall): {req_e2e:.4f} seconds")
        if save_s > 0:
            logger.info(f"[{stage_ts()}] [{title}] REQUEST_E2E(no_save): {max(req_e2e - save_s, 0.0):.4f} seconds")

    logger.info(f"[{stage_ts()}] [{title}] TOTAL(sum of components, excl RequestE2EStage): {total_components:.4f} seconds")
    logger.info(f"[{stage_ts()}] [{title}] ==============================")
    return total_components


def stage_print_summary(title: str = "StageSummary"):
    if not STAGE_SUMMARY_ENABLED():
        return
    if STAGE_LOG_RANK0_ONLY() and not _is_rank0():
        return

    rec = _get_stage_records()
    init_total = 0.0
    req_total = 0.0

    if STAGE_SUMMARY_INCLUDE_INIT():
        init_total = _stage_print_summary_from_records(list(rec.get("init", [])), title=f"{title}.INIT")
    req_total = _stage_print_summary_from_records(list(rec.get("request", [])), title=f"{title}.REQUEST")
    if STAGE_SUMMARY_INCLUDE_INIT():
        logger.info(f"[{stage_ts()}] [{title}.E2E] INIT+REQUEST: {init_total + req_total:.4f} seconds")


class StageContext:
    """
    Lightweight stage logger for end-to-end style tracing.

    Controlled by env:
      - LIGHTX2V_STAGE_LOG=1 enables logs
      - LIGHTX2V_STAGE_RANK0_ONLY=1 prints only rank0 (default)
      - LIGHTX2V_STAGE_SYNC=1 cuda/xpu sync at enter/exit for accurate timing (default)
    """

    def __init__(self, stage_name: str, sync=None, rank0_only=None):
        self.stage_name = stage_name
        if sync is None:
            sync = os.getenv("LIGHTX2V_STAGE_SYNC", "1") in ("1", "true", "True")
        if rank0_only is None:
            rank0_only = STAGE_LOG_RANK0_ONLY()
        self.sync = bool(sync)
        self.rank0_only = bool(rank0_only)
        self.enabled = STAGE_LOG_ENABLED() and (not self.rank0_only or _is_rank0())
        self.elapsed = None

    def _sync(self):
        if not self.sync:
            return
        try:
            # torch.cuda / torch.xpu / torch.npu provides synchronize()
            torch_device_module.synchronize()
        except Exception:
            # keep stage logging best-effort; profiling contexts may still error elsewhere
            pass

    def __enter__(self):
        if self.enabled:
            logger.info(f"[{stage_ts()}] [{self.stage_name}] started...")
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sync()
        self.elapsed = time.perf_counter() - self._t0
        if self.enabled:
            logger.info(f"[{stage_ts()}] [{self.stage_name}] finished in {self.elapsed:.4f} seconds")
        if self.elapsed is not None:
            stage_record(self.stage_name, self.elapsed)
        return False


def _get_excluded_time_stack():
    if not hasattr(_excluded_time_local, "stack"):
        _excluded_time_local.stack = []
    return _excluded_time_local.stack


class _ProfilingContext:
    def __init__(self, name, recorder_mode=0, metrics_func=None, metrics_labels=None):
        """
        recorder_mode = 0: disable recorder
        recorder_mode = 1: enable recorder
        recorder_mode = 2: enable recorder and force disable logger
        """
        if recorder_mode == 0:
            recorder_mode = GET_RECORDER_MODE()
        self.name = name
        if dist.is_initialized():
            self.rank_info = f"Rank {dist.get_rank()}"
        else:
            self.rank_info = "Single GPU"
        self.enable_recorder = recorder_mode > 0
        self.enable_logger = recorder_mode <= 1
        self.metrics_func = metrics_func
        self.metrics_labels = metrics_labels

    def __enter__(self):
        torch_device_module.synchronize()
        self.start_time = time.perf_counter()
        _get_excluded_time_stack().append(0.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch_device_module.synchronize()
        total_elapsed = time.perf_counter() - self.start_time
        excluded = _get_excluded_time_stack().pop()
        elapsed = total_elapsed - excluded
        if self.enable_recorder and self.metrics_func:
            if self.metrics_labels:
                self.metrics_func.labels(*self.metrics_labels).observe(elapsed)
            else:
                self.metrics_func.observe(elapsed)
        if self.enable_logger:
            logger.info(f"[Profile] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds")
        return False

    async def __aenter__(self):
        torch_device_module.synchronize()
        self.start_time = time.perf_counter()
        _get_excluded_time_stack().append(0.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        torch_device_module.synchronize()
        total_elapsed = time.perf_counter() - self.start_time
        excluded = _get_excluded_time_stack().pop()
        elapsed = total_elapsed - excluded
        if self.enable_recorder and self.metrics_func:
            if self.metrics_labels:
                self.metrics_func.labels(*self.metrics_labels).observe(elapsed)
            else:
                self.metrics_func.observe(elapsed)
        if self.enable_logger:
            logger.info(f"[Profile] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds")
        return False

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper


class _NullContext:
    # Context manager without decision branch logic overhead
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __call__(self, func):
        return func


class _ExcludedProfilingContext:
    """用于标记应该从外层 profiling 中排除的时间段"""

    def __init__(self, name=None):
        self.name = name
        if dist.is_initialized():
            self.rank_info = f"Rank {dist.get_rank()}"
        else:
            self.rank_info = "Single GPU"

    def __enter__(self):
        torch_device_module.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch_device_module.synchronize()
        elapsed = time.perf_counter() - self.start_time
        stack = _get_excluded_time_stack()
        for i in range(len(stack)):
            stack[i] += elapsed
        if self.name and CHECK_PROFILING_DEBUG_LEVEL(1):
            logger.info(f"[Profile-Excluded] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds (excluded from outer profiling)")
        return False

    async def __aenter__(self):
        torch_device_module.synchronize()
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        torch_device_module.synchronize()
        elapsed = time.perf_counter() - self.start_time
        stack = _get_excluded_time_stack()
        for i in range(len(stack)):
            stack[i] += elapsed
        if self.name and CHECK_PROFILING_DEBUG_LEVEL(1):
            logger.info(f"[Profile-Excluded] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds (excluded from outer profiling)")
        return False

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper


class _ProfilingContextL1(_ProfilingContext):
    """Level 1 profiling context with Level1_Log prefix."""

    def __init__(self, name, recorder_mode=0, metrics_func=None, metrics_labels=None):
        super().__init__(f"Level1_Log {name}", recorder_mode, metrics_func, metrics_labels)


class _ProfilingContextL2(_ProfilingContext):
    """Level 2 profiling context with Level2_Log prefix."""

    def __init__(self, name, recorder_mode=0, metrics_func=None, metrics_labels=None):
        super().__init__(f"Level2_Log {name}", recorder_mode, metrics_func, metrics_labels)


"""
PROFILING_DEBUG_LEVEL=0: [Default] disable all profiling
PROFILING_DEBUG_LEVEL=1: enable ProfilingContext4DebugL1
PROFILING_DEBUG_LEVEL=2: enable ProfilingContext4DebugL1 and ProfilingContext4DebugL2
"""
ProfilingContext4DebugL1 = _ProfilingContextL1 if CHECK_PROFILING_DEBUG_LEVEL(1) else _NullContext  # if user >= 1, enable profiling
ProfilingContext4DebugL2 = _ProfilingContextL2 if CHECK_PROFILING_DEBUG_LEVEL(2) else _NullContext  # if user >= 2, enable profiling
ExcludedProfilingContext = _ExcludedProfilingContext if CHECK_PROFILING_DEBUG_LEVEL(1) else _NullContext
