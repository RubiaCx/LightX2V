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
