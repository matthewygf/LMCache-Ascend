# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Unit tests for host-staging H2H copy cancellation safety.

These tests exercise ``HcclChannel._async_h2h_copy`` on a minimally
constructed instance (no live HCCL/NPU traffic). Import still requires
the extension module to be present.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import asyncio
import threading
import time

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
import pytest
import torch

try:
    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_channel import HcclChannel

    _hccl_channel_available = True
except ImportError:
    _hccl_channel_available = False
    HcclChannel = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.skipif(
    not _hccl_channel_available,
    reason="hccl channel extension not built (set HCOMM_SRC_PATH at build time)",
)


def _make_channel(*, copy_threads: int = 2) -> HcclChannel:
    channel = object.__new__(HcclChannel)
    channel._os_staging_copy_threads = copy_threads
    channel._staging_copy_pool = ThreadPoolExecutor(
        max_workers=copy_threads,
        thread_name_prefix="test-staging-copy",
    )
    return channel


def _run(coro):
    return asyncio.run(coro)


class TestAsyncH2HCopyDrain:
    def test_cancel_waits_for_executor_workers(self, monkeypatch):
        """Cancelled await must not return while H2H workers still write.

        Sync-get timeout cancels ``_handle_pull_mode_transfer`` during
        ``copy_receiver_staging_to``. Callers then ``release_staged()``.
        If ThreadPoolExecutor workers are orphaned, arena freelist reuse
        UAFs the pages under in-flight ``torch._foreach_copy_``.
        """
        channel = _make_channel(copy_threads=2)
        started = threading.Barrier(3)  # 2 workers + main
        release_gate = threading.Event()
        unfinished_at_return = []

        def slow_foreach(dst_slice, src_slice):
            started.wait(timeout=2)
            # Hold the copy open until the test observes cancel path.
            assert release_gate.wait(timeout=2)
            # Marker read after gate opens — must complete before await returns.
            unfinished_at_return.append(threading.current_thread().name)

        monkeypatch.setattr(torch, "_foreach_copy_", slow_foreach)

        src = [torch.ones(8), torch.ones(8)]
        dst = [torch.zeros(8), torch.zeros(8)]

        async def _cancel_mid_copy():
            task = asyncio.create_task(channel._async_h2h_copy(src, dst))
            # Wait until both slice workers have entered the copy body.
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: started.wait(timeout=2)
            )
            task.cancel()
            # Let workers finish only after cancel is requested — the
            # drained await must still block until they complete.
            await asyncio.sleep(0.05)
            release_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            # If drain worked, both workers finished before await returned.
            assert len(unfinished_at_return) == 2

        try:
            _run(_cancel_mid_copy())
        finally:
            release_gate.set()
            channel._staging_copy_pool.shutdown(wait=True)

    def test_exception_in_one_slice_drains_siblings(self, monkeypatch):
        """A failing slice must not freelist while sibling workers still run."""
        channel = _make_channel(copy_threads=2)
        sibling_started = threading.Event()
        sibling_done = threading.Event()
        fail_gate = threading.Event()
        call_count = {"n": 0}
        lock = threading.Lock()

        def flaky_foreach(dst_slice, src_slice):
            with lock:
                call_count["n"] += 1
                idx = call_count["n"]
            if idx == 1:
                sibling_started.set()
                # Slow sibling — must finish before _async_h2h_copy raises.
                time.sleep(0.2)
                sibling_done.set()
                return
            # Fast failing slice: wait until sibling has started, then raise.
            assert sibling_started.wait(timeout=2)
            fail_gate.set()
            raise RuntimeError("boom-slice")

        monkeypatch.setattr(torch, "_foreach_copy_", flaky_foreach)

        src = [torch.ones(4), torch.ones(4)]
        dst = [torch.zeros(4), torch.zeros(4)]

        async def _run_copy():
            with pytest.raises(RuntimeError, match="boom-slice"):
                await channel._async_h2h_copy(src, dst)
            assert sibling_done.is_set()

        try:
            _run(_run_copy())
        finally:
            channel._staging_copy_pool.shutdown(wait=True)


class TestStageCancelReleasesArena:
    def test_stage_cancelled_error_releases_staged(self, monkeypatch):
        """Cancelled stage() must return arena pages (after H2H drain)."""
        channel = _make_channel(copy_threads=1)
        channel._use_host_staging = True
        channel._staging_arena = MagicMock()
        channel._staging_lock = threading.Lock()

        slot = MagicMock()
        slot.tensor = torch.zeros(4)
        slot.meta = MagicMock(shape=(4,), dtype=torch.float32, fmt=MagicMock())
        channel._staging_arena.allocate.return_value = slot

        released = []

        def _release(objs):
            released.extend(objs)

        channel.release_staged = _release  # type: ignore[method-assign]
        channel.get_local_buffer_refs = MagicMock(return_value=([], []))

        async def cancel_copy(*_args, **_kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(channel, "_async_h2h_copy", cancel_copy)

        src_obj = MagicMock()
        src_obj.tensor = torch.ones(4)
        src_obj.meta = slot.meta

        async def _run_stage():
            with pytest.raises(asyncio.CancelledError):
                await channel.stage([src_obj])

        try:
            _run(_run_stage())
            assert released == [slot]
        finally:
            channel._staging_copy_pool.shutdown(wait=True)
