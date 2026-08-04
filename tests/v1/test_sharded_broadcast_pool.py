# SPDX-License-Identifier: Apache-2.0
"""Tests for sharded-broadcast pool device binding and failure cleanup."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


class TestPipelinePoolFailureCleanup:
    def test_sender_releases_cpu_mem_objs_when_pool_alloc_fails(self, monkeypatch):
        """Pool OOM must not leak sender CPU mem_objs.

        retrieve() skips ref_count_down for the first rank under
        save_only_first_rank, so the pipeline finally-block is the only
        owner that can release them.
        """
        # First Party
        from lmcache_ascend.v1 import cache_engine as ce_mod

        monkeypatch.setattr(ce_mod.torch.npu, "current_device", lambda: 0)

        engine = object.__new__(AscendLMCacheEngine)
        engine.metadata = SimpleNamespace(
            worker_id=0,
            first_rank=0,
            is_first_rank=lambda: True,
        )
        engine.broadcast_stream = SimpleNamespace()
        engine.broadcast_fn = Mock()
        engine._ensure_merged_pool = Mock(return_value=False)
        engine._merged_pool = []
        engine._pool_scatter_ev = []
        engine._pool_bytes = 0

        mem_a = SimpleNamespace(ref_count_down=Mock())
        mem_b = SimpleNamespace(ref_count_down=Mock())
        chunks = [
            (object(), mem_a, 0, 256),
            (object(), mem_b, 256, 512),
        ]
        plan = {
            "meta": [],
            "shard_plan": [(0, 2)],
            "shard_layouts": [[(0, 0, 64), (1, 64, 64)]],
            "max_shard_bytes": 128,
        }
        load_stream = SimpleNamespace(synchronize=Mock())

        with pytest.raises(RuntimeError, match="Failed to allocate merged broadcast"):
            engine._pipeline_broadcast_and_load(
                plan,
                load_stream,
                reordered_chunks=chunks,
            )

        mem_a.ref_count_down.assert_called_once()
        mem_b.ref_count_down.assert_called_once()
        # Must not enter the broadcast collective after pool failure.
        engine.broadcast_fn.assert_not_called()
        engine._ensure_merged_pool.assert_called_once_with(128, "npu:0")

    def test_pool_alloc_uses_current_device(self, monkeypatch):
        """Broadcast pool must be allocated on npu:{current_device()}."""
        # First Party
        from lmcache_ascend.v1 import cache_engine as ce_mod

        monkeypatch.setattr(ce_mod.torch.npu, "current_device", lambda: 1)

        engine = object.__new__(AscendLMCacheEngine)
        engine.metadata = SimpleNamespace(
            worker_id=9,
            first_rank=0,
            is_first_rank=lambda: False,
        )
        engine.broadcast_stream = SimpleNamespace()
        engine.broadcast_fn = Mock()
        engine._ensure_merged_pool = Mock(return_value=False)
        engine._merged_pool = []
        engine._pool_scatter_ev = []
        engine._pool_bytes = 0

        plan = {
            "meta": [],
            "shard_plan": [],
            "shard_layouts": [],
            "max_shard_bytes": 256,
        }
        load_stream = SimpleNamespace(synchronize=Mock())

        with pytest.raises(RuntimeError, match="npu:1"):
            engine._pipeline_broadcast_and_load(
                plan,
                load_stream,
                ret_mask=None,
            )

        engine._ensure_merged_pool.assert_called_once_with(256, "npu:1")

    def test_logs_error_when_load_stream_sync_fails(self, monkeypatch):
        """Cleanup must log if load_stream.synchronize() raises."""
        # First Party
        from lmcache_ascend.v1 import cache_engine as ce_mod

        monkeypatch.setattr(ce_mod.torch.npu, "current_device", lambda: 0)
        log_error = Mock()
        monkeypatch.setattr(ce_mod.logger, "error", log_error)

        engine = object.__new__(AscendLMCacheEngine)
        engine.metadata = SimpleNamespace(
            worker_id=0,
            first_rank=0,
            is_first_rank=lambda: True,
        )
        engine.broadcast_stream = SimpleNamespace()
        engine.broadcast_fn = Mock()
        engine._ensure_merged_pool = Mock(return_value=False)
        engine._merged_pool = []
        engine._pool_scatter_ev = []
        engine._pool_bytes = 0

        mem = SimpleNamespace(ref_count_down=Mock())
        chunks = [(object(), mem, 0, 256)]
        plan = {
            "meta": [],
            "shard_plan": [(0, 1)],
            "shard_layouts": [[(0, 0, 64)]],
            "max_shard_bytes": 64,
        }
        load_stream = SimpleNamespace(
            synchronize=Mock(side_effect=RuntimeError("sync failed"))
        )

        with pytest.raises(RuntimeError, match="Failed to allocate merged broadcast"):
            engine._pipeline_broadcast_and_load(
                plan,
                load_stream,
                reordered_chunks=chunks,
            )

        log_error.assert_called_once()
        assert "synchronize load_stream" in log_error.call_args.args[0]
        mem.ref_count_down.assert_called_once()
