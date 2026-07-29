# SPDX-License-Identifier: Apache-2.0
"""Tests for host-local NPU device selection in AscendLMCacheEngine."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


def _engine_with_metadata(**meta_fields):
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = SimpleNamespace(**meta_fields)
    return engine


class TestLocalNpuDeviceId:
    def test_prefers_local_worker_id_over_global_rank(self):
        """Multi-node: global rank 9 must map to local NPU 1, not npu:9."""
        engine = _engine_with_metadata(worker_id=9, local_worker_id=1)
        assert engine._local_npu_device_id() == 1

    def test_uses_local_worker_id_when_equal_to_worker_id(self):
        engine = _engine_with_metadata(worker_id=3, local_worker_id=3)
        assert engine._local_npu_device_id() == 3

    def test_fallback_modulo_when_local_worker_id_missing(self, monkeypatch):
        engine = _engine_with_metadata(worker_id=9)
        monkeypatch.setattr(
            "lmcache_ascend.v1.cache_engine.torch.npu.device_count",
            lambda: 8,
        )
        assert engine._local_npu_device_id() == 1


class TestPipelinePoolFailureCleanup:
    def test_sender_releases_cpu_mem_objs_when_pool_alloc_fails(self):
        """Pool OOM must not leak sender CPU mem_objs.

        retrieve() skips ref_count_down for the first rank under
        save_only_first_rank, so the pipeline finally-block is the only
        owner that can release them.
        """
        engine = object.__new__(AscendLMCacheEngine)
        engine.metadata = SimpleNamespace(
            worker_id=0,
            local_worker_id=0,
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
        # Pool was attempted on the local device string.
        engine._ensure_merged_pool.assert_called_once_with(128, "npu:0")

    def test_pool_alloc_uses_local_worker_id_device(self):
        """Broadcast pool must be allocated on npu:{local_worker_id}."""
        engine = object.__new__(AscendLMCacheEngine)
        engine.metadata = SimpleNamespace(
            worker_id=9,
            local_worker_id=1,
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
