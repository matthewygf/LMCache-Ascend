# SPDX-License-Identifier: Apache-2.0
"""Tests for coordinated merged-pool OOM abort in sharded broadcast."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


def _engine(*, worker_id=0, world_size=2, first_rank=0, is_first=True):
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = SimpleNamespace(
        worker_id=worker_id,
        world_size=world_size,
        first_rank=first_rank,
        is_first_rank=lambda: is_first,
    )
    engine.broadcast_stream = SimpleNamespace(synchronize=Mock())
    engine.broadcast_fn = Mock()
    engine._merged_pool = [object(), object()]
    engine._pool_scatter_ev = [object(), object()]
    engine._pool_bytes = 1024
    engine._submit_togpu = Mock()
    return engine


def _fake_allgather(rank_flags: dict[int, bool]):
    """Simulate successive src-rooted broadcasts of per-rank bools."""

    def _broadcast(payload, src):
        # Source rank supplies its flag; others pass None and receive it.
        return rank_flags[src] if payload is None else payload

    return _broadcast


class TestAllgatherOk:
    def test_true_when_all_ranks_ok(self):
        engine = _engine(worker_id=1, world_size=2)
        engine.broadcast_object_fn = Mock(
            side_effect=_fake_allgather({0: True, 1: True})
        )
        assert engine._allgather_ok(True) is True
        assert engine.broadcast_object_fn.call_count == 2

    def test_false_when_peer_reports_failure(self):
        engine = _engine(worker_id=0, world_size=2)
        engine.broadcast_object_fn = Mock(
            side_effect=_fake_allgather({0: True, 1: False})
        )
        assert engine._allgather_ok(True) is False

    def test_false_when_local_ok_false(self):
        engine = _engine(worker_id=1, world_size=2)
        engine.broadcast_object_fn = Mock(
            side_effect=_fake_allgather({0: True, 1: False})
        )
        assert engine._allgather_ok(False) is False


class TestPipelinePoolOkGate:
    def test_asymmetric_oom_aborts_without_broadcast_fn(self):
        """Peer pool OOM must stop every rank before the shard collective."""
        engine = _engine(worker_id=0, world_size=2, is_first=True)
        # Local alloc succeeds; rank 1 reports failure via allgather.
        engine._ensure_merged_pool = Mock(return_value=True)
        engine.broadcast_object_fn = Mock(
            side_effect=_fake_allgather({0: True, 1: False})
        )

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

        with pytest.raises(RuntimeError, match="at least one TP rank"):
            engine._pipeline_broadcast_and_load(
                plan, load_stream, reordered_chunks=chunks
            )

        engine.broadcast_fn.assert_not_called()
        # retrieve() skips ref_count_down for the sender; finally owns it.
        mem_a.ref_count_down.assert_called_once_with()
        mem_b.ref_count_down.assert_called_once_with()

    def test_receiver_aborts_when_any_rank_fails_pool_alloc(self):
        engine = _engine(worker_id=1, world_size=2, is_first=False)
        engine._ensure_merged_pool = Mock(return_value=True)
        engine.broadcast_object_fn = Mock(
            side_effect=_fake_allgather({0: False, 1: True})
        )

        plan = {
            "meta": [],
            "shard_plan": [(0, 1)],
            "shard_layouts": [[(0, 0, 64)]],
            "max_shard_bytes": 64,
        }
        load_stream = SimpleNamespace(synchronize=Mock())

        with pytest.raises(RuntimeError, match="at least one TP rank"):
            engine._pipeline_broadcast_and_load(
                plan, load_stream, ret_mask=Mock()
            )

        engine.broadcast_fn.assert_not_called()

    def test_local_oom_still_participates_in_allgather(self):
        """Failing rank must not raise before peers observe the gate."""
        engine = _engine(worker_id=1, world_size=2, is_first=False)
        engine._ensure_merged_pool = Mock(return_value=False)
        seen_srcs: list[int] = []

        def _broadcast(payload, src):
            seen_srcs.append(src)
            return _fake_allgather({0: True, 1: False})(payload, src)

        engine.broadcast_object_fn = Mock(side_effect=_broadcast)

        plan = {
            "meta": [],
            "shard_plan": [(0, 1)],
            "shard_layouts": [[(0, 0, 64)]],
            "max_shard_bytes": 64,
        }
        load_stream = SimpleNamespace(synchronize=Mock())

        with pytest.raises(RuntimeError, match="at least one TP rank"):
            engine._pipeline_broadcast_and_load(
                plan, load_stream, ret_mask=Mock()
            )

        assert seen_srcs == [0, 1]
        engine.broadcast_fn.assert_not_called()
