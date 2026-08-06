# SPDX-License-Identifier: Apache-2.0
"""Tests for sharded-broadcast coordinated abort and stream drain."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


def _engine(*, worker_id=0, first_rank=0, is_first=True):
    engine = object.__new__(AscendLMCacheEngine)
    engine.metadata = SimpleNamespace(
        worker_id=worker_id,
        first_rank=first_rank,
        is_first_rank=lambda: is_first,
    )
    engine.broadcast_stream = SimpleNamespace(synchronize=Mock())
    engine.broadcast_fn = Mock()
    engine.broadcast_object_fn = Mock(side_effect=lambda obj, _src: obj)
    engine._ensure_merged_pool = Mock(return_value=True)
    engine._merged_pool = [object(), object()]
    engine._pool_scatter_ev = [object(), object()]
    engine._pool_bytes = 1024
    engine._submit_togpu = Mock()
    return engine


class TestSenderChunksBroadcastable:
    def test_true_when_all_raw_tensors_present(self):
        engine = _engine()
        chunks = [
            (object(), SimpleNamespace(raw_tensor=object()), 0, 256),
            (object(), SimpleNamespace(raw_tensor=object()), 256, 512),
        ]
        assert engine._sender_chunks_broadcastable(chunks) is True

    def test_false_when_proxy_raw_tensor_none(self):
        engine = _engine()
        chunks = [
            (object(), SimpleNamespace(raw_tensor=object()), 0, 256),
            (object(), SimpleNamespace(raw_tensor=None), 256, 512),
        ]
        assert engine._sender_chunks_broadcastable(chunks) is False


class TestPipelineBroadcastAbortGate:
    def test_sender_aborts_without_broadcast_fn_when_proxy_present(self):
        """delay-pull proxies must not enter the shard collective loop."""
        engine = _engine(is_first=True)
        # Echo the sender's can_run decision (False).
        engine.broadcast_object_fn = Mock(side_effect=lambda obj, _src: obj)

        proxy = SimpleNamespace(raw_tensor=None, ref_count_down=Mock())
        real = SimpleNamespace(raw_tensor=object(), ref_count_down=Mock())
        chunks = [
            (object(), real, 0, 256),
            (object(), proxy, 256, 512),
        ]
        plan = {
            "meta": [],
            "shard_plan": [(0, 2)],
            "shard_layouts": [[(0, 0, 64), (1, 64, 64)]],
            "max_shard_bytes": 128,
        }
        load_stream = SimpleNamespace(synchronize=Mock())

        engine._pipeline_broadcast_and_load(
            plan, load_stream, reordered_chunks=chunks
        )

        engine.broadcast_fn.assert_not_called()
        engine._ensure_merged_pool.assert_not_called()
        # retrieve() skips ref_count_down for the sender; abort path owns it.
        real.ref_count_down.assert_called_once_with()
        proxy.ref_count_down.assert_called_once_with()

    def test_receiver_aborts_when_sender_signals_not_ready(self):
        """Receivers must observe the sender's can_run=False and skip broadcast_fn."""
        engine = _engine(worker_id=1, is_first=False)
        # Non-root passes None; collective returns sender's False.
        engine.broadcast_object_fn = Mock(return_value=False)

        plan = {
            "meta": [],
            "shard_plan": [(0, 1)],
            "shard_layouts": [[(0, 0, 64)]],
            "max_shard_bytes": 64,
        }
        load_stream = SimpleNamespace(synchronize=Mock())
        ret_mask = Mock()

        engine._pipeline_broadcast_and_load(
            plan, load_stream, ret_mask=ret_mask
        )

        engine.broadcast_object_fn.assert_called_once_with(
            None, engine.metadata.first_rank
        )
        engine.broadcast_fn.assert_not_called()
        engine._ensure_merged_pool.assert_not_called()


class TestBroadcastStreamDrainOnFailure:
    def test_finally_synchronizes_broadcast_stream_before_cpu_release(self):
        """Partial H2D must be drained before sender CPU mem_obj release."""
        engine = _engine(is_first=True)
        engine.broadcast_object_fn = Mock(side_effect=lambda obj, _src: True)

        order: list[str] = []
        engine.broadcast_stream.synchronize = Mock(
            side_effect=lambda: order.append("broadcast_sync")
        )
        load_stream = SimpleNamespace(
            synchronize=Mock(side_effect=lambda: order.append("load_sync"))
        )

        mem_a = SimpleNamespace(
            raw_tensor=object(),
            ref_count_down=Mock(side_effect=lambda: order.append("cpu_release")),
        )
        # Second chunk blows up inside fill after the gate has passed.
        mem_b = SimpleNamespace(raw_tensor=object(), ref_count_down=Mock())
        chunks = [
            (object(), mem_a, 0, 256),
            (object(), mem_b, 256, 512),
        ]

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated mid-shard failure")

        engine._fill_shard_sender = _boom
        # Pool alloc succeeds so we enter the try/finally path.
        engine._ensure_merged_pool = Mock(return_value=True)
        engine._merged_pool = [Mock(), Mock()]

        plan = {
            "meta": [(0, 256, {}), (256, 512, {})],
            "shard_plan": [(0, 2)],
            "shard_layouts": [[(0, 0, 64), (1, 64, 64)]],
            "max_shard_bytes": 128,
        }

        with pytest.raises(RuntimeError, match="simulated mid-shard failure"):
            engine._pipeline_broadcast_and_load(
                plan, load_stream, reordered_chunks=chunks
            )

        assert order[0] == "broadcast_sync"
        assert "load_sync" in order
        assert "cpu_release" in order
        assert order.index("broadcast_sync") < order.index("cpu_release")
