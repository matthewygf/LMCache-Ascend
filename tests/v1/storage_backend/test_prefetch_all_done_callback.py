# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ascend ``patched_prefetch_all_done_callback``.

Two critical correctness gaps in the Ascend overlay:

1. Hot-cache inject must skip delay-pull ``is_proxy`` placeholders (same as
   ``get`` / ``batched_get``) -- otherwise LocalCPU "hits" later re-resolve
   consumed proxies against already-released remote/host-staging buffers.
2. Tier-gap cleanup must unpack ``(key, mem_obj)`` pairs from
   ``gather_with_keys()`` before calling ``ref_count_down`` -- iterating the
   bare tuple raises AttributeError and aborts before the scheduler response.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache_ascend.v1.storage_backend.storage_manager import (
    patched_prefetch_all_done_callback,
)


def _obj(*, is_proxy: bool = False) -> MagicMock:
    obj = MagicMock()
    obj.is_proxy = is_proxy
    return obj


def _manager(*, use_hot: bool = True):
    local = MagicMock()
    local.use_hot = use_hot
    async_lookup_server = MagicMock()
    event_manager = MagicMock()
    manager = SimpleNamespace(
        async_lookup_server=async_lookup_server,
        event_manager=event_manager,
        local_cpu_backend=local,
    )
    return manager, local, async_lookup_server


def test_prefetch_callback_skips_proxy_hot_cache_write_back():
    """Delay-pull proxies must not be mirrored into LocalCPUBackend."""
    manager, local, async_server = _manager()
    real = _obj(is_proxy=False)
    proxy = _obj(is_proxy=True)
    task = MagicMock()
    # Tier layout matches gather_with_keys: list of (key, mem_obj) pairs.
    task.result.return_value = [
        [("k0", real), ("k1", proxy)],
    ]

    patched_prefetch_all_done_callback(
        manager,
        task,
        lookup_id="req-1",
        cum_chunk_lengths_total=[0, 256, 512],
        tier_expected_chunks=[2],
    )

    local.batched_submit_put_task.assert_called_once_with(["k0"], [real])
    async_server.send_response_to_scheduler.assert_called_once_with("req-1", 512)
    proxy.ref_count_down.assert_not_called()


def test_prefetch_callback_all_proxies_skips_hot_cache_put():
    """If every retrieved chunk is a proxy, do not call batched_submit_put_task."""
    manager, local, async_server = _manager()
    proxy0 = _obj(is_proxy=True)
    proxy1 = _obj(is_proxy=True)
    task = MagicMock()
    task.result.return_value = [[("k0", proxy0), ("k1", proxy1)]]

    patched_prefetch_all_done_callback(
        manager,
        task,
        lookup_id="req-2",
        cum_chunk_lengths_total=[0, 256, 512],
        tier_expected_chunks=[2],
    )

    local.batched_submit_put_task.assert_not_called()
    async_server.send_response_to_scheduler.assert_called_once_with("req-2", 512)


def test_prefetch_callback_tier_gap_releases_subsequent_tuples():
    """A mid-tier shortfall must unpack tuples and release later-tier objs."""
    manager, local, async_server = _manager()
    t0_a = _obj()
    t0_b = _obj()
    t1_a = _obj()  # shortfall: expected 2, got 1
    t2_a = _obj()
    t2_b = _obj()
    task = MagicMock()
    task.result.return_value = [
        [("k0", t0_a), ("k1", t0_b)],
        [("k2", t1_a)],
        [("k3", t2_a), ("k4", t2_b)],
    ]

    patched_prefetch_all_done_callback(
        manager,
        task,
        lookup_id="req-3",
        cum_chunk_lengths_total=[0, 256, 512, 768, 1024, 1280],
        tier_expected_chunks=[2, 2, 2],
    )

    # Prefix stops after tier 1's single chunk: 2 + 1 = 3 chunks → 768 tokens.
    async_server.send_response_to_scheduler.assert_called_once_with("req-3", 768)
    t2_a.ref_count_down.assert_called_once_with()
    t2_b.ref_count_down.assert_called_once_with()
    # Only the contiguous prefix is injected (tiers 0 + partial tier 1).
    local.batched_submit_put_task.assert_any_call(["k0", "k1"], [t0_a, t0_b])
    local.batched_submit_put_task.assert_any_call(["k2"], [t1_a])
