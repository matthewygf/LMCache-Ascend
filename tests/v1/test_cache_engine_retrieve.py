# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ascend LMCache retrieval."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


class _RetrieveStats:
    def profile_process_tokens(self):
        return nullcontext()

    def profile_to_gpu(self):
        return nullcontext()

    def time_to_retrieve(self):
        return 0.001


def _make_retrieve_engine(*, remove_after_retrieve: bool, enable_pd: bool):
    engine = object.__new__(AscendLMCacheEngine)
    stats = _RetrieveStats()
    memory_obj_0 = SimpleNamespace(ref_count_down=Mock())
    memory_obj_1 = SimpleNamespace(ref_count_down=Mock())
    chunks = [
        ("key-0", memory_obj_0, 0, 256),
        ("key-1", memory_obj_1, 256, 512),
    ]

    engine.gpu_connector = SimpleNamespace(batched_to_gpu=Mock())
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda _kwargs: "test-request"
    engine._log_kvcache_for_check = lambda **_kwargs: None
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _num_tokens: stats,
        on_retrieve_finished=Mock(),
    )
    engine._is_passive = lambda: False
    engine.async_loading = False
    engine.save_only_first_rank = False
    engine.remove_after_retrieve = remove_after_retrieve
    engine.retrieve_locations = ["PDBackend"]
    engine.storage_manager = SimpleNamespace(remove=Mock())
    engine.config = SimpleNamespace(enable_pd=enable_pd)
    # Mimic lmcache 0.4.4: parent helper is absent; Ascend must supply it.
    assert hasattr(AscendLMCacheEngine, "_is_sync_pd_backend")

    def process_tokens(tokens, mask, ret_mask, **kwargs):
        ret_mask[:] = True
        return chunks, 1024

    engine._process_tokens_internal = process_tokens
    return engine, memory_obj_0, memory_obj_1, chunks


def test_retrieve_loads_cache_to_gpu_in_normal_mode():
    """A cache hit must copy the retrieved CPU KV into the serving KV cache."""
    engine, memory_obj_0, memory_obj_1, _chunks = _make_retrieve_engine(
        remove_after_retrieve=False,
        enable_pd=False,
    )

    slot_mapping = object()
    result = engine.retrieve(
        list(range(512)),
        slot_mapping=slot_mapping,
    )

    engine.gpu_connector.batched_to_gpu.assert_called_once_with(
        [memory_obj_0, memory_obj_1],
        [0, 256],
        [256, 512],
        slot_mapping=slot_mapping,
    )
    assert torch.all(result)
    memory_obj_0.ref_count_down.assert_called_once_with()
    memory_obj_1.ref_count_down.assert_called_once_with()


def test_is_sync_pd_backend_true_when_pd_enabled():
    """Ascend PD remove never decrefs; retrieve cleanup must treat PD as sync."""
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(enable_pd=True)
    assert engine._is_sync_pd_backend() is True
    engine.config = SimpleNamespace(enable_pd=False)
    assert engine._is_sync_pd_backend() is False


def test_retrieve_pd_receiver_remove_after_retrieve_decrefs_without_attrerror():
    """1p1d decoder (remove_after_retrieve) must not AttributeError on 0.4.4.

    Before the Ascend shim, retrieve cleanup called missing
    ``LMCacheEngine._is_sync_pd_backend`` after ``storage_manager.remove``,
    crashing every PD decoder retrieve on the supported lmcache tag.
    """
    engine, memory_obj_0, memory_obj_1, chunks = _make_retrieve_engine(
        remove_after_retrieve=True,
        enable_pd=True,
    )

    result = engine.retrieve(list(range(512)), slot_mapping=object())

    assert torch.all(result)
    assert engine.storage_manager.remove.call_count == 2
    engine.storage_manager.remove.assert_any_call("key-0", ["PDBackend"])
    engine.storage_manager.remove.assert_any_call("key-1", ["PDBackend"])
    memory_obj_0.ref_count_down.assert_called_once_with()
    memory_obj_1.ref_count_down.assert_called_once_with()
    # Sanity: chunks were the ones retrieved
    assert [c[0] for c in chunks] == ["key-0", "key-1"]
