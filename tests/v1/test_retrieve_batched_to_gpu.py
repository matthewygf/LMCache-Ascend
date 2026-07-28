# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for AscendLMCacheEngine.retrieve GPU load path.

PR #257 overrode retrieve for sharded broadcast but omitted the
non-save_only_first_rank branch that must call batched_to_gpu.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch


def _make_retrieve_engine(
    *,
    save_only_first_rank: bool,
    async_loading: bool = False,
    remove_after_retrieve: bool = False,
):
    # First Party
    from lmcache_ascend.v1 import cache_engine as ce_mod

    engine = object.__new__(ce_mod.AscendLMCacheEngine)
    engine.save_only_first_rank = save_only_first_rank
    engine.async_loading = async_loading
    engine.remove_after_retrieve = remove_after_retrieve
    engine.retrieve_locations = None
    engine.storage_manager = None

    engine.gpu_connector = MagicMock()
    engine.metadata = SimpleNamespace(
        is_first_rank=lambda: True,
        worker_id=0,
    )

    # Health / logging / stats helpers used by retrieve()
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda kwargs: "test-req"
    engine._log_kvcache_for_check = lambda **kwargs: None
    engine._is_passive = lambda: False
    engine._is_sync_pd_backend = lambda: False

    retrieve_stats = MagicMock()
    retrieve_stats.profile_process_tokens.return_value.__enter__ = lambda s: None
    retrieve_stats.profile_process_tokens.return_value.__exit__ = (
        lambda s, *a: None
    )
    retrieve_stats.profile_broadcast.return_value.__enter__ = lambda s: None
    retrieve_stats.profile_broadcast.return_value.__exit__ = lambda s, *a: None
    retrieve_stats.profile_to_gpu.return_value.__enter__ = lambda s: None
    retrieve_stats.profile_to_gpu.return_value.__exit__ = lambda s, *a: None
    retrieve_stats.time_to_retrieve.return_value = 0.001
    engine.stats_monitor = MagicMock()
    engine.stats_monitor.on_retrieve_request.return_value = retrieve_stats

    mem_obj = MagicMock()
    key = MagicMock()
    chunks = [(key, mem_obj, 0, 16)]
    engine._process_tokens_internal = MagicMock(return_value=(chunks, 1024))
    engine._async_process_tokens_internal = MagicMock(return_value=(chunks, 1024))
    engine._pipelined_sharded_broadcast_and_load = MagicMock()

    return engine, mem_obj, retrieve_stats


class TestRetrieveBatchedToGpu:
    def test_calls_batched_to_gpu_when_not_save_only_first_rank(self):
        """Non-MLA path must load retrieved objects onto the NPU."""
        engine, mem_obj, _ = _make_retrieve_engine(save_only_first_rank=False)
        tokens = torch.arange(16, dtype=torch.long)

        ret_mask = engine.retrieve(tokens)

        engine.gpu_connector.batched_to_gpu.assert_called_once()
        args, _kwargs = engine.gpu_connector.batched_to_gpu.call_args
        assert args[0] == [mem_obj]
        assert args[1] == [0]
        assert args[2] == [16]
        engine._pipelined_sharded_broadcast_and_load.assert_not_called()
        mem_obj.ref_count_down.assert_called_once()
        assert bool(ret_mask.shape[0] == 16)

    def test_uses_sharded_pipeline_when_save_only_first_rank(self):
        """MLA/save_only_first_rank path keeps the sharded broadcast pipeline."""
        engine, mem_obj, _ = _make_retrieve_engine(save_only_first_rank=True)
        tokens = torch.arange(16, dtype=torch.long)

        engine.retrieve(tokens)

        engine._pipelined_sharded_broadcast_and_load.assert_called_once()
        engine.gpu_connector.batched_to_gpu.assert_not_called()
        # Sender ownership moves into the pipeline's finally block.
        mem_obj.ref_count_down.assert_not_called()

    def test_skips_batched_to_gpu_when_no_chunks(self):
        """Miss path must not call batched_to_gpu with empty chunks."""
        engine, _mem_obj, _ = _make_retrieve_engine(save_only_first_rank=False)
        engine._process_tokens_internal = MagicMock(return_value=([], 0))
        tokens = torch.arange(8, dtype=torch.long)

        engine.retrieve(tokens)

        engine.gpu_connector.batched_to_gpu.assert_not_called()
        engine._pipelined_sharded_broadcast_and_load.assert_not_called()
