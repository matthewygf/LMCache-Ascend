# SPDX-License-Identifier: Apache-2.0
"""Soft-failed P2P loads must not persist unfilled pages into LocalCPU.

vLLM calls wait_for_save before get_block_ids_with_load_errors(), so a same-step
``_local_persist_skip`` after soft-fail would D2H garbage/partial NPU pages and
poison later local hits.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

CHUNK = 16


def _adapter_mod():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    return pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")


def test_mark_failed_retains_ids_until_wait_for_save():
    adapter_mod = _adapter_mod()
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata

    engine = MagicMock()
    engine.gpu_connector = MagicMock()
    engine.gpu_connector.drain_failed_load_req_ids = MagicMock(
        return_value={"req-failed"}
    )

    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.lmcache_engine = engine
    adapter._lmcache_chunk_size = CHUNK
    adapter._invalid_block_ids = set()
    adapter._failed_load_req_ids_pending_save = set()
    adapter.record_failed_blocks = MagicMock(return_value={7, 8})
    adapter._parent = MagicMock()

    req = SimpleNamespace(
        req_id="req-failed",
        token_ids=list(range(4 * CHUNK)),
        slot_mapping=list(range(4 * CHUNK)),
        load_spec=SimpleNamespace(
            can_load=True,
            lmcache_cached_tokens=4 * CHUNK,
            vllm_cached_tokens=0,
        ),
    )
    metadata = MagicMock(spec=LMCacheConnectorMetadata)
    metadata.requests = [req]
    adapter._parent._get_connector_metadata.return_value = metadata

    adapter._mark_failed_p2p_loads_for_recompute()

    assert adapter._failed_load_req_ids_pending_save == {"req-failed"}
    assert adapter._invalid_block_ids == {7, 8}


def test_wait_for_save_skips_local_persist_for_soft_failed_req(monkeypatch):
    adapter_mod = _adapter_mod()
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata
    import torch

    engine = MagicMock()
    engine._is_passive.return_value = False

    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.kv_role = "kv_both"
    adapter.lmcache_engine = engine
    adapter.use_layerwise = False
    adapter.enable_blending = False
    adapter.kv_caches = {"layer0": MagicMock()}
    adapter._lmcache_chunk_size = CHUNK
    adapter._failed_load_req_ids_pending_save = {"req-failed"}
    adapter._finished_req_ids_waiting_for_save = set()
    adapter._late_finished_sending = set()
    adapter._wait_for_save_done = False
    adapter._layerwise_save_storers = {}
    adapter._replay_finished_stores_after_save = MagicMock()
    adapter._local_persist_skip = MagicMock(return_value=0)
    adapter._parent = MagicMock()

    req = SimpleNamespace(
        req_id="req-failed",
        token_ids=list(range(4 * CHUNK)),
        slot_mapping=torch.arange(4 * CHUNK, dtype=torch.long),
        request_configs=None,
        load_spec=SimpleNamespace(
            can_load=True,
            lmcache_cached_tokens=4 * CHUNK,
            vllm_cached_tokens=0,
        ),
        save_spec=SimpleNamespace(
            can_save=False,
            skip_leading_tokens=4 * CHUNK,
        ),
        is_last_prefill=True,
        disagg_spec=None,
    )
    metadata = MagicMock(spec=LMCacheConnectorMetadata)
    metadata.requests = [req]
    adapter._parent._get_connector_metadata.return_value = metadata

    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter.torch.npu.Event",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "lmcache_ascend.integration.vllm.vllm_v1_adapter.torch.npu.stream",
        MagicMock(
            return_value=MagicMock(
                __enter__=lambda s: None,
                __exit__=lambda *a: None,
            )
        ),
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self, raising=False)

    adapter.wait_for_save()

    adapter._local_persist_skip.assert_not_called()
    engine.store.assert_not_called()
    engine.lookup_unpin.assert_called_once_with("req-failed")
    assert adapter._failed_load_req_ids_pending_save == set()
    assert adapter._wait_for_save_done is True
