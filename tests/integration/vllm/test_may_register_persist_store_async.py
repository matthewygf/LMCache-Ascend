# SPDX-License-Identifier: Apache-2.0
"""store_async + MTP must delay GPU free for Ascend persist-only saves.

``_may_register_store_after_wait_for_save`` previously returned False whenever
``save_spec.can_save`` was False. Ascend ``wait_for_save`` can still D2H via
``_local_persist_skip`` in that case (disk/remote hit, LocalCPU cold, first
step below the next chunk boundary because RequestTracker seeds
``num_saved_tokens`` from LMCache hits). With ``store_async`` and MTP,
``get_finished`` then frees paged KV before that D2H — silent wrong KV in
LocalCPU. GLM recommended configs enable both ``store_async`` and deepseek_mtp.
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


def _make_adapter(*, kv_role="kv_both", local_cpu=True):
    adapter_mod = _adapter_mod()
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.kv_role = kv_role
    adapter.config = SimpleNamespace(local_cpu=local_cpu)
    adapter._lmcache_chunk_size = CHUNK
    return adapter


def _make_request(
    *,
    can_save=False,
    skip_leading_tokens=4 * CHUNK,
    can_load=True,
    lmcache_cached_tokens=4 * CHUNK,
    num_tokens=5 * CHUNK,
):
    return SimpleNamespace(
        req_id="req-persist",
        token_ids=list(range(num_tokens)),
        save_spec=SimpleNamespace(
            can_save=can_save,
            skip_leading_tokens=skip_leading_tokens,
        ),
        load_spec=SimpleNamespace(
            can_load=can_load,
            lmcache_cached_tokens=lmcache_cached_tokens,
        ),
    )


def test_persist_only_can_save_false_registers_when_local_cpu():
    """can_save=False + can_load must still delay free under store_async."""
    adapter = _make_adapter(local_cpu=True)
    request = _make_request(can_save=False, can_load=True)

    assert adapter._may_register_store_after_wait_for_save(request) is True


def test_persist_only_does_not_register_without_local_cpu():
    adapter = _make_adapter(local_cpu=False)
    request = _make_request(can_save=False, can_load=True)

    assert adapter._may_register_store_after_wait_for_save(request) is False


def test_persist_only_does_not_register_without_load():
    adapter = _make_adapter(local_cpu=True)
    request = _make_request(can_save=False, can_load=False)

    assert adapter._may_register_store_after_wait_for_save(request) is False


def test_consumer_never_registers():
    adapter = _make_adapter(kv_role="kv_consumer", local_cpu=True)
    request = _make_request(can_save=False, can_load=True)

    assert adapter._may_register_store_after_wait_for_save(request) is False


def test_normal_can_save_still_registers():
    adapter = _make_adapter(local_cpu=True)
    request = _make_request(
        can_save=True,
        skip_leading_tokens=0,
        can_load=False,
        lmcache_cached_tokens=0,
        num_tokens=4 * CHUNK,
    )

    assert adapter._may_register_store_after_wait_for_save(request) is True


def test_get_finished_holds_persist_only_req_before_wait_for_save():
    """MTP-style get_finished before wait_for_save must not free persist-only reqs."""
    adapter_mod = _adapter_mod()
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata

    adapter = _make_adapter(local_cpu=True)
    adapter.store_async = True
    adapter._wait_for_save_done = False
    adapter._finished_req_ids_waiting_for_save = set()
    adapter._late_finished_sending = set()
    adapter.lmcache_engine = MagicMock()
    adapter.lmcache_engine.get_finished_stores.return_value = set()

    request = _make_request(can_save=False, can_load=True)
    metadata = MagicMock(spec=LMCacheConnectorMetadata)
    metadata.requests = [request]
    adapter._parent = MagicMock()
    adapter._parent._get_connector_metadata.return_value = metadata

    finished_sending, _ = adapter.get_finished({request.req_id})

    assert finished_sending is None
    assert request.req_id in adapter._finished_req_ids_waiting_for_save
    adapter.lmcache_engine.get_finished_stores.assert_called_once()
    queried = adapter.lmcache_engine.get_finished_stores.call_args[0][0]
    assert request.req_id not in queried
