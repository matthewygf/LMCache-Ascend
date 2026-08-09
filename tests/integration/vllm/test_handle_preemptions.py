# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Ascend ``handle_preemptions`` across vLLM API shapes.

vLLM ≤0.18 passes ``set[str]`` request ids. vLLM ≥0.23 always passes
``KVConnectorMetadata`` (with ``preempted_req_ids`` attached by Ascend's
``build_connector_meta``). Both shapes must drain async stores / unpin
lookups without raising.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest


def _import_and_patch_vllm_connector():
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")

    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1,
    )

    lmcache_ascend = pytest.importorskip("lmcache_ascend")
    lmcache_ascend._patch_vllm_v1_adapter()
    return LMCacheConnectorV1


def _make_adapter(adapter_mod, *, store_async, kv_role, lmcache_engine):
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)
    adapter.store_async = store_async
    adapter.kv_role = kv_role
    adapter._manager = SimpleNamespace(lmcache_engine=lmcache_engine)
    return adapter


def test_lmcache_connector_delegates_preemptions_after_ascend_patch():
    """Ascend patches the outer vLLM connector to delegate preemptions."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    preempted_req_ids = {"req-1", "req-2"}
    connector.handle_preemptions(preempted_req_ids)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(
        preempted_req_ids
    )


def test_lmcache_connector_preemption_patch_handles_no_inner_impl():
    """The Ascend patch should tolerate inner implementations without a hook."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = object()

    connector.handle_preemptions({"req-1"})


def test_lmcache_connector_preemption_patch_accepts_metadata():
    """vLLM ≥0.23 passes connector metadata, not a bare set."""
    LMCacheConnectorV1 = _import_and_patch_vllm_connector()

    connector = object.__new__(LMCacheConnectorV1)
    connector._lmcache_engine = MagicMock()

    metadata = SimpleNamespace(preempted_req_ids={"req-meta"})
    connector.handle_preemptions(metadata)

    connector._lmcache_engine.handle_preemptions.assert_called_once_with(metadata)


def test_ascend_adapter_drains_pending_stores_for_async_producer():
    """Async non-consumer workers must drain pending stores before reuse."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = {"req-1"}
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    preempted_req_ids = {"req-1", "req-2"}
    adapter.handle_preemptions(preempted_req_ids)

    lmcache_engine.wait_for_pending_stores.assert_called_once_with(preempted_req_ids)


def test_ascend_adapter_drains_from_metadata_preempted_req_ids():
    """Metadata-shaped calls (vLLM ≥0.23) must extract and drain preempted ids."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    lmcache_engine.wait_for_pending_stores.return_value = {"req-9"}
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    metadata = SimpleNamespace(preempted_req_ids={"req-9"})
    adapter.handle_preemptions(metadata)

    lmcache_engine.lookup_unpin.assert_called_once_with("req-9")
    lmcache_engine.wait_for_pending_stores.assert_called_once_with({"req-9"})


def test_ascend_adapter_metadata_without_preempted_ids_is_noop():
    """Empty / missing preempted ids must not raise or unpin."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock()
    adapter = _make_adapter(
        adapter_mod,
        store_async=True,
        kv_role="kv_both",
        lmcache_engine=lmcache_engine,
    )

    # Upstream LMCache metadata historically had no preempted_req_ids attr.
    adapter.handle_preemptions(SimpleNamespace(requests=[]))

    lmcache_engine.lookup_unpin.assert_not_called()
    lmcache_engine.wait_for_pending_stores.assert_not_called()


def test_extract_preempted_req_ids_shapes():
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")
    extract = adapter_mod._extract_preempted_req_ids

    assert extract({"a", "b"}) == {"a", "b"}
    assert extract(["a"]) == {"a"}
    assert extract(SimpleNamespace(preempted_req_ids={"z"})) == {"z"}
    assert extract(SimpleNamespace(requests=[])) == set()


def test_build_connector_meta_attaches_preempted_req_ids(monkeypatch):
    """Ascend build_connector_meta must copy scheduler preempted ids onto meta."""
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    base_meta = SimpleNamespace()
    adapter = object.__new__(adapter_mod.LMCacheAscendConnectorV1Impl)

    def _fake_super_build(_self, _scheduler_output):
        return base_meta

    monkeypatch.setattr(
        adapter_mod.LMCacheConnectorV1Impl,
        "build_connector_meta",
        _fake_super_build,
    )

    scheduler_output = SimpleNamespace(preempted_req_ids={"p1", "p2"})
    meta = adapter.build_connector_meta(scheduler_output)

    assert meta is base_meta
    assert meta.preempted_req_ids == {"p1", "p2"}


@pytest.mark.parametrize(
    ("store_async", "kv_role", "has_engine"),
    [
        (False, "kv_both", True),
        (True, "kv_consumer", True),
        (True, "kv_both", False),
    ],
)
def test_ascend_adapter_skips_preemption_drain_when_not_required(
    store_async, kv_role, has_engine
):
    pytest.importorskip("lmcache")
    pytest.importorskip("vllm")
    adapter_mod = pytest.importorskip("lmcache_ascend.integration.vllm.vllm_v1_adapter")

    lmcache_engine = MagicMock() if has_engine else None
    adapter = _make_adapter(
        adapter_mod,
        store_async=store_async,
        kv_role=kv_role,
        lmcache_engine=lmcache_engine,
    )

    adapter.handle_preemptions({"req-1"})

    if has_engine:
        lmcache_engine.wait_for_pending_stores.assert_not_called()
