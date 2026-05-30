# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Tests for HcclOneSidedChannel hardening logic.

Two layers live in this file:

1. Device-free unit tests (run anywhere the extension imports):
     * equal send/recv slot enforcement (and rejection of legacy unequal knobs),
     * the teardown helpers in isolation (guarded failures, idempotency),
     * that a real read failure (``ok=False`` ack) is WIRED to teardown through
       ``batched_read`` and drops the connector peer,
     * that the happy path does NOT tear the peer down,
     * RECONNECT: after teardown drops the peer, ``lazy_init_peer_connection``
       re-handshakes and re-registers a fresh peer/connection.
   Device work (transport, NotifyPool, RDMA, ZMQ I/O) is mocked here.

2. Hardware (2-NPU) integration tests (``@pytest.mark.hardware``, skipped
   without two NPUs), driving the real data plane end to end (real
   agent.batch_write / WriteAsync + Post/Wait):
     * a clean receiver-driven pull + verify,
     * long/ping-pong slot-reuse stress: many slices (num_objs >> recv_slots)
       and repeated pulls on one connection, forcing sustained staging reuse,
     * sender-side failure -> ok=False -> both ends tear down -> reconnect +
       correct retry,
     * receiver-side copy-out failure -> only the receiver tears down, the
       sender's stale listener conn is closed on the re-handshake overwrite ->
       reconnect + correct retry.
   Override device ids with OS_TEST_SENDER_DEV / OS_TEST_RECEIVER_DEV.
"""

# Standard
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock
import asyncio
import faulthandler
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
import warnings

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, PagedCpuGpuMemoryAllocator
import msgspec
import pytest
import torch
import torch_npu  # noqa: F401

pytest.importorskip("lmcache_ascend.hccl_onesided_npu_comms")

# First Party
from lmcache_ascend.v1.transfer_channel import CreateTransferChannel  # noqa: E402
from lmcache_ascend.v1.transfer_channel.buffer_config import BufferType  # noqa: E402
from lmcache_ascend.v1.transfer_channel.hccl_onesided_channel import (  # noqa: E402
    HcclOneSidedChannel,
    _PeerState,
    _RecvInflightDrainError,
)
from lmcache_ascend.v1.transfer_channel.hccl_onesided_protocol import (  # noqa: E402
    OneSidedInitResponse,
    OneSidedReadAck,
    OneSidedReadRequest,
)
from lmcache_ascend.v1.transfer_channel.transfer_spec import (  # noqa: E402
    TS_RECEIVER_ID,
    TS_REMOTE_BUFFER_UUIDS,
    TS_REMOTE_MEM_INDEXES,
    TS_STREAM,
)
import lmcache_ascend.v1.transfer_channel.hccl_onesided_channel as oschan  # noqa: E402

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Item 1: equal send_slots == recv_slots enforcement
# ---------------------------------------------------------------------------
class TestDeriveEqualSlots:
    def test_derives_half_of_num_slots(self):
        assert HcclOneSidedChannel._derive_equal_slots(4, {}) == 2
        assert HcclOneSidedChannel._derive_equal_slots(8, {}) == 4
        # Odd num_slots floors to the per-direction count.
        assert HcclOneSidedChannel._derive_equal_slots(5, {}) == 2

    def test_rejects_too_few_slots(self):
        with pytest.raises(ValueError):
            HcclOneSidedChannel._derive_equal_slots(1, {})
        with pytest.raises(ValueError):
            HcclOneSidedChannel._derive_equal_slots(0, {})

    def test_rejects_conflicting_recv_slots(self):
        with pytest.raises(ValueError):
            HcclOneSidedChannel._derive_equal_slots(4, {"os_recv_slots": 3})

    def test_rejects_conflicting_send_slots(self):
        with pytest.raises(ValueError):
            HcclOneSidedChannel._derive_equal_slots(4, {"os_send_slots": 1})

    def test_accepts_legacy_knobs_when_they_match(self):
        # Explicitly passing the pinned value is allowed (no surprise raise).
        assert (
            HcclOneSidedChannel._derive_equal_slots(
                4, {"os_recv_slots": 2, "os_send_slots": 2}
            )
            == 2
        )


# ---------------------------------------------------------------------------
# Test helpers for the teardown unit tests
# ---------------------------------------------------------------------------
def _make_peer(conn_handle: int = 0xABCD) -> _PeerState:
    return _PeerState(
        conn_handle=conn_handle,
        transfer_url="tcp://127.0.0.1:0",
        transfer_req_socket=MagicMock(name="transfer_req_socket"),
        remote_buffers=MagicMock(name="remote_buffers"),
        remote_slot_bytes=256,
        remote_recv_slots=2,
        remote_send_slots=2,
    )


def _bare_channel() -> HcclOneSidedChannel:
    """Build a channel instance without running __init__ (no device needed).

    Only the attributes touched by the teardown helpers are populated.
    """
    ch = object.__new__(HcclOneSidedChannel)
    ch._state_lock = threading.Lock()
    ch._connector_peers = {}
    ch._listener_peers = {}
    ch.agent = MagicMock(name="agent")
    ch._send_stream = MagicMock(name="send_stream")
    ch._send_inflight_event = MagicMock(name="send_inflight_event")
    return ch


# ---------------------------------------------------------------------------
# Item 2a: receiver-side teardown (_teardown_connector_peer)
# ---------------------------------------------------------------------------
class TestTeardownConnectorPeer:
    def test_drops_peer_closes_socket_and_conn(self):
        ch = _bare_channel()
        peer = _make_peer(conn_handle=0x1111)
        sock = peer.transfer_req_socket
        ch._connector_peers["peerA"] = peer
        stream = MagicMock(name="stream")

        ch._teardown_connector_peer("peerA", peer, stream)

        stream.synchronize.assert_called_once_with()
        ch.agent.close_connection.assert_called_once_with(0x1111)
        sock.close.assert_called_once_with(linger=0)
        assert peer.transfer_req_socket is None
        assert "peerA" not in ch._connector_peers

    def test_synchronize_failure_is_guarded(self):
        # The originating failure may itself be a device/synchronize error;
        # close_connection, socket close, and peer drop must still run.
        ch = _bare_channel()
        peer = _make_peer(conn_handle=0x2222)
        sock = peer.transfer_req_socket
        ch._connector_peers["peerB"] = peer
        stream = MagicMock(name="stream")
        stream.synchronize.side_effect = RuntimeError("device wait failed")

        ch._teardown_connector_peer("peerB", peer, stream)

        ch.agent.close_connection.assert_called_once_with(0x2222)
        sock.close.assert_called_once_with(linger=0)
        assert "peerB" not in ch._connector_peers

    def test_close_connection_failure_is_guarded(self):
        ch = _bare_channel()
        peer = _make_peer()
        sock = peer.transfer_req_socket
        ch._connector_peers["peerC"] = peer
        ch.agent.close_connection.side_effect = RuntimeError("hccl error")
        stream = MagicMock(name="stream")

        # Must not raise; socket close + peer drop still happen.
        ch._teardown_connector_peer("peerC", peer, stream)

        sock.close.assert_called_once_with(linger=0)
        assert "peerC" not in ch._connector_peers

    def test_does_not_drop_a_different_peer_object(self):
        # Idempotency / race safety: if the registry already holds a fresh peer
        # (e.g. a concurrent re-handshake), teardown of the stale peer must not
        # evict the replacement.
        ch = _bare_channel()
        stale = _make_peer()
        fresh = _make_peer()
        ch._connector_peers["peerD"] = fresh
        stream = MagicMock(name="stream")

        ch._teardown_connector_peer("peerD", stale, stream)

        assert ch._connector_peers["peerD"] is fresh

    def test_handles_none_socket(self):
        ch = _bare_channel()
        peer = _make_peer()
        peer.transfer_req_socket = None
        ch._connector_peers["peerE"] = peer
        stream = MagicMock(name="stream")

        ch._teardown_connector_peer("peerE", peer, stream)

        assert "peerE" not in ch._connector_peers

    def test_handles_none_stream(self):
        ch = _bare_channel()
        peer = _make_peer(conn_handle=0x3333)
        ch._connector_peers["peerF"] = peer

        ch._teardown_connector_peer("peerF", peer, None)

        ch.agent.close_connection.assert_called_once_with(0x3333)
        assert "peerF" not in ch._connector_peers


# ---------------------------------------------------------------------------
# Item 2a: sender-side teardown (_teardown_listener_peer)
# ---------------------------------------------------------------------------
class TestTeardownListenerPeer:
    def test_drops_peer_resets_event_and_closes_conn(self):
        ch = _bare_channel()
        peer = _make_peer(conn_handle=0x4444)
        ch._listener_peers["recvX"] = peer

        ch._teardown_listener_peer("recvX", peer)

        ch._send_stream.synchronize.assert_called_once_with()
        ch.agent.close_connection.assert_called_once_with(0x4444)
        assert ch._send_inflight_event is None
        assert "recvX" not in ch._listener_peers

    def test_synchronize_failure_is_guarded(self):
        ch = _bare_channel()
        peer = _make_peer(conn_handle=0x5555)
        ch._listener_peers["recvY"] = peer
        ch._send_stream.synchronize.side_effect = RuntimeError("drain failed")

        ch._teardown_listener_peer("recvY", peer)

        ch.agent.close_connection.assert_called_once_with(0x5555)
        assert ch._send_inflight_event is None
        assert "recvY" not in ch._listener_peers

    def test_does_not_drop_a_different_peer_object(self):
        ch = _bare_channel()
        stale = _make_peer()
        fresh = _make_peer()
        ch._listener_peers["recvZ"] = fresh

        ch._teardown_listener_peer("recvZ", stale)

        assert ch._listener_peers["recvZ"] is fresh


# ---------------------------------------------------------------------------
# Item 2a wiring: a real read failure must TRIGGER teardown (not just the
# helper in isolation), and the happy path must NOT tear down.
# ---------------------------------------------------------------------------
def _read_channel() -> HcclOneSidedChannel:
    """A channel wired enough to drive batched_read() with mocked I/O."""
    ch = _bare_channel()
    ch._recv_staging_lock = threading.Lock()
    ch._recv_inflight_event = None
    ch._recv_inflight_peer_id = None
    ch._recv_inflight_stream = None
    ch.transport_stream = MagicMock(name="transport_stream")
    ch._request_seq = 0
    ch._ack_timeout_sec = 5.0
    ch._slot_bytes = 256
    ch._recv_slots = 2
    ch._send_slots = 2
    ch.local_id = "recv1"
    ch.handle_device = 0
    ch.event_loop = None
    return ch


def _wire_read_plan(ch: HcclOneSidedChannel, peer: _PeerState) -> None:
    # Bypass address resolution / device copies; we only care about the
    # ack -> teardown control flow.
    ch._build_read_plan = MagicMock(
        return_value=(peer, [0], [256], [0], [BufferType.NPU])
    )
    ch._infer_receiver_id_for_request = MagicMock(return_value="recv1")
    ch._copy_staging_to_dsts = MagicMock(name="_copy_staging_to_dsts")


class _FakeEvent:
    def __init__(self):
        self.record = MagicMock(name="record")
        self.synchronize = MagicMock(name="synchronize")


def _sender_channel(send_slots: int = 2, recv_slots: int = 2) -> HcclOneSidedChannel:
    """A channel wired enough to drive ``_handle_read_request`` (sender side).

    Mirrors the slot bookkeeping set up in ``__init__`` (``_send_slot_base ==
    send_slots``) but mocks every device touch (agent ops, stage-in copy,
    stream/event), so the per-object send loop can be exercised host-only.
    """
    ch = object.__new__(HcclOneSidedChannel)
    ch._state_lock = threading.Lock()
    ch._listener_peers = {}
    ch.agent = MagicMock(name="agent")
    ch.agent.get_staging_base.return_value = 0
    ch._send_staging_lock = threading.Lock()
    ch._send_inflight_event = None
    ch._send_stream = MagicMock(name="send_stream")
    ch._send_stream.npu_stream = 0
    ch._slot_bytes = 256
    ch._send_slots = send_slots
    ch._send_slot_base = send_slots
    ch._recv_slots = recv_slots
    ch._copy_src_to_staging = MagicMock(name="_copy_src_to_staging")
    return ch


def _register_listener_peer(
    ch: HcclOneSidedChannel,
    peer_id: str = "recv1",
    remote_recv_slots: int = 2,
) -> _PeerState:
    peer = _make_peer()
    peer.remote_recv_slots = remote_recv_slots
    peer.remote_slot_bytes = ch._slot_bytes
    peer.remote_dirty_slots = set()
    peer.ready_event.set()
    ch._listener_peers[peer_id] = peer
    return peer


class TestReadFailureTriggersTeardown:
    def test_ok_false_ack_drops_peer_via_batched_read(self):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1111)
        sock = peer.transfer_req_socket
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(
                OneSidedReadAck(ok=False, error="remote boom")
            )
        )

        with pytest.raises(RuntimeError):
            ch.batched_read([object()], {TS_RECEIVER_ID: "recv1"})

        # Teardown actually fired through the except wiring:
        assert "recv1" not in ch._connector_peers
        ch.agent.close_connection.assert_called_once_with(0x1111)
        sock.close.assert_called_once_with(linger=0)
        assert peer.transfer_req_socket is None
        # ok=False raises before the staging copy is ever attempted.
        ch._copy_staging_to_dsts.assert_not_called()

    def test_send_failure_drops_peer_via_batched_read(self):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1212)
        peer.transfer_req_socket.send.side_effect = RuntimeError("zmq EFSM")
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(name="_wait_for_transfer_ack")

        with pytest.raises(RuntimeError):
            ch.batched_read([object()], {TS_RECEIVER_ID: "recv1"})

        assert "recv1" not in ch._connector_peers
        ch.agent.close_connection.assert_called_once_with(0x1212)
        # A send that never recv'd must not be followed by an ack wait.
        ch._wait_for_transfer_ack.assert_not_called()

    def test_happy_path_keeps_peer(self):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1313)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=True))
        )

        n = ch.batched_read([object()], {TS_RECEIVER_ID: "recv1"})

        assert n == 1
        assert ch._connector_peers["recv1"] is peer
        ch.agent.close_connection.assert_not_called()
        ch._copy_staging_to_dsts.assert_called_once()
        ch.transport_stream.synchronize.assert_called_once_with()


class TestSubmitBatchedReadOverlap:
    def test_submit_returns_unsynchronized_event(self, monkeypatch):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1414)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=True))
        )
        event = _FakeEvent()
        monkeypatch.setattr(oschan.torch.npu, "Event", MagicMock(return_value=event))

        ret = ch.submit_batched_read([object()], {TS_RECEIVER_ID: "recv1"})

        assert ret is event
        event.record.assert_called_once_with(ch.transport_stream)
        event.synchronize.assert_not_called()
        assert ch._recv_inflight_event is event
        assert ch._recv_inflight_peer_id == "recv1"
        assert ch._recv_inflight_stream is ch.transport_stream

    def test_same_peer_submit_chains_previous_event_without_host_sync(
        self, monkeypatch
    ):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1515)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        previous_event = _FakeEvent()
        ch._recv_inflight_event = previous_event
        ch._recv_inflight_peer_id = "recv1"
        ch._recv_inflight_stream = ch.transport_stream
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=True))
        )
        new_event = _FakeEvent()
        monkeypatch.setattr(
            oschan.torch.npu, "Event", MagicMock(return_value=new_event)
        )

        ret = ch.submit_batched_read([object()], {TS_RECEIVER_ID: "recv1"})

        assert ret is new_event
        ch.transport_stream.wait_event.assert_called_once_with(previous_event)
        previous_event.synchronize.assert_not_called()
        new_event.synchronize.assert_not_called()
        assert ch._recv_inflight_event is new_event
        assert ch._recv_inflight_peer_id == "recv1"

    def test_different_peer_submit_drains_previous_event_before_send(self, monkeypatch):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1616)
        ch._connector_peers["peerB"] = peer
        _wire_read_plan(ch, peer)
        previous_event = _FakeEvent()
        ch._recv_inflight_event = previous_event
        ch._recv_inflight_peer_id = "peerA"
        ch._recv_inflight_stream = ch.transport_stream
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=True))
        )
        new_event = _FakeEvent()
        monkeypatch.setattr(
            oschan.torch.npu, "Event", MagicMock(return_value=new_event)
        )
        order = []
        previous_event.synchronize.side_effect = lambda: order.append("drain")
        peer.transfer_req_socket.send.side_effect = lambda _msg: order.append("send")

        ret = ch.submit_batched_read([object()], {TS_RECEIVER_ID: "peerB"})

        assert ret is new_event
        assert order == ["drain", "send"]
        ch.transport_stream.wait_event.assert_not_called()
        assert ch._recv_inflight_peer_id == "peerB"

    def test_different_peer_drain_failure_tears_down_previous_peer_only(self):
        ch = _read_channel()
        previous_peer = _make_peer(conn_handle=0x1717)
        next_peer = _make_peer(conn_handle=0x1818)
        previous_sock = MagicMock(name="previous_transfer_req_socket")
        previous_peer.transfer_req_socket = previous_sock
        ch._connector_peers["peerA"] = previous_peer
        ch._connector_peers["peerB"] = next_peer
        _wire_read_plan(ch, next_peer)
        previous_event = _FakeEvent()
        previous_event.synchronize.side_effect = RuntimeError("device wait failed")
        ch._recv_inflight_event = previous_event
        ch._recv_inflight_peer_id = "peerA"
        ch._recv_inflight_stream = ch.transport_stream

        with pytest.raises(_RecvInflightDrainError):
            ch.submit_batched_read([object()], {TS_RECEIVER_ID: "peerB"})

        previous_sock.close.assert_called_once_with(linger=0)
        assert previous_peer.transfer_req_socket is None
        assert "peerA" not in ch._connector_peers
        assert "peerB" in ch._connector_peers
        next_peer.transfer_req_socket.send.assert_not_called()


# ---------------------------------------------------------------------------
# submit_batched_read guards: it is a blocking call and the overlap contract
# pins its copy-out to the channel transport_stream.
# ---------------------------------------------------------------------------
class TestSubmitBatchedReadGuards:
    def test_rejects_call_on_channel_event_loop(self):
        # A blocking entrypoint run on the loop that also drives init/recovery
        # would wedge those coroutines, so it must raise before doing any work.
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1919)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        loop = asyncio.new_event_loop()
        ch.event_loop = loop

        async def _call():
            return ch.submit_batched_read([object()], {TS_RECEIVER_ID: "recv1"})

        try:
            with pytest.raises(RuntimeError):
                loop.run_until_complete(_call())
        finally:
            loop.close()
        # It bailed before touching the read plan / wire.
        peer.transfer_req_socket.send.assert_not_called()

    def test_allows_call_off_event_loop(self):
        # No running loop -> the guard is a no-op even when event_loop is set.
        ch = _read_channel()
        ch.event_loop = asyncio.new_event_loop()
        try:
            ch._assert_not_on_event_loop("submit_batched_read")
        finally:
            ch.event_loop.close()

    def test_rejects_foreign_ts_stream(self, monkeypatch):
        # A caller-provided TS_STREAM would split the copy-out enqueue stream
        # from the recorded event and break the connector's pool-reuse fence.
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x1A1A)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        event = _FakeEvent()
        monkeypatch.setattr(oschan.torch.npu, "Event", MagicMock(return_value=event))

        with pytest.raises(ValueError):
            ch.submit_batched_read(
                [object()],
                {TS_RECEIVER_ID: "recv1", TS_STREAM: 0xDEADBEEF},
            )

        # Rejected before any wire I/O or event record.
        peer.transfer_req_socket.send.assert_not_called()
        event.record.assert_not_called()


# ---------------------------------------------------------------------------
# async_batched_read must publish/retire the in-flight copy-out the same way
# submit_batched_read does, so a concurrent read to a DIFFERENT peer host-drains
# it (under the staging lock) before its sender can overwrite shared staging.
# ---------------------------------------------------------------------------
class TestAsyncBatchedReadInflight:
    def test_publishes_inflight_before_offlock_sync_and_retires_after(
        self, monkeypatch
    ):
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x2020)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=True))
        )
        monkeypatch.setattr(oschan.torch.npu, "set_device", MagicMock())

        seen = {}
        event = _FakeEvent()

        def _capture_inflight_at_sync():
            # The off-lock synchronize() runs only AFTER the in-flight marker is
            # published; a cross-peer read would observe and drain it here.
            seen["event"] = ch._recv_inflight_event
            seen["peer"] = ch._recv_inflight_peer_id
            seen["stream"] = ch._recv_inflight_stream

        event.synchronize.side_effect = _capture_inflight_at_sync
        monkeypatch.setattr(oschan.torch.npu, "Event", MagicMock(return_value=event))

        loop = asyncio.new_event_loop()
        try:
            n = loop.run_until_complete(
                ch.async_batched_read([object()], {TS_RECEIVER_ID: "recv1"})
            )
        finally:
            loop.close()

        assert n == 1
        event.record.assert_called_once_with(ch.transport_stream)
        event.synchronize.assert_called_once_with()
        # Published before the blocking drain...
        assert seen["event"] is event
        assert seen["peer"] == "recv1"
        assert seen["stream"] is ch.transport_stream
        # ...and retired after it drained (lone read owns then clears it).
        assert ch._recv_inflight_event is None
        assert ch._recv_inflight_peer_id is None
        assert ch._recv_inflight_stream is None

    def test_does_not_clear_inflight_taken_over_by_a_later_read(self, monkeypatch):
        # If a later read takes ownership of the staging window while this read
        # blocks off-lock, the retiring read must NOT clobber the new marker.
        ch = _read_channel()
        peer = _make_peer(conn_handle=0x2121)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=True))
        )
        monkeypatch.setattr(oschan.torch.npu, "set_device", MagicMock())

        event = _FakeEvent()
        later_event = _FakeEvent()

        def _simulate_takeover():
            # Mimic a concurrent read that grabbed the window during the drain.
            ch._recv_inflight_event = later_event
            ch._recv_inflight_peer_id = "recv2"

        event.synchronize.side_effect = _simulate_takeover
        monkeypatch.setattr(oschan.torch.npu, "Event", MagicMock(return_value=event))

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                ch.async_batched_read([object()], {TS_RECEIVER_ID: "recv1"})
            )
        finally:
            loop.close()

        assert ch._recv_inflight_event is later_event
        assert ch._recv_inflight_peer_id == "recv2"


# ---------------------------------------------------------------------------
# Throughput characterization: the sender stages + writes + posts ONE object
# at a time on a single _send_stream, and can only keep recv_slots writes in
# flight before blocking on a consumed notify. These device-free tests pin that
# behavior (no descriptor batching, slot-bounded pipeline depth) and provide a
# host-side per-object overhead micro-benchmark. The real GB/s number lives in
# the hardware benchmark below.
# ---------------------------------------------------------------------------
def _drive_read_request(
    ch: HcclOneSidedChannel,
    num_objs: int,
    monkeypatch,
    peer_id: str = "recv1",
) -> None:
    # torch.npu.stream(...) is a context manager and torch.npu.Event() records
    # the in-flight event; both are device ops, so stub them out.
    monkeypatch.setattr(oschan.torch.npu, "stream", MagicMock(name="stream_ctx"))
    monkeypatch.setattr(oschan.torch.npu, "Event", MagicMock(return_value=_FakeEvent()))
    req = OneSidedReadRequest(
        receiver_id=peer_id,
        request_id=0,
        sender_local_addrs=list(range(num_objs)),
        sizes=[ch._slot_bytes] * num_objs,
    )
    ch._handle_read_request(req)


class TestSenderSendsOneObjectAtATime:
    @pytest.mark.parametrize("num_objs", [1, 2, 4, 8, 16])
    def test_one_write_and_post_per_object(self, num_objs, monkeypatch):
        ch = _sender_channel(send_slots=2, recv_slots=2)
        _register_listener_peer(ch, remote_recv_slots=2)

        _drive_read_request(ch, num_objs, monkeypatch)

        # The send loop issues a separate stage-in copy, WriteAsync and
        # data_ready Post per object: a batched request of N objects becomes N
        # round-trips on the wire, never one fused N-descriptor transfer.
        assert ch.agent.batch_write.call_count == num_objs
        assert ch.agent.post.call_count == num_objs
        assert ch._copy_src_to_staging.call_count == num_objs

    @pytest.mark.parametrize("num_objs", [1, 4, 16])
    def test_batch_write_uses_single_descriptor_lists(self, num_objs, monkeypatch):
        ch = _sender_channel(send_slots=2, recv_slots=2)
        _register_listener_peer(ch, remote_recv_slots=2)

        _drive_read_request(ch, num_objs, monkeypatch)

        # Each batch_write carries exactly ONE (local, remote) MemDetails pair.
        # The C++ BatchWrite accepts a vector, so descriptor-level batching is
        # left on the table here -- this is the throughput knob the docstring
        # callout is about.
        for call in ch.agent.batch_write.call_args_list:
            conn, local_mds, remote_mds, _stream = call.args
            assert conn == ch._listener_peers["recv1"].conn_handle
            assert len(local_mds) == 1
            assert len(remote_mds) == 1

    @pytest.mark.parametrize(
        "num_objs,slots,expected_waits",
        [
            (2, 2, 0),  # exactly fills the ring: no reuse, no wait
            (4, 2, 2),  # reuses each of the 2 slots once
            (8, 2, 6),  # depth-2 pipeline: 6 of 8 objs block on consumed
            (8, 4, 4),  # deeper ring -> fewer stalls for the same batch
            (4, 4, 0),  # ring as deep as the batch -> no back-pressure
        ],
    )
    def test_reuse_gate_is_bounded_by_slot_count(
        self, num_objs, slots, expected_waits, monkeypatch
    ):
        ch = _sender_channel(send_slots=slots, recv_slots=slots)
        _register_listener_peer(ch, remote_recv_slots=slots)

        _drive_read_request(ch, num_objs, monkeypatch)

        # wait(consumed) fires once per slot REUSE, i.e. max(0, N - recv_slots).
        # This is the structural reason throughput is capped by the staging
        # pipeline depth (recv_slots == os_num_slots // 2), independent of how
        # many objects a single (delay_pull) request batches.
        assert ch.agent.wait.call_count == expected_waits


def test_sender_host_overhead_benchmark(monkeypatch):
    """Micro-benchmark of the host-side per-object cost of the send loop.

    With all device ops mocked to return instantly, this isolates the pure
    Python orchestration tax that every object pays (slot math, MemDetails
    construction, per-op pybind dispatch). It does NOT measure link bandwidth;
    it bounds the small-message ceiling implied by the one-object-at-a-time
    structure. No hard threshold is asserted (host speed varies); the value is
    logged for inspection via ``pytest -s``.
    """
    num_objs = 4000
    ch = _sender_channel(send_slots=2, recv_slots=2)
    _register_listener_peer(ch, remote_recv_slots=2)

    monkeypatch.setattr(oschan.torch.npu, "stream", MagicMock(name="stream_ctx"))
    monkeypatch.setattr(oschan.torch.npu, "Event", MagicMock(return_value=_FakeEvent()))
    req = OneSidedReadRequest(
        receiver_id="recv1",
        request_id=0,
        sender_local_addrs=list(range(num_objs)),
        sizes=[ch._slot_bytes] * num_objs,
    )

    start = time.perf_counter()
    ch._handle_read_request(req)
    elapsed = time.perf_counter() - start

    assert ch.agent.batch_write.call_count == num_objs
    per_obj_us = elapsed / num_objs * 1e6
    msg = (
        f"[hccl_onesided sender host overhead] {per_obj_us:.3f} us/obj over "
        f"{num_objs} objs (total {elapsed * 1e3:.3f} ms, device ops mocked)"
    )
    logger.info(msg)
    print("\n" + msg)


# ---------------------------------------------------------------------------
# Item 2a: teardown -> RECONNECT. After a failure drops the peer, the next
# pull's lazy_init_peer_connection re-handshakes and re-registers a fresh peer.
# ---------------------------------------------------------------------------
class TestTeardownThenReconnect:
    def _reconnect_channel(self) -> HcclOneSidedChannel:
        ch = _read_channel()
        ch.zmq_context = MagicMock(name="zmq_context")
        ch._make_buffer_infos = MagicMock(return_value=[])
        ch.agent.get_client_meta = MagicMock(return_value=b"client_meta")
        ch.agent.connect = MagicMock(return_value=0x9999)
        return ch

    def _mock_zmq(self, ch: HcclOneSidedChannel, monkeypatch) -> MagicMock:
        # One mock socket services both the init REQ socket and the
        # per-peer transfer REQ socket built in _register_peer.
        sock = MagicMock(name="zmq_socket")
        sock.recv.return_value = msgspec.msgpack.encode(
            OneSidedInitResponse(
                server_meta_bytes=pickle.dumps(b"server_meta"),
                buffer_infos=[],
                transfer_url="tcp://127.0.0.1:5555",
                slot_bytes=256,
                recv_slots=2,
                send_slots=2,
            )
        )
        monkeypatch.setattr(oschan, "get_zmq_socket", MagicMock(return_value=sock))
        monkeypatch.setattr(
            oschan, "get_zmq_context", MagicMock(return_value=MagicMock())
        )
        return sock

    def test_failure_then_reconnect_reregisters_peer(self, monkeypatch):
        ch = self._reconnect_channel()
        peer = _make_peer(conn_handle=0x1111)
        ch._connector_peers["recv1"] = peer
        _wire_read_plan(ch, peer)
        ch._wait_for_transfer_ack = MagicMock(
            return_value=msgspec.msgpack.encode(OneSidedReadAck(ok=False))
        )

        # 1) read failure tears the peer down
        with pytest.raises(RuntimeError):
            ch.batched_read([object()], {TS_RECEIVER_ID: "recv1"})
        assert "recv1" not in ch._connector_peers

        # 2) next pull re-handshakes (mocked ZMQ + agent.connect)
        self._mock_zmq(ch, monkeypatch)
        ch.lazy_init_peer_connection(
            local_id="recv1", peer_id="recv1", peer_init_url="127.0.0.1:6000"
        )

        # 3) a fresh peer/connection is registered (reconnect)
        assert "recv1" in ch._connector_peers
        new_peer = ch._connector_peers["recv1"]
        assert new_peer is not peer
        assert new_peer.conn_handle == 0x9999
        ch.agent.connect.assert_called_once()

    def test_lazy_init_is_noop_when_peer_present(self, monkeypatch):
        # Sanity: reconnect only happens when the peer was actually dropped;
        # an existing peer short-circuits without a new handshake.
        ch = self._reconnect_channel()
        peer = _make_peer(conn_handle=0x1111)
        ch._connector_peers["recv1"] = peer
        self._mock_zmq(ch, monkeypatch)

        ch.lazy_init_peer_connection(
            local_id="recv1", peer_id="recv1", peer_init_url="127.0.0.1:6000"
        )

        assert ch._connector_peers["recv1"] is peer
        ch.agent.connect.assert_not_called()


# ===========================================================================
# Hardware (2-NPU) integration tests
#
# These drive the real receiver-requested / sender-write data plane end to end
# (transport, NotifyPool, RDMA write, ZMQ), including the teardown + reconnect
# error paths. They are skipped unless two NPU devices are available.
# ===========================================================================
SENDER_DEV = int(os.environ.get("OS_TEST_SENDER_DEV", "0"))
RECEIVER_DEV = int(os.environ.get("OS_TEST_RECEIVER_DEV", "1"))


@dataclass
class OsChannelTestConfig:
    num_objs: int
    kv_shape: Tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    send_device_id: int = SENDER_DEV
    recv_device_id: int = RECEIVER_DEV
    timeout: int = 60
    # Number of back-to-back successful pulls on the SAME connection (no
    # re-handshake between them). >1 stresses cross-call staging/event ordering
    # on top of the within-call slot reuse driven by a large num_objs.
    num_reads: int = 1
    # Inject a deterministic one-shot failure to drive the teardown paths:
    #   sender   -> _handle_read_request raises (ok=False); both ends tear down.
    #   receiver -> _copy_staging_to_dsts raises after the sender's ok=True, so
    #               only the receiver tears down and the sender's stale listener
    #               conn is closed on the re-handshake overwrite.
    inject_sender_failure: bool = False
    inject_receiver_failure: bool = False
    # Throughput benchmark knobs. When measure_throughput is set, the receiver
    # times num_reads back-to-back pulls (after warmup_reads untimed pulls) and
    # records GB/s / objs-per-second into shared_dict["bench"].
    measure_throughput: bool = False
    warmup_reads: int = 5
    # OneSided staging tuning forwarded to the channel. os_num_slots controls
    # the pipeline depth (recv_slots == os_num_slots // 2); sweeping it shows
    # how the one-object-at-a-time staging protocol scales with depth.
    os_num_slots: Optional[int] = None
    os_slot_bytes: Optional[int] = None
    os_staging_bytes: Optional[int] = None


def _byte_size(kv_shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for d in kv_shape:
        n *= d
    return n * torch.tensor([], dtype=dtype).itemsize


def _get_allocator(
    device_id: int, kv_shape: Tuple[int, ...], dtype: torch.dtype
) -> PagedCpuGpuMemoryAllocator:
    allocator = PagedCpuGpuMemoryAllocator()
    buffer_size = _byte_size(kv_shape, dtype) * 200
    allocator.init_gpu_memory_allocator(
        buffer_size,
        [torch.Size(kv_shape)],
        [dtype],
        MemoryFormat.KV_2LTD,
        device_id,
    )
    return allocator


def _os_kwargs(config: OsChannelTestConfig) -> Dict[str, Any]:
    """Collect the optional OneSided staging knobs set on the config."""
    kwargs: Dict[str, Any] = {}
    if config.os_num_slots is not None:
        kwargs["os_num_slots"] = config.os_num_slots
    if config.os_slot_bytes is not None:
        kwargs["os_slot_bytes"] = config.os_slot_bytes
    if config.os_staging_bytes is not None:
        kwargs["os_staging_bytes"] = config.os_staging_bytes
    return kwargs


def _make_channel(
    role: str,
    device_id: int,
    buffer_ptr: int,
    buffer_size: int,
    align_bytes: int,
    local_url: str,
    local_id: str,
    os_kwargs: Optional[Dict[str, Any]] = None,
):
    return CreateTransferChannel(
        channel_type="hccl_onesided",
        async_mode=False,
        role=role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        buffer_type="npu",
        align_bytes=align_bytes,
        tp_rank=0,
        peer_init_url=local_url,
        local_id=local_id,
        **(os_kwargs or {}),
    )


def _wait_for(shared_dict: Dict[str, Any], key: str, timeout: float) -> None:
    start = time.time()
    while key not in shared_dict:
        time.sleep(0.1)
        if time.time() - start > timeout:
            raise TimeoutError(f"timed out waiting for shared flag {key!r}")


def _install_one_shot_copy_failure(channel) -> None:
    """Make the sender's stage-in copy raise exactly once.

    This raises INSIDE the real _handle_read_request (after its try: is
    entered), so the sender exercises _teardown_listener_peer, and the
    _transfer_loop reports ok=False to the receiver.
    """
    orig = channel._copy_src_to_staging
    state = {"failed": False}

    def flaky(*args, **kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("INJECTED one-shot sender stage-in failure")
        return orig(*args, **kwargs)

    channel._copy_src_to_staging = flaky


def _install_one_shot_copy_out_failure(channel) -> None:
    """Make the receiver's stage-out copy raise exactly once.

    The sender's write/post has already succeeded (ok=True), so only the
    receiver tears down; the sender's stale listener conn is then closed on the
    re-handshake overwrite in _handle_init_msg.
    """
    orig = channel._copy_staging_to_dsts
    state = {"failed": False}

    def flaky(*args, **kwargs):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("INJECTED one-shot receiver stage-out failure")
        return orig(*args, **kwargs)

    channel._copy_staging_to_dsts = flaky


def sender_process(config: OsChannelTestConfig, shared_dict: Dict[str, Any]) -> None:
    try:
        faulthandler.enable()
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        torch.npu.set_device(config.send_device_id)
        send_id = str(config.send_device_id)

        allocator = _get_allocator(config.send_device_id, config.kv_shape, config.dtype)
        align = _byte_size(config.kv_shape, config.dtype)

        objs = []
        expected = []
        for i in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            val = float(i) + 0.5
            obj.tensor.fill_(val)
            objs.append(obj)
            expected.append(val)

        local_url = f"0.0.0.0:398{config.send_device_id}"
        channel = _make_channel(
            "sender",
            config.send_device_id,
            allocator.gpu_allocator.buffer_ptr,
            allocator.gpu_allocator.buffer_size,
            align,
            local_url,
            send_id,
            os_kwargs=_os_kwargs(config),
        )

        if config.inject_sender_failure:
            _install_one_shot_copy_failure(channel)

        shared_dict["sender_buffer_uuid"] = channel.mem_handles[0].uuid
        shared_dict["expected_values"] = expected
        shared_dict["sender_init_done"] = True

        # Stay alive serving reads (background threads) until the receiver is
        # done with both the failed attempt and the successful retry.
        _wait_for(shared_dict, "receiver_done", config.timeout)
        time.sleep(0.2)
        channel.close()
    except Exception as e:
        logger.error(f"Sender process failed: {e}")
        sys.exit(1)


def _measure_read_throughput(
    channel,
    objs,
    transfer_spec: Dict[str, Any],
    expected,
    config: OsChannelTestConfig,
    shared_dict: Dict[str, Any],
    bytes_per_obj: int,
) -> None:
    """Time steady-state receiver-driven pull throughput and record it.

    ``batched_read`` host-syncs its transport stream before returning, so each
    timed iteration is a fully-drained end-to-end pull (request -> sender
    stage/write/post per object -> receiver copy-out). Throughput here is the
    aggregate over a single connection and a single ``_send_stream``, i.e. the
    one-object-at-a-time staging protocol's effective bandwidth.
    """
    # Warmup: absorb the first-touch handshake completion and any lazy device
    # allocations so they do not skew the steady-state measurement.
    for _ in range(config.warmup_reads):
        n = channel.batched_read(objs, transfer_spec)
        assert n == config.num_objs
    torch.npu.synchronize()

    start = time.perf_counter()
    total_objs = 0
    for _ in range(config.num_reads):
        total_objs += channel.batched_read(objs, transfer_spec)
    elapsed = time.perf_counter() - start

    # A fast-but-wrong path is not throughput: verify the last pull landed.
    for i, obj in enumerate(objs):
        data = obj.tensor.cpu()
        if not bool((data == expected[i]).all()):
            sample = data.flatten()[:5].float().numpy()
            raise AssertionError(
                f"benchmark object {i}: expected {expected[i]}, got {sample}"
            )

    total_bytes = total_objs * bytes_per_obj
    shared_dict["bench"] = {
        "os_num_slots": config.os_num_slots,
        "num_objs": config.num_objs,
        "num_reads": config.num_reads,
        "bytes_per_obj": bytes_per_obj,
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "gbps": (total_bytes / elapsed / 1e9) if elapsed > 0 else 0.0,
        "objs_per_s": (total_objs / elapsed) if elapsed > 0 else 0.0,
        "us_per_obj": (elapsed / total_objs * 1e6) if total_objs else 0.0,
    }
    logger.info(
        "Receiver throughput: %.3f GB/s, %.0f objs/s over %d objs x %d reads",
        shared_dict["bench"]["gbps"],
        shared_dict["bench"]["objs_per_s"],
        config.num_objs,
        config.num_reads,
    )


def receiver_process(config: OsChannelTestConfig, shared_dict: Dict[str, Any]) -> None:
    try:
        faulthandler.enable()
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        torch.npu.set_device(config.recv_device_id)
        send_id = str(config.send_device_id)
        recv_id = str(config.recv_device_id)

        allocator = _get_allocator(config.recv_device_id, config.kv_shape, config.dtype)
        align = _byte_size(config.kv_shape, config.dtype)

        objs = []
        for _ in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            obj.tensor.zero_()
            objs.append(obj)

        recv_url = f"0.0.0.0:398{config.recv_device_id}"
        sender_url = f"127.0.0.1:398{config.send_device_id}"
        channel = _make_channel(
            "receiver",
            config.recv_device_id,
            allocator.gpu_allocator.buffer_ptr,
            allocator.gpu_allocator.buffer_size,
            align,
            recv_url,
            recv_id,
            os_kwargs=_os_kwargs(config),
        )

        _wait_for(shared_dict, "sender_init_done", 30)
        sender_uuid = shared_dict["sender_buffer_uuid"]
        expected = list(shared_dict["expected_values"])

        if config.inject_receiver_failure:
            _install_one_shot_copy_out_failure(channel)

        channel.lazy_init_peer_connection(
            local_id=recv_id, peer_id=send_id, peer_init_url=sender_url
        )
        assert send_id in channel._connector_peers, "handshake did not register peer"

        transfer_spec = {
            TS_RECEIVER_ID: send_id,
            TS_REMOTE_BUFFER_UUIDS: [sender_uuid] * config.num_objs,
            TS_REMOTE_MEM_INDEXES: list(range(config.num_objs)),
        }

        expect_first_failure = (
            config.inject_sender_failure or config.inject_receiver_failure
        )
        if expect_first_failure:
            # 1) first pull must fail (sender ok=False, or receiver copy-out)
            raised = False
            try:
                channel.batched_read(objs, transfer_spec)
            except Exception as e:
                raised = True
                logger.info(f"Receiver: expected first-pull failure: {e}")
            assert raised, "injected failure did not surface to the receiver"

            # 2) teardown must have dropped the connector peer
            assert send_id not in channel._connector_peers, (
                "connector peer was not torn down after read failure"
            )

            # 3) reconnect: re-handshake into a fresh transport
            channel.lazy_init_peer_connection(
                local_id=recv_id, peer_id=send_id, peer_init_url=sender_url
            )
            assert send_id in channel._connector_peers, "reconnect did not re-register"

        if config.measure_throughput:
            _measure_read_throughput(
                channel, objs, transfer_spec, expected, config, shared_dict, align
            )
            shared_dict["receiver_done"] = True
            channel.close()
            return

        # Successful pull(s): the retry after reconnect / the baseline pull,
        # repeated num_reads times on the SAME connection. batched_read host-syncs
        # its transport stream before returning, so obj.tensor.cpu() below sees
        # the completed copy-out for each call. The real corruption signal is the
        # within-call slot reuse: num_objs >> recv_slots reuses the staging slots
        # many times, and each object carries a DISTINCT value (i + 0.5), so a
        # torn or misrouted WriteAsync surfaces as a per-object mismatch.
        for read_idx in range(config.num_reads):
            n = channel.batched_read(objs, transfer_spec)
            assert n == config.num_objs

            for i, obj in enumerate(objs):
                data = obj.tensor.cpu()
                if not bool((data == expected[i]).all()):
                    sample = data.flatten()[:5].float().numpy()
                    raise AssertionError(
                        f"read {read_idx} object {i}: "
                        f"expected {expected[i]}, got {sample}"
                    )

        logger.info(
            f"Receiver: verified {config.num_objs} objects "
            f"across {config.num_reads} read(s)."
        )
        shared_dict["receiver_done"] = True
        channel.close()
    except Exception as e:
        logger.error(f"Receiver process failed: {e}")
        # Unblock the sender even on failure so it does not hang to timeout.
        shared_dict["receiver_done"] = True
        sys.exit(1)


def _run(config: OsChannelTestConfig) -> Dict[str, Any]:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Manager() as manager:
        shared_dict = manager.dict()
        p_recv = mp.Process(
            target=receiver_process, args=(config, shared_dict), name="OsReceiver"
        )
        p_send = mp.Process(
            target=sender_process, args=(config, shared_dict), name="OsSender"
        )
        p_send.start()
        p_recv.start()

        p_recv.join(timeout=config.timeout)
        p_send.join(timeout=config.timeout)

        errors = []
        for proc in (p_send, p_recv):
            if proc.is_alive():
                proc.terminate()
                errors.append(f"{proc.name} timed out")
            elif proc.exitcode != 0:
                errors.append(f"{proc.name} failed with exitcode {proc.exitcode}")
        # Snapshot before the manager shuts the shared dict's server down.
        result = dict(shared_dict)
        if errors:
            pytest.fail("\n".join(errors))
    return result


_SKIP = pytest.mark.skipif(
    not torch.npu.is_available()
    or torch.npu.device_count() <= max(SENDER_DEV, RECEIVER_DEV),
    reason="Requires NPU devices for OS_TEST_SENDER_DEV and OS_TEST_RECEIVER_DEV",
)


@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize("num_objs", [1, 4, 16])
def test_onesided_read_baseline(num_objs):
    _run(
        OsChannelTestConfig(
            num_objs=num_objs,
            kv_shape=(4, 2, 8, 4, 16),
            inject_sender_failure=False,
        )
    )


@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize(
    "num_objs,num_reads",
    [
        (32, 1),  # one long pull: 32 slices ping-pong over the 2 recv slots
        (8, 8),  # the pingpong shape (8 batches), repeated 8x on one conn
        (64, 4),  # sustained: 64 slices x 4 sequential pulls = 256 transfers
    ],
)
def test_onesided_long_batches_slot_reuse(num_objs, num_reads):
    """Channel-level analog of test/test_hccl_onesided_pingpong.py.

    With the default os_num_slots (4 -> recv_slots == send_slots == 2), any
    num_objs > 2 FORCES the sender to reuse a staging slot before the receiver
    has finished with its previous occupant, gated only by the per-slot
    consumed/data_ready notify round-trip. Driving many slices (large num_objs)
    and repeating the pull on the same connection (num_reads) exercises the real
    agent.batch_write (WriteAsync) + Post/Wait back-pressure under sustained slot
    reuse, end to end. A torn write or a missed reuse gate would corrupt at least
    one object and fail verification.
    """
    _run(
        OsChannelTestConfig(
            num_objs=num_objs,
            num_reads=num_reads,
            kv_shape=(4, 2, 8, 4, 16),
            timeout=120,
            inject_sender_failure=False,
        )
    )


@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize("num_objs", [1, 4])
def test_onesided_teardown_and_reconnect(num_objs):
    """Sender-side failure: ok=False on both ends, then reconnect + retry."""
    _run(
        OsChannelTestConfig(
            num_objs=num_objs,
            kv_shape=(4, 2, 8, 4, 16),
            inject_sender_failure=True,
        )
    )


@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize("num_objs", [1, 2])
def test_onesided_receiver_failure_teardown_and_reconnect(num_objs):
    """Receiver-side copy-out failure: only the receiver tears down; the
    sender's stale listener conn is closed on the re-handshake overwrite,
    then the retry pull succeeds.

    Bounded to num_objs <= recv_slots (default num_slots//2 = 2): with slot
    reuse the receiver-only failure would leave the sender's send stream parked
    on wait(consumed) until the transport timeout, which is a distinct
    stream-drain path; this test targets the teardown/reconnect control flow.
    """
    _run(
        OsChannelTestConfig(
            num_objs=num_objs,
            kv_shape=(4, 2, 8, 4, 16),
            inject_receiver_failure=True,
        )
    )


# ===========================================================================
# Symmetric (DP2) bidirectional handshake repro
#
# Two peers connect to each other AT THE SAME TIME and then each pulls from the
# other. This is the exact shape that wedged the regular ``hccl`` channel for
# ~120s (see P2P_HANDSHAKE_FIX_PLAN.md §3/§4.4): both nodes initiate a connect
# concurrently, so EACH agent must run ``connect`` (receiver role) and
# ``accept`` (sender role) simultaneously. The one-sided agent gives every
# connection its own NotifyPool and never holds its agent mutex across the
# blocking handshake (csrc/hccl_onesided/onesided_agent.cpp), so this should NOT
# deadlock. A regression toward a shared/serialized handshake engine surfaces
# here as a process-join timeout.
# ===========================================================================
def symmetric_process(
    config: OsChannelTestConfig, shared_dict: Dict[str, Any], idx: int
) -> None:
    try:
        faulthandler.enable()
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        peer_idx = 1 - idx
        my_dev = config.send_device_id if idx == 0 else config.recv_device_id
        peer_dev = config.recv_device_id if idx == 0 else config.send_device_id
        torch.npu.set_device(my_dev)
        my_id = str(my_dev)
        peer_id = str(peer_dev)

        allocator = _get_allocator(my_dev, config.kv_shape, config.dtype)
        align = _byte_size(config.kv_shape, config.dtype)

        # Distinct value space per side so a cross-read is verifiable end to end.
        # The data we SERVE to the peer (src_objs) and the buffers we READ INTO
        # (dst_objs) MUST be different memory: reusing one buffer for both roles
        # races the peer's RDMA read of our pages against our own copy-out into
        # them. src_objs are the first num_objs allocations (pages 0..n-1), so
        # the peer's TS_REMOTE_MEM_INDEXES=range(num_objs) resolves to them.
        #
        # Values must stay bf16-EXACT (|val| < 128 keeps the .5 mantissa bit),
        # else verification fails on a perfectly correct transfer. Opposite
        # signs per side make a cross-talk read (peer vs own buffer) unmissable.
        sign = 1.0 if idx == 0 else -1.0
        src_objs = []
        expected = []
        for i in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            val = sign * (float(i) + 0.5)
            obj.tensor.fill_(val)
            src_objs.append(obj)
            expected.append(val)

        dst_objs = []
        for _ in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            obj.tensor.zero_()
            dst_objs.append(obj)

        # Separate port range (399x) from the one-directional tests (398x).
        my_url = f"0.0.0.0:399{my_dev}"
        channel = _make_channel(
            "both",
            my_dev,
            allocator.gpu_allocator.buffer_ptr,
            allocator.gpu_allocator.buffer_size,
            align,
            my_url,
            my_id,
            os_kwargs=_os_kwargs(config),
        )

        shared_dict[f"uuid_{idx}"] = channel.mem_handles[0].uuid
        shared_dict[f"expected_{idx}"] = expected
        shared_dict[f"init_{idx}"] = True

        _wait_for(shared_dict, f"init_{peer_idx}", 30)
        peer_uuid = shared_dict[f"uuid_{peer_idx}"]
        peer_expected = list(shared_dict[f"expected_{peer_idx}"])
        peer_url = f"127.0.0.1:399{peer_dev}"

        # Rendezvous so BOTH sides call connect at (close to) the same instant:
        # this maximizes the concurrent connect+accept overlap on each agent.
        shared_dict[f"ready_{idx}"] = True
        _wait_for(shared_dict, f"ready_{peer_idx}", 30)

        channel.lazy_init_peer_connection(
            local_id=my_id, peer_id=peer_id, peer_init_url=peer_url
        )
        assert peer_id in channel._connector_peers, (
            "symmetric handshake did not register the connector peer"
        )

        transfer_spec = {
            TS_RECEIVER_ID: peer_id,
            TS_REMOTE_BUFFER_UUIDS: [peer_uuid] * config.num_objs,
            TS_REMOTE_MEM_INDEXES: list(range(config.num_objs)),
        }

        for read_idx in range(config.num_reads):
            n = channel.batched_read(dst_objs, transfer_spec)
            assert n == config.num_objs
            for i, obj in enumerate(dst_objs):
                data = obj.tensor.cpu()
                if not bool((data == peer_expected[i]).all()):
                    sample = data.flatten()[:5].float().numpy()
                    raise AssertionError(
                        f"side {idx} read {read_idx} object {i}: "
                        f"expected {peer_expected[i]}, got {sample}"
                    )

        logger.info(
            "Symmetric side %d: verified %d objs x %d read(s) from peer %s",
            idx,
            config.num_objs,
            config.num_reads,
            peer_id,
        )
        shared_dict[f"done_{idx}"] = True
        # Stay alive serving the peer's reads until it has finished too.
        _wait_for(shared_dict, f"done_{peer_idx}", config.timeout)
        time.sleep(0.2)
        channel.close()
    except Exception as e:
        logger.error("Symmetric side %d failed: %s", idx, e)
        # Unblock the peer even on failure so it does not hang to timeout.
        shared_dict[f"done_{idx}"] = True
        sys.exit(1)


def _run_symmetric(config: OsChannelTestConfig) -> Dict[str, Any]:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Manager() as manager:
        shared_dict = manager.dict()
        procs = [
            mp.Process(
                target=symmetric_process,
                args=(config, shared_dict, idx),
                name=f"OsSym{idx}",
            )
            for idx in (0, 1)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=config.timeout)

        errors = []
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                errors.append(f"{proc.name} timed out (possible handshake deadlock)")
            elif proc.exitcode != 0:
                errors.append(f"{proc.name} failed with exitcode {proc.exitcode}")
        result = dict(shared_dict)
        if errors:
            pytest.fail("\n".join(errors))
    return result


@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize(
    "num_objs,num_reads",
    [
        (1, 1),  # minimal: one concurrent connect+accept per side
        (4, 4),  # repeated pulls on the symmetric pair
        (16, 2),  # slot reuse on top of the symmetric pair
    ],
)
def test_onesided_symmetric_bidirectional_handshake(num_objs, num_reads):
    """Both peers connect simultaneously, then each pulls from the other.

    Regression guard for the DP2 stall: if the one-sided handshake ever becomes
    serialized (so a node cannot ``accept`` while it is ``connect``-ing), this
    test hangs and fails on the process-join timeout instead of passing.
    """
    _run_symmetric(
        OsChannelTestConfig(
            num_objs=num_objs,
            num_reads=num_reads,
            kv_shape=(4, 2, 8, 4, 16),
            timeout=120,
        )
    )


# ===========================================================================
# Hardware throughput benchmark
#
# Quantifies the cost of the one-object-at-a-time staging protocol end to end.
# These are measurements, not pass/fail correctness gates (beyond a sanity
# check that data landed), so run with `-s` to see the reported GB/s, e.g.:
#
#   pytest -s -m hardware tests/v1/transfer_channel/test_hccl_onesided_channel.py \
#       -k throughput
# ===========================================================================
@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize("os_num_slots", [4, 8, 16, 32])
def test_onesided_read_throughput_vs_pipeline_depth(os_num_slots):
    """Sweep staging pipeline depth (recv_slots == os_num_slots // 2) and report
    receiver-driven pull throughput.

    The sender stages + WriteAsync + Posts every object individually on a single
    ``_send_stream`` (see ``_handle_read_request``) and can only keep
    ``recv_slots`` writes in flight before blocking on a consumed notify. Deeper
    rings let more per-object transfers overlap, so the GB/s delta across this
    sweep is exactly the throughput headroom left on the table by the
    one-object-at-a-time protocol (the same path delay_pull exercises through
    ``submit_batched_read``).
    """
    kv_shape = (8, 2, 16, 8, 128)
    slot_bytes = _byte_size(kv_shape, torch.bfloat16)
    result = _run(
        OsChannelTestConfig(
            num_objs=64,
            num_reads=100,
            warmup_reads=10,
            kv_shape=kv_shape,
            timeout=240,
            measure_throughput=True,
            os_num_slots=os_num_slots,
            os_slot_bytes=slot_bytes,
            # Staging must fit slot_bytes * num_slots (recv ring + send ring).
            os_staging_bytes=slot_bytes * os_num_slots,
        )
    )

    bench = result.get("bench")
    assert bench is not None, "benchmark did not record results (peer crashed?)"
    report = (
        f"[hccl_onesided throughput] os_num_slots={os_num_slots} "
        f"(pipeline depth={os_num_slots // 2}): "
        f"{bench['gbps']:.3f} GB/s | {bench['objs_per_s']:.0f} objs/s | "
        f"{bench['us_per_obj']:.2f} us/obj "
        f"({bench['num_objs']} objs x {bench['num_reads']} reads, "
        f"{bench['bytes_per_obj']} B/obj)"
    )
    logger.info(report)
    print("\n" + report)

    assert bench["gbps"] > 0.0


@pytest.mark.hardware
@_SKIP
@pytest.mark.parametrize("num_objs", [1, 8, 32, 128])
def test_onesided_read_throughput_vs_batch_size(num_objs):
    """Report throughput as the per-request object count grows at a fixed ring.

    With the default ring (os_num_slots=4 -> depth 2), a request batching N
    objects still serializes N stage/write/post round-trips on the sender. This
    shows how per-object overhead amortizes (or fails to) as the batch grows,
    which is the regime delay_pull micro-batches hit.
    """
    kv_shape = (8, 2, 16, 8, 128)
    result = _run(
        OsChannelTestConfig(
            num_objs=num_objs,
            num_reads=100,
            warmup_reads=10,
            kv_shape=kv_shape,
            timeout=240,
            measure_throughput=True,
        )
    )

    bench = result.get("bench")
    assert bench is not None, "benchmark did not record results (peer crashed?)"
    report = (
        f"[hccl_onesided throughput] num_objs={num_objs} (default ring): "
        f"{bench['gbps']:.3f} GB/s | {bench['objs_per_s']:.0f} objs/s | "
        f"{bench['us_per_obj']:.2f} us/obj "
        f"({bench['num_reads']} reads, {bench['bytes_per_obj']} B/obj)"
    )
    logger.info(report)
    print("\n" + report)

    assert bench["gbps"] > 0.0
