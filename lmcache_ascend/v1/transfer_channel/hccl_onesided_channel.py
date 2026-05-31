# SPDX-License-Identifier: Apache-2.0
"""HCCL write-driven one-sided transfer channel.

This channel keeps the PingPong-style socket handshake, but replaces
BatchChannel's credit ring with a bounded staging protocol:

  sender:   stage source pages -> WriteAsync into receiver staging ->
            Post(data_ready)
  receiver: Wait(data_ready) -> copy receiver staging to destination pages ->
            Post(consumed)

Only the small staging region owned by ``OneSidedAgent`` is HCCL-registered.
User KV buffers are copied into/out of staging on explicit non-default streams.

Scaling note (data plane): the sender side currently serializes all peers
through a single ``_transfer_loop`` thread, a single REP ``_transfer_socket``,
one ``_send_stream`` and ``_send_staging_lock``, so one slow receiver causes
cross-peer head-of-line blocking; ``_recv_staging_lock`` is also held across the
network round-trip and the device drain. See "Stage 6: Python data-plane
scaling" in ``hccl_onesided_channel_2211ac4c.plan.md`` for the planned
sharded-sender / dedicated-executor follow-ups.

TODO(high-concurrency hardening):
  - Move blocking one-sided reads off the process default executor so a backlog
    of data-plane work cannot starve connect/init/recovery coroutines.
  - Shard the sender data plane by peer (or N-way): worker/stream/staging
    ownership, and possibly ROUTER/DEALER or multiple REP sockets, so one slow
    receiver does not block unrelated peers.
  - Narrow ``_recv_staging_lock`` or make receive staging ownership per request
    / per slot so independent reads can overlap safely.
  - Phase 1 submit overlap only chains same-peer reads. Cross-peer submits still
    host-drain the prior read before sending the next request; true cross-peer
    overlap needs per-peer/request receive staging windows plus wire-visible slot
    bases.
  - Stress receiver-only failure with slot reuse (``num_objs > recv_slots``);
    today that path can park the sender on ``wait(consumed)`` until the HCCL
    transport timeout before teardown/re-handshake recovers.
  - Keep ``os_ack_timeout_sec`` below the sync caller timeout (or raise
    ``p2p_sync_get_timeout_s``) so the channel tears down poisoned peers before
    synchronous callers give up.
  - When used through ``AscendP2PBackend``, run in pull mode. The non-pull path
    calls ``async_batched_write``, which this receiver-requested channel does
    not implement.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple, Union
import asyncio
import pickle
import queue
import threading
import time

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.rpc_utils import get_ip, get_zmq_context, get_zmq_socket
from lmcache.v1.transfer_channel.abstract import BaseTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    InitSideMsgBase,
    InitSideRetMsgBase,
    SideMsg,
)
import msgspec
import torch
import zmq

# First Party
import lmcache_ascend.hccl_onesided_npu_comms as hos

# Local
from .buffer_config import (
    BufferConfig,
    BufferType,
    MemHandleMeta,
    PeerBufferInfo,
    RemotePeerBufferList,
    resolve_buffer_ref,
    resolve_local_addr,
)
from .hccl_onesided_protocol import (
    OneSidedInitRequest,
    OneSidedInitResponse,
    OneSidedMsg,
    OneSidedReadAck,
    OneSidedReadRequest,
)
from .transfer_spec import (
    TS_RECEIVER_ID,
    TS_REMOTE_BUFFER_UUIDS,
    TS_REMOTE_MEM_INDEXES,
    TS_STREAM,
)

logger = init_logger(__name__)

# Default upper bound for waiting on a transfer ack. The sender acks right
# after ENQUEUING its stream work (not after completion), so a healthy ack is
# fast regardless of transfer size; this only needs to ride out sender-side
# head-of-line blocking under concurrency. Sized below the p2p backend's
# p2p_sync_get_timeout_s (default max(30.0, ...)) so a sync caller surfaces a
# clean error/teardown instead of wedging. Override via os_ack_timeout_sec.
_DEFAULT_ACK_TIMEOUT_SEC = 30.0
_ASYNC_EVENT_POLL_SEC = 0.001
# Emit a one-shot diagnostic when an ack wait or a sender enqueue runs long
# (well before the hard ack timeout fires), so a developing wedge is visible in
# the logs instead of only surfacing as a fatal 30s timeout.
_SLOW_ACK_WARN_SEC = 2.0
_SLOW_ENQUEUE_WARN_SEC = 2.0


class _RecvInflightDrainError(RuntimeError):
    """Previous receive staging drain failed before a new peer request."""

    def __init__(self, previous_peer_id: Optional[str], next_peer_id: str):
        super().__init__(
            f"failed to drain previous OneSided read from "
            f"{previous_peer_id!r} before requesting {next_peer_id!r}"
        )
        self.previous_peer_id = previous_peer_id
        self.next_peer_id = next_peer_id


class _PeerState:
    __slots__ = (
        "conn_handle",
        "transfer_url",
        "transfer_req_socket",
        "transfer_req_lock",
        "remote_buffers",
        "remote_dirty_slots",
        "remote_slot_bytes",
        "remote_recv_slots",
        "remote_send_slots",
        "ready_event",
        "handshake_error",
    )

    def __init__(
        self,
        conn_handle: int,
        transfer_url: str,
        transfer_req_socket: Optional[zmq.Socket],
        remote_buffers: RemotePeerBufferList,
        remote_slot_bytes: int,
        remote_recv_slots: int,
        remote_send_slots: int,
    ):
        self.conn_handle = conn_handle
        self.transfer_url = transfer_url
        self.transfer_req_socket = transfer_req_socket
        self.transfer_req_lock = threading.Lock()
        self.remote_buffers = remote_buffers
        self.remote_dirty_slots: set[int] = set()
        self.remote_slot_bytes = remote_slot_bytes
        self.remote_recv_slots = remote_recv_slots
        self.remote_send_slots = remote_send_slots
        self.ready_event = threading.Event()
        self.handshake_error: Optional[BaseException] = None


class HcclOneSidedChannel(BaseTransferChannel):
    """Receiver-requested, sender-write HCCL staging channel."""

    _channel_name = "hccl_onesided"

    def __init__(
        self,
        async_mode: bool = False,
        buffers: Optional[List[BufferConfig]] = None,
        **kwargs,
    ):
        assert "role" in kwargs
        self.role = kwargs["role"]

        if buffers is None:
            assert "buffer_ptr" in kwargs
            assert "buffer_size" in kwargs
            assert "align_bytes" in kwargs
            buffers = [
                BufferConfig(
                    ptr=kwargs["buffer_ptr"],
                    size=kwargs["buffer_size"],
                    device_id=-1,
                    device_type=BufferType.CPU,
                    align_bytes=kwargs["align_bytes"],
                )
            ]

        self.page_size = buffers[0].align_bytes
        self.handle_device = torch.npu.current_device()

        cfg = hos.OneSidedConfig()
        cfg.staging_bytes = int(kwargs.get("os_staging_bytes", cfg.staging_bytes))
        cfg.num_slots = int(kwargs.get("os_num_slots", max(cfg.num_slots, 4)))
        cfg.tc = int(kwargs.get("os_tc", cfg.tc))
        cfg.sl = int(kwargs.get("os_sl", cfg.sl))
        cfg.timeout_sec = int(kwargs.get("os_timeout_sec", cfg.timeout_sec))
        if cfg.num_slots < 2:
            raise ValueError("os_num_slots must be >= 2 for full-duplex safety")
        self._os_config = cfg
        self._slot_bytes = int(kwargs.get("os_slot_bytes", self.page_size))
        if self._slot_bytes <= 0:
            raise ValueError("os_slot_bytes must be positive")
        if self.page_size > self._slot_bytes:
            raise ValueError(
                f"page_size {self.page_size} exceeds os_slot_bytes {self._slot_bytes}"
            )
        if self._slot_bytes * cfg.num_slots > cfg.staging_bytes:
            raise ValueError(
                "os_slot_bytes * os_num_slots must fit in os_staging_bytes"
            )
        slots = self._derive_equal_slots(cfg.num_slots, kwargs)
        self._recv_slots = slots
        self._send_slots = slots
        self._send_slot_base = slots

        self._ack_timeout_sec = float(
            kwargs.get("os_ack_timeout_sec", _DEFAULT_ACK_TIMEOUT_SEC)
        )
        if self._ack_timeout_sec <= 0:
            raise ValueError("os_ack_timeout_sec must be positive")

        # Worker count for the dedicated handshake executors. Should be >= the
        # number of peers that may handshake concurrently so connects/accepts
        # never queue. Sized per pool (connect and accept get one each).
        self._handshake_workers = int(kwargs.get("os_handshake_workers", 8))
        if self._handshake_workers < 1:
            raise ValueError("os_handshake_workers must be >= 1")

        self.agent = hos.OneSidedAgent.get_instance(self.handle_device)
        self.agent.init(cfg)

        self.mem_handles: List[MemHandleMeta] = []
        self._uuid_to_handle: dict[str, MemHandleMeta] = {}
        for buf in buffers:
            if buf.align_bytes <= 0:
                raise ValueError("BufferConfig.align_bytes must be positive")
            if buf.size % buf.align_bytes != 0:
                raise ValueError(
                    f"BufferConfig size {buf.size} must be a multiple of "
                    f"align_bytes {buf.align_bytes}"
                )
            addrs = list(range(buf.ptr, buf.ptr + buf.size, buf.align_bytes))
            meta = MemHandleMeta(
                mem_handle=None,
                buffer_ptr=buf.ptr,
                buffer_size=buf.size,
                page_size=buf.align_bytes,
                local_buffer_addrs=addrs,
                buffer_type=buf.device_type,
            )
            self.mem_handles.append(meta)
            self._uuid_to_handle[meta.uuid] = meta

        self.running = True
        self._state_lock = threading.Lock()
        self._connector_peers: dict[str, _PeerState] = {}
        self._listener_peers: dict[str, _PeerState] = {}
        self._peer_handshake_locks: dict[str, asyncio.Lock] = {}
        # Invoked with the peer_id whenever a connector (receiver-side) peer is
        # dropped from ``_connector_peers``. The owning backend uses it to evict
        # its own "already connected" cache (target_peer_info_mapping) so the
        # next lookup re-runs ensure_peer_connection -> re-handshake. Without
        # this, the channel forgets the peer but the backend still thinks it is
        # connected, so every later pull raises KeyError("Unknown peer ...").
        self._on_connector_peer_torndown: Optional[Callable[[str], None]] = None
        # One staging region per agent, partitioned by slot range so symmetric
        # pulls can progress without corrupting local staging:
        #   [0, recv_slots)                      remote writes into us
        #   [recv_slots, recv_slots+send_slots)  we stage and write to remote
        self._recv_staging_lock = threading.Lock()
        self._send_staging_lock = threading.Lock()
        self._recv_inflight_event: Optional[torch.npu.Event] = None
        self._recv_inflight_peer_id: Optional[str] = None
        self._recv_inflight_stream: Optional[torch.npu.Stream] = None
        self._send_inflight_event: Optional[torch.npu.Event] = None
        self.side_channels: List[zmq.Socket] = []
        self.running_threads: List[threading.Thread] = []
        self.local_id: Optional[str] = kwargs.get("local_id")
        self._request_seq = 0

        self.async_mode = async_mode
        self.zmq_context = get_zmq_context(use_asyncio=async_mode)
        self.event_loop = kwargs.get("event_loop", None)
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)

        peer_init_url = kwargs["peer_init_url"]
        if isinstance(peer_init_url, str) and peer_init_url.startswith("tcp://"):
            peer_init_url = peer_init_url[len("tcp://") :]
        self.peer_init_url = peer_init_url
        self._init_socket = get_zmq_socket(
            self.zmq_context, self.peer_init_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self._init_socket)
        init_last_endpoint = self._init_socket.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(init_last_endpoint, bytes):
            init_last_endpoint = init_last_endpoint.decode("utf-8")
        self.peer_init_url_resolved = self._resolve_url(init_last_endpoint, kwargs)

        transfer_bind = kwargs.get("os_transfer_bind_addr", "0.0.0.0:0")
        if isinstance(transfer_bind, str) and transfer_bind.startswith("tcp://"):
            transfer_bind = transfer_bind[len("tcp://") :]
        self._transfer_zmq_context = get_zmq_context(use_asyncio=False)
        self._transfer_socket = get_zmq_socket(
            self._transfer_zmq_context, transfer_bind, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self._transfer_socket)
        transfer_last_endpoint = self._transfer_socket.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(transfer_last_endpoint, bytes):
            transfer_last_endpoint = transfer_last_endpoint.decode("utf-8")
        self.transfer_url = self._resolve_url(transfer_last_endpoint, kwargs)

        # Dedicated non-default streams. Never use torch.npu.default_stream().
        self.transport_stream = torch.npu.Stream(torch.npu.current_device())
        self._send_stream = torch.npu.Stream(torch.npu.current_device())

        # Dedicated handshake executors, isolated from the asyncio default
        # executor (which carries blocking data-plane reads via
        # ``async_batched_read``). connect() and the accept-start dispatch get
        # SEPARATE pools on purpose: a pool full of blocked connects must never
        # be able to starve the accepts that would unblock them — the "a node
        # must always be able to accept while it is connecting" invariant from
        # P2P_HANDSHAKE_FIX_PLAN.md §4.3. Keeping both off the default executor
        # also means a data-plane read backlog cannot delay a handshake.
        self._connect_pool = ThreadPoolExecutor(
            max_workers=self._handshake_workers,
            thread_name_prefix="os-hs-connect",
        )
        self._accept_pool = ThreadPoolExecutor(
            max_workers=self._handshake_workers,
            thread_name_prefix="os-hs-accept",
        )
        # Stale-connection teardown (drain _send_stream + close_connection) on
        # an old_peer replacement must NOT run inline in the serialized init
        # accept path: a parked _send_stream there would block EVERY subsequent
        # handshake to this node. Hand it to a dedicated recovery pool so the
        # init loop can answer immediately.
        self._recovery_pool = ThreadPoolExecutor(
            max_workers=self._handshake_workers,
            thread_name_prefix="os-recovery",
        )

        # decouple-ack: the _transfer_loop validates + ACKs a read request and
        # then hands the actual staging/WriteAsync/Post enqueue to this single
        # FIFO worker. Post/Wait/WriteAsync are stream-enqueue ops, so the only
        # way the sender stalls is the ACL stream task queue filling (e.g. a
        # Wait(consumed) for a dead receiver that never drains). Doing that
        # enqueue here instead of inline keeps the _transfer_loop free to ACK
        # every other peer, so a single stuck stream can no longer push healthy
        # receivers past their ack timeout. One worker preserves global FIFO
        # order, which is a superset of the per-peer ordering the receiver
        # expects. ``None`` is the shutdown sentinel.
        self._send_job_queue: "queue.Queue[Optional[Tuple[OneSidedReadRequest, _PeerState]]]" = (  # noqa: E501
            queue.Queue()
        )

        self._init_side_channels()
        self._start_transfer_worker()
        self._start_send_worker()

    @staticmethod
    def _derive_equal_slots(num_slots: int, kwargs: dict) -> int:
        """Derive a single per-direction slot count from ``os_num_slots``.

        recv_slots and send_slots are intentionally EQUAL. The sender reuses
        local send slot ``send_slot_base + idx % send_slots`` at ``idx +
        send_slots``, and the only thing guaranteeing the prior WriteAsync out
        of that slot has drained is the consumed round-trip keyed on the remote
        recv slot ``idx % recv_slots``. That protection holds iff
        ``send_slots % recv_slots == 0``; forcing equality is the simplest
        guarantee and avoids torn outbound writes under slot reuse.

        The legacy ``os_recv_slots`` / ``os_send_slots`` knobs are no longer
        independently configurable: passing either with a value other than
        ``num_slots // 2`` raises ``ValueError`` so misconfig is loud rather
        than silently corrupting.
        """
        slots = num_slots // 2
        if slots < 1:
            raise ValueError("os_num_slots must be >= 2 (one recv + one send slot)")
        for legacy_key in ("os_recv_slots", "os_send_slots"):
            if legacy_key in kwargs and int(kwargs[legacy_key]) != slots:
                raise ValueError(
                    f"{legacy_key} is no longer configurable: recv_slots and "
                    f"send_slots are pinned equal to os_num_slots // 2 "
                    f"({slots}). Set os_num_slots instead."
                )
        return slots

    @staticmethod
    def _resolve_url(bound_url: str, kwargs: dict) -> str:
        url = bound_url
        if url.startswith("tcp://"):
            url = url[len("tcp://") :]
        if url.startswith("0.0.0.0"):
            advertised = kwargs.get("os_advertised_host")
            if not advertised:
                advertised = kwargs.get("pp_advertised_host")
            if not advertised:
                advertised = get_ip()
            url = url.replace("0.0.0.0", advertised, 1)
        return url

    def _init_side_channels(self) -> None:
        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            t = threading.Thread(target=self._init_loop, daemon=True)
            t.start()
            self.running_threads.append(t)

    def _start_transfer_worker(self) -> None:
        t = threading.Thread(target=self._transfer_loop, daemon=True)
        t.start()
        self.running_threads.append(t)

    def _start_send_worker(self) -> None:
        t = threading.Thread(target=self._send_worker_loop, daemon=True)
        t.start()
        self.running_threads.append(t)

    def _make_buffer_infos(self) -> List[PeerBufferInfo]:
        return [
            PeerBufferInfo(
                uuid=meta.uuid,
                buffer_ptr=meta.buffer_ptr,
                buffer_size=meta.buffer_size,
                page_size=meta.page_size,
                is_device=(meta.buffer_type == BufferType.NPU),
            )
            for meta in self.mem_handles
        ]

    @staticmethod
    def _make_accept_meta(client_meta, server_meta):
        meta = hos.HcclQpMeta()
        meta.dev_id = client_meta.dev_id
        meta.phy_dev_id = client_meta.phy_dev_id
        meta.ipv4_addr = client_meta.ipv4_addr
        meta.listen_port = server_meta.listen_port
        meta.tag_ctrl = server_meta.tag_ctrl
        return meta

    def _capture_local_id(self, local_id: str) -> None:
        if self.local_id is None:
            self.local_id = local_id
        elif self.local_id != local_id:
            raise ValueError(
                f"HcclOneSidedChannel local_id mismatch: have={self.local_id!r}, "
                f"got={local_id!r}"
            )

    def lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        self._capture_local_id(local_id)
        with self._state_lock:
            already_has_peer = peer_id in self._connector_peers
        if already_has_peer:
            if init_side_msg is None:
                return None
            init_tmp_socket = get_zmq_socket(
                self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
            )
            try:
                return self.send_init_side_msg(init_tmp_socket, init_side_msg)
            finally:
                init_tmp_socket.close()

        init_tmp_socket = get_zmq_socket(
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
        )
        try:
            req = OneSidedInitRequest(
                local_id=local_id,
                client_meta_bytes=pickle.dumps(self.agent.get_client_meta()),
                buffer_infos=self._make_buffer_infos(),
                slot_bytes=self._slot_bytes,
                recv_slots=self._recv_slots,
                send_slots=self._send_slots,
            )
            init_tmp_socket.send(msgspec.msgpack.encode(req))
            resp_bytes = init_tmp_socket.recv()
            resp = msgspec.msgpack.decode(resp_bytes, type=OneSidedMsg)
            if not isinstance(resp, OneSidedInitResponse) or not resp.server_meta_bytes:
                raise ConnectionError(
                    f"OneSided init handshake failed for peer {peer_id}: "
                    f"{type(resp).__name__}"
                )
            conn_handle = self.agent.connect(pickle.loads(resp.server_meta_bytes))
            self._register_peer(peer_id, conn_handle, resp)
            if init_side_msg is not None:
                return self.send_init_side_msg(init_tmp_socket, init_side_msg)
            return None
        finally:
            init_tmp_socket.close()

    def set_connector_teardown_callback(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        """Register a hook fired when a connector peer is torn down.

        The backend registers this so it can invalidate its own connection
        cache and force a re-handshake on the next lookup. The callback runs on
        whichever thread performed the teardown (transfer loop / recovery pool /
        close), so it must be cheap and thread-safe.
        """
        self._on_connector_peer_torndown = callback

    def _get_peer_handshake_lock(self, peer_id: str) -> asyncio.Lock:
        lock = self._peer_handshake_locks.get(peer_id)
        if lock is None:
            lock = asyncio.Lock()
            self._peer_handshake_locks[peer_id] = lock
        return lock

    async def async_lazy_init_peer_connection(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        async with self._get_peer_handshake_lock(peer_id):
            return await self._async_lazy_init_peer_connection_locked(
                local_id, peer_id, peer_init_url, init_side_msg
            )

    async def _async_lazy_init_peer_connection_locked(
        self,
        local_id: str,
        peer_id: str,
        peer_init_url: str,
        init_side_msg: Optional[InitSideMsgBase] = None,
    ) -> Optional[InitSideRetMsgBase]:
        self._capture_local_id(local_id)
        with self._state_lock:
            already_has_peer = peer_id in self._connector_peers
        if already_has_peer:
            if init_side_msg is None:
                return None
            init_tmp_socket = get_zmq_socket(
                self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
            )
            try:
                return await self.async_send_init_side_msg(
                    init_tmp_socket, init_side_msg
                )
            finally:
                init_tmp_socket.close()

        init_tmp_socket = get_zmq_socket(
            self.zmq_context, peer_init_url, "tcp", zmq.REQ, "connect"
        )
        try:
            req = OneSidedInitRequest(
                local_id=local_id,
                client_meta_bytes=pickle.dumps(self.agent.get_client_meta()),
                buffer_infos=self._make_buffer_infos(),
                slot_bytes=self._slot_bytes,
                recv_slots=self._recv_slots,
                send_slots=self._send_slots,
            )
            await init_tmp_socket.send(msgspec.msgpack.encode(req))
            resp_bytes = await init_tmp_socket.recv()
            resp = msgspec.msgpack.decode(resp_bytes, type=OneSidedMsg)
            if not isinstance(resp, OneSidedInitResponse) or not resp.server_meta_bytes:
                raise ConnectionError(
                    f"OneSided init handshake failed for peer {peer_id}: "
                    f"{type(resp).__name__}"
                )

            server_meta = pickle.loads(resp.server_meta_bytes)
            loop = asyncio.get_running_loop()
            device = self.handle_device

            def _connect_blocking() -> int:
                torch.npu.set_device(device)
                return self.agent.connect(server_meta)

            conn_handle = await loop.run_in_executor(
                self._connect_pool, _connect_blocking
            )
            self._register_peer(peer_id, conn_handle, resp)

            if init_side_msg is not None:
                return await self.async_send_init_side_msg(
                    init_tmp_socket, init_side_msg
                )
            return None
        finally:
            init_tmp_socket.close()

    def _register_peer(
        self,
        peer_id: str,
        conn_handle: int,
        resp: OneSidedInitResponse,
    ) -> None:
        sync_ctx = get_zmq_context(use_asyncio=False)
        transfer_req = get_zmq_socket(
            sync_ctx, resp.transfer_url, "tcp", zmq.REQ, "connect"
        )
        ack_timeout_ms = int(self._ack_timeout_sec * 1000)
        transfer_req.setsockopt(zmq.RCVTIMEO, ack_timeout_ms)
        transfer_req.setsockopt(zmq.SNDTIMEO, ack_timeout_ms)
        peer = _PeerState(
            conn_handle=conn_handle,
            transfer_url=resp.transfer_url,
            transfer_req_socket=transfer_req,
            remote_buffers=RemotePeerBufferList(resp.buffer_infos),
            remote_slot_bytes=resp.slot_bytes,
            remote_recv_slots=resp.recv_slots,
            remote_send_slots=resp.send_slots,
        )
        peer.ready_event.set()
        with self._state_lock:
            self._connector_peers[peer_id] = peer

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        with self._state_lock:
            return receiver_or_sender_id in self._listener_peers

    def _make_error_init_response(self) -> OneSidedInitResponse:
        return OneSidedInitResponse(
            server_meta_bytes=b"",
            buffer_infos=[],
            transfer_url="",
            slot_bytes=0,
            recv_slots=0,
            send_slots=0,
        )

    def _drain_and_close_stale_listener(self, conn_handle: int, local_id: str) -> None:
        """Retire a replaced listener connection off the init accept path.

        Runs on ``_recovery_pool`` so a parked ``_send_stream`` cannot block the
        serialized init loop. The synchronize drains the old conn's enqueued
        sends (bounded by the C++ transport timeout, ``os_timeout_sec``) before
        ``close_connection``; both steps are guarded so a device error can't
        kill the recovery worker. The replacement connection uses a different
        conn_handle, so closing this one concurrently is safe.
        """
        torch.npu.set_device(self.handle_device)
        try:
            self._send_stream.synchronize()
        except Exception as e:
            logger.error(
                "OneSided recovery: _send_stream.synchronize() failed while "
                "retiring stale listener conn for %s: %s",
                local_id,
                e,
            )
        try:
            self.agent.close_connection(conn_handle)
        except Exception as e:
            logger.warning(
                "OneSided recovery: failed to close stale listener conn for %s: %s",
                local_id,
                e,
            )

    def _handle_init_msg(
        self, req: Union[OneSidedMsg, InitSideMsgBase]
    ) -> Union[OneSidedMsg, InitSideRetMsgBase]:
        if isinstance(req, OneSidedInitRequest):
            logger.info("OneSided: init request from %s", req.local_id)
            server_meta = self.agent.get_server_meta()
            client_meta = pickle.loads(req.client_meta_bytes)
            peer = _PeerState(
                conn_handle=0,
                transfer_url="",
                transfer_req_socket=None,
                remote_buffers=RemotePeerBufferList(req.buffer_infos),
                remote_slot_bytes=req.slot_bytes,
                remote_recv_slots=req.recv_slots,
                remote_send_slots=req.send_slots,
            )
            local_id = req.local_id
            with self._state_lock:
                old_peer = self._listener_peers.get(local_id)
                self._listener_peers[local_id] = peer
            if old_peer is not None:
                # A prior connection from the same receiver is being replaced
                # (e.g. the receiver tore down its end after a copy/drain error
                # while our send succeeded, so we never ran _teardown_listener_
                # peer). Drop the stale transport so the C++ conns_ map does not
                # leak the old ConnState -- but do it OFF this thread. Draining
                # _send_stream here (an ok=True ack only means the old conn's
                # writes/posts were ENQUEUED) can block on a parked stream, and
                # _async_init_loop processes inits serially, so a block here
                # wedges every subsequent handshake to this node. Hand the
                # drain+close to the recovery pool; the new connection uses a
                # fresh conn_handle, so retiring the old one concurrently is
                # safe.
                self._recovery_pool.submit(
                    self._drain_and_close_stale_listener,
                    old_peer.conn_handle,
                    local_id,
                )

            accept_started = threading.Event()

            def _complete_handshake() -> None:
                torch.npu.set_device(self.handle_device)
                try:
                    accept_started.set()
                    accept_meta = self._make_accept_meta(client_meta, server_meta)
                    peer.conn_handle = self.agent.accept(accept_meta)
                    logger.info("OneSided: accepted connection from %s", local_id)
                except Exception as e:
                    peer.handshake_error = e
                    with self._state_lock:
                        if self._listener_peers.get(local_id) is peer:
                            del self._listener_peers[local_id]
                    logger.error("OneSided handshake failed for %s: %s", local_id, e)
                finally:
                    peer.ready_event.set()

            t = threading.Thread(target=_complete_handshake, daemon=True)
            t.start()
            if not accept_started.wait(timeout=10.0):
                raise TimeoutError("Timed out waiting for OneSided accept thread")

            return OneSidedInitResponse(
                server_meta_bytes=pickle.dumps(server_meta),
                buffer_infos=self._make_buffer_infos(),
                transfer_url=self.transfer_url,
                slot_bytes=self._slot_bytes,
                recv_slots=self._recv_slots,
                send_slots=self._send_slots,
            )

        if isinstance(req, InitSideMsgBase):
            return self.handle_init_side_msg(req)

        raise ValueError(f"Unsupported init message type: {type(req)}")

    def _init_loop(self) -> None:
        sock = self._init_socket
        self.init_side_channel = sock
        torch.npu.set_device(self.handle_device)
        sock.setsockopt(zmq.RCVTIMEO, 1000)

        while self.running:
            try:
                req_bytes = sock.recv()
            except zmq.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            try:
                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[OneSidedMsg, SideMsg]
                )
                resp = self._handle_init_msg(req)
                sock.send(msgspec.msgpack.encode(resp))
            except Exception as e:
                logger.error("OneSided init loop failure: %s", e)
                try:
                    sock.send(msgspec.msgpack.encode(self._make_error_init_response()))
                except Exception:
                    pass
                if self.running:
                    time.sleep(0.01)
        sock.close()

    async def _async_init_loop(self) -> None:
        sock = self._init_socket
        self.init_side_channel = sock
        torch.npu.set_device(self.handle_device)
        loop = asyncio.get_running_loop()

        while self.running:
            try:
                req_bytes = await asyncio.wait_for(sock.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                req = msgspec.msgpack.decode(
                    req_bytes, type=Union[OneSidedMsg, SideMsg]
                )
                resp = await loop.run_in_executor(
                    self._accept_pool, self._handle_init_msg, req
                )
                await sock.send(msgspec.msgpack.encode(resp))
            except Exception as e:
                logger.error("OneSided async init loop failure: %s", e)
                try:
                    await sock.send(
                        msgspec.msgpack.encode(self._make_error_init_response())
                    )
                except Exception:
                    pass
                if self.running:
                    await asyncio.sleep(0.01)
        sock.close()

    def _transfer_loop(self) -> None:
        sock = self._transfer_socket
        torch.npu.set_device(self.handle_device)
        sock.setsockopt(zmq.RCVTIMEO, 1000)

        while self.running:
            try:
                req_bytes = sock.recv()
            except zmq.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            self._dispatch_transfer_message(sock, req_bytes)

        sock.close()

    def _dispatch_transfer_message(self, sock: zmq.Socket, req_bytes: bytes) -> None:
        """Validate + ACK a transfer request, then queue its device enqueue.

        decouple-ack: the listener peer is resolved and the request validated
        here (failures still reported as ok=False), the ACK is sent, and the
        actual staging/WriteAsync/Post is handed to the send worker. The ACK no
        longer waits on the (potentially stream-backpressured) enqueue, so one
        stalled peer cannot time out every other receiver. Never raises: any
        error becomes an ok=False ack.
        """
        try:
            req = msgspec.msgpack.decode(req_bytes, type=OneSidedMsg)
        except Exception as e:
            logger.error("OneSided transfer decode failure: %s", e)
            self._send_transfer_ack(
                sock, OneSidedReadAck(ok=False, error=f"decode_error:{e}")
            )
            return

        try:
            if isinstance(req, OneSidedReadRequest):
                peer = self._validate_read_request(req)
                self._send_transfer_ack(sock, OneSidedReadAck(ok=True))
                self._send_job_queue.put((req, peer))
            else:
                raise ValueError(f"Unsupported transfer message: {type(req).__name__}")
        except Exception as e:
            logger.error("OneSided transfer dispatch failure: %s", e)
            self._send_transfer_ack(sock, OneSidedReadAck(ok=False, error=str(e)))

    def _send_worker_loop(self) -> None:
        """Drain queued read requests and run their device enqueue off-loop.

        The ACK already went out in ``_transfer_loop``; this worker owns
        ``_send_stream`` / ``_send_staging_lock`` / ``_send_inflight_event``
        exclusively (single FIFO consumer), so the chained inflight-event
        ordering stays correct. If the enqueue raises, ``_run_send_job`` has
        already torn down the listener peer; the receiver then hits its
        device-side ``data_ready`` wait timeout and recomputes.
        """
        torch.npu.set_device(self.handle_device)
        while self.running:
            try:
                job = self._send_job_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                break
            req, peer = job
            try:
                self._run_send_job(req, peer)
            except Exception as e:
                logger.error(
                    "OneSided send worker: enqueue failed for receiver %s "
                    "(listener torn down; receiver will recompute): %s",
                    req.receiver_id,
                    e,
                )

    @staticmethod
    def _send_transfer_ack(sock: zmq.Socket, ack: OneSidedReadAck) -> None:
        try:
            sock.send(msgspec.msgpack.encode(ack))
        except Exception as e:
            logger.error("OneSided: failed to send transfer ack: %s", e)

    _PEER_READY_TIMEOUT_SEC = 30.0

    def _await_peer_ready(
        self,
        peer_id: str,
        timeout: float = _PEER_READY_TIMEOUT_SEC,
        role: str = "connector",
    ) -> _PeerState:
        if role == "connector":
            registry = self._connector_peers
        elif role == "listener":
            registry = self._listener_peers
        else:
            raise ValueError(f"unknown peer role {role!r}")
        with self._state_lock:
            peer = registry.get(peer_id)
        if peer is None:
            raise KeyError(f"Unknown peer {peer_id!r} for role {role!r}")
        if not peer.ready_event.wait(timeout=timeout):
            raise TimeoutError(
                f"OneSided peer {peer_id!r} handshake did not complete "
                f"within {timeout}s"
            )
        if peer.handshake_error is not None:
            raise RuntimeError(
                f"OneSided peer {peer_id!r} handshake failed: {peer.handshake_error}"
            )
        return peer

    def get_local_mem_indices(
        self, objects: Union[List[bytes], List[MemoryObj]]
    ) -> List[int]:
        if not objects:
            return []
        if isinstance(objects[0], MemoryObj):
            return [obj.meta.address for obj in objects]  # type: ignore[union-attr]
        raise NotImplementedError("Sending raw bytes is not supported")

    def get_local_buffer_refs(
        self, objects: Union[List[bytes], List[MemoryObj]]
    ) -> Tuple[List[str], List[int]]:
        if not objects:
            return [], []
        if not isinstance(objects[0], MemoryObj):
            raise NotImplementedError("Sending raw bytes is not supported")
        buffer_uuids: List[str] = []
        mem_indexes: List[int] = []
        for mem_obj in objects:
            assert isinstance(mem_obj, MemoryObj)
            buf_uuid, mem_idx = resolve_buffer_ref(
                self.mem_handles, mem_obj.data_ptr, mem_obj.meta.address
            )
            buffer_uuids.append(buf_uuid)
            mem_indexes.append(mem_idx)
        return buffer_uuids, mem_indexes

    def _get_local_addr_and_kind(self, ptr: int, idx: int) -> Tuple[int, BufferType]:
        addr = resolve_local_addr(self.mem_handles, ptr, idx)
        for meta in self.mem_handles:
            if meta.buffer_ptr <= ptr < meta.buffer_ptr + meta.buffer_size:
                return addr, meta.buffer_type
        raise ValueError(f"Pointer {ptr} not found in any registered memory handle.")

    def _get_local_kind_by_addr(self, addr: int) -> BufferType:
        for meta in self.mem_handles:
            if meta.buffer_ptr <= addr < meta.buffer_ptr + meta.buffer_size:
                return meta.buffer_type
        raise ValueError(f"Address {addr} not found in any registered memory handle.")

    def _copy_src_to_staging(
        self,
        staging_addr: int,
        src_addr: int,
        size: int,
        stream_ptr: int,
    ) -> None:
        src_kind = self._get_local_kind_by_addr(src_addr)
        if src_kind == BufferType.NPU:
            hos.acl_memcpy_async_d2d(staging_addr, src_addr, size, stream_ptr)
        else:
            hos.acl_memcpy_async_h2d(staging_addr, src_addr, size, stream_ptr)

    def _resolve_remote_addrs(self, transfer_spec: dict) -> List[int]:
        if (
            TS_REMOTE_BUFFER_UUIDS in transfer_spec
            and TS_REMOTE_MEM_INDEXES in transfer_spec
        ):
            peer_id = transfer_spec[TS_RECEIVER_ID]
            with self._state_lock:
                remote_buffers = self._connector_peers[peer_id].remote_buffers
            return [
                remote_buffers.resolve_addr(buf_uuid, page_idx)
                for buf_uuid, page_idx in zip(
                    transfer_spec[TS_REMOTE_BUFFER_UUIDS],
                    transfer_spec[TS_REMOTE_MEM_INDEXES],
                    strict=True,
                )
            ]
        raise ValueError(
            "transfer_spec must contain (remote_buffer_uuids, remote_mem_indexes)"
        )

    def _build_read_plan(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: dict,
    ) -> Tuple[_PeerState, List[int], List[int], List[int], List[BufferType]]:
        peer_id = transfer_spec[TS_RECEIVER_ID]
        peer = self._await_peer_ready(peer_id, role="connector")
        sender_local_addrs = self._resolve_remote_addrs(transfer_spec)

        local_dst_addrs: List[int] = []
        dst_kinds: List[BufferType] = []
        sizes: List[int] = []
        for mem_obj in buffers:
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError("Sending raw bytes is not supported")
            addr, kind = self._get_local_addr_and_kind(
                mem_obj.data_ptr, mem_obj.meta.address
            )
            local_dst_addrs.append(addr)
            dst_kinds.append(kind)
            sizes.append(self.page_size)

        if len(local_dst_addrs) != len(sender_local_addrs):
            raise ValueError(
                f"local dst count ({len(local_dst_addrs)}) != sender addr count "
                f"({len(sender_local_addrs)})"
            )
        return peer, sender_local_addrs, sizes, local_dst_addrs, dst_kinds

    def _infer_receiver_id_for_request(self, transfer_spec: dict) -> str:
        if self.local_id is not None:
            return self.local_id
        explicit = transfer_spec.get("local_id")
        if explicit:
            self._capture_local_id(explicit)
            return explicit
        raise RuntimeError(
            "HcclOneSidedChannel local_id not set. Call lazy_init_peer_connection "
            "first, or pass local_id=... when constructing the channel."
        )

    def _next_request_id(self) -> int:
        with self._state_lock:
            request_id = self._request_seq
            self._request_seq += 1
        return request_id

    def _slot_offset(self, slot: int) -> int:
        return slot * self._slot_bytes

    def _recv_slot(self, idx: int) -> int:
        return idx % self._recv_slots

    def _send_slot(self, idx: int) -> int:
        return self._send_slot_base + (idx % self._send_slots)

    @staticmethod
    def _remote_recv_slot(peer: _PeerState, idx: int) -> int:
        return idx % peer.remote_recv_slots

    @staticmethod
    def _remote_slot_offset(peer: _PeerState, slot: int) -> int:
        return slot * peer.remote_slot_bytes

    def batched_send(self, *args, **kwargs):
        raise NotImplementedError("HcclOneSidedChannel uses receiver-requested reads")

    def batched_recv(self, *args, **kwargs):
        raise NotImplementedError("HcclOneSidedChannel uses receiver-requested reads")

    async def async_batched_send(self, *args, **kwargs):
        raise NotImplementedError("HcclOneSidedChannel uses receiver-requested reads")

    async def async_batched_recv(self, *args, **kwargs):
        raise NotImplementedError("HcclOneSidedChannel uses receiver-requested reads")

    def batched_write(self, *args, **kwargs):
        raise NotImplementedError("HcclOneSidedChannel does not support external write")

    async def async_batched_write(self, *args, **kwargs):
        raise NotImplementedError("HcclOneSidedChannel does not support external write")

    def _copy_staging_to_dsts(
        self,
        conn_handle: int,
        local_dst_addrs: List[int],
        sizes: List[int],
        dst_kinds: List[BufferType],
        stream_ptr: int,
    ) -> None:
        staging_base = self.agent.get_staging_base()
        for idx, (dst_addr, size, dst_kind) in enumerate(
            zip(local_dst_addrs, sizes, dst_kinds, strict=True)
        ):
            slot = self._recv_slot(idx)
            slot_off = self._slot_offset(slot)
            if idx >= self._recv_slots:
                self.agent.post(
                    conn_handle,
                    notify_idx=hos.consumed_notify(slot),
                    stream=stream_ptr,
                )
            self.agent.wait(
                conn_handle,
                notify_idx=hos.data_ready_notify(slot),
                stream=stream_ptr,
            )
            if dst_kind == BufferType.NPU:
                hos.acl_memcpy_async_d2d(
                    dst_addr,
                    staging_base + slot_off,
                    size,
                    stream_ptr,
                )
            else:
                hos.acl_memcpy_async_d2h(
                    dst_addr,
                    staging_base + slot_off,
                    size,
                    stream_ptr,
                )
        final_slots = min(len(sizes), self._recv_slots)
        for slot in range(final_slots):
            self.agent.post(
                conn_handle,
                notify_idx=hos.consumed_notify(slot),
                stream=stream_ptr,
            )

    def _clear_recv_inflight(self) -> None:
        self._recv_inflight_event = None
        self._recv_inflight_peer_id = None
        self._recv_inflight_stream = None

    def _prepare_recv_staging_for_peer(
        self,
        peer_id: str,
        stream: torch.npu.Stream,
    ) -> None:
        """Preserve receiver staging before sending the next read request.

        Same-peer submits can chain device-side: the sender owns
        ``remote_dirty_slots`` and waits on our per-slot ``consumed`` notify
        before overwriting a staging slot. Different peers have separate
        transports/notifies and cannot see each other's consumed credits, so
        drain the previous read event on the host before the new peer can write
        into the shared receive staging region.
        """
        event = self._recv_inflight_event
        if event is None:
            self._clear_recv_inflight()
            return

        previous_peer_id = self._recv_inflight_peer_id
        if previous_peer_id == peer_id:
            stream.wait_event(event)
        else:
            try:
                event.synchronize()
            except Exception as e:
                previous_stream = self._recv_inflight_stream
                previous_peer: Optional[_PeerState] = None
                if previous_peer_id is not None:
                    with self._state_lock:
                        previous_peer = self._connector_peers.get(previous_peer_id)
                self._clear_recv_inflight()
                if previous_peer is not None and previous_peer_id is not None:
                    self._teardown_connector_peer(
                        previous_peer_id, previous_peer, previous_stream
                    )
                raise _RecvInflightDrainError(previous_peer_id, peer_id) from e
        self._clear_recv_inflight()

    def batched_read(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        if not buffers:
            return 0
        peer, sender_addrs, sizes, local_dst_addrs, dst_kinds = self._build_read_plan(
            buffers, transfer_spec
        )
        request_id = self._next_request_id()
        req = OneSidedReadRequest(
            receiver_id=self._infer_receiver_id_for_request(transfer_spec),
            request_id=request_id,
            sender_local_addrs=sender_addrs,
            sizes=sizes,
        )

        peer_id = transfer_spec[TS_RECEIVER_ID]
        with self._recv_staging_lock:
            try:
                self._prepare_recv_staging_for_peer(peer_id, self.transport_stream)
                with peer.transfer_req_lock:
                    assert peer.transfer_req_socket is not None
                    peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                    ack_bytes = self._wait_for_transfer_ack(peer)
                ack = msgspec.msgpack.decode(ack_bytes, type=OneSidedMsg)
                self._raise_on_ack_failure(ack)
                self._copy_staging_to_dsts(
                    peer.conn_handle,
                    local_dst_addrs,
                    sizes,
                    dst_kinds,
                    self.transport_stream.npu_stream,
                )
                self.transport_stream.synchronize()
            except _RecvInflightDrainError:
                raise
            except Exception:
                self._clear_recv_inflight()
                self._teardown_connector_peer(peer_id, peer, self.transport_stream)
                raise
        return len(buffers)

    async def async_batched_read(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        if not buffers:
            return 0
        peer, sender_addrs, sizes, local_dst_addrs, dst_kinds = self._build_read_plan(
            buffers, transfer_spec
        )
        request_id = self._next_request_id()
        req = OneSidedReadRequest(
            receiver_id=self._infer_receiver_id_for_request(transfer_spec),
            request_id=request_id,
            sender_local_addrs=sender_addrs,
            sizes=sizes,
        )

        loop = asyncio.get_running_loop()
        stream_ptr = self._get_stream_ptr(transfer_spec)
        peer_id = transfer_spec[TS_RECEIVER_ID]

        def _run_blocking() -> int:
            torch.npu.set_device(self.handle_device)
            stream = self._get_torch_stream(transfer_spec)
            event: Optional[torch.npu.Event] = None
            try:
                with self._recv_staging_lock:
                    self._prepare_recv_staging_for_peer(peer_id, stream)
                    with peer.transfer_req_lock:
                        assert peer.transfer_req_socket is not None
                        peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                        ack_bytes = self._wait_for_transfer_ack(peer)
                    ack = msgspec.msgpack.decode(ack_bytes, type=OneSidedMsg)
                    self._raise_on_ack_failure(ack)
                    self._copy_staging_to_dsts(
                        peer.conn_handle,
                        local_dst_addrs,
                        sizes,
                        dst_kinds,
                        stream_ptr,
                    )
                    event = torch.npu.Event()
                    event.record(stream)
                    # Publish the in-flight copy-out BEFORE dropping the lock so a
                    # concurrent read to a DIFFERENT peer host-drains it (under the
                    # staging lock, in its _prepare_recv_staging_for_peer) before
                    # its sender can overwrite the shared receive staging. A
                    # same-peer read instead chains device-side on this event.
                    # Without publishing it, the off-lock synchronize() below
                    # leaves a window where a cross-peer read could send its
                    # request while this copy-out is still draining and corrupt
                    # the staging region.
                    self._recv_inflight_event = event
                    self._recv_inflight_peer_id = peer_id
                    self._recv_inflight_stream = stream
                # Block on the device event off-lock. Event.synchronize() wraps
                # npuEventSynchronize() and releases the GIL during the device
                # wait, so the sender loop and other reads make progress instead
                # of this executor thread spin-polling event.query() under
                # sleep().
                event.synchronize()
                # This blocking read fully drained; retire our own in-flight
                # marker, but only if a later read has not already taken
                # ownership of the staging window.
                with self._recv_staging_lock:
                    if self._recv_inflight_event is event:
                        self._clear_recv_inflight()
                return len(buffers)
            except _RecvInflightDrainError:
                raise
            except Exception:
                with self._recv_staging_lock:
                    if event is not None and self._recv_inflight_event is event:
                        self._clear_recv_inflight()
                self._teardown_connector_peer(peer_id, peer, stream)
                raise

        return await loop.run_in_executor(None, _run_blocking)

    def submit_batched_read(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> Optional[torch.npu.Event]:
        """Submit a read and return an event for receiver-buffer completion.

        The HCCL data plane writes into this channel's shared receive staging.
        This method returns after the sender ack and after receiver-side
        wait/copy-out/consumed work has been queued, but before that work is
        host-drained. A subsequent same-peer submit may chain on the returned
        event device-side; a different peer is host-drained before its request is
        sent because cross-peer HCCL connections do not share staging-slot
        consumed credits.
        """
        assert transfer_spec is not None
        self._assert_not_on_event_loop("submit_batched_read")
        if not buffers:
            return None
        peer, sender_addrs, sizes, local_dst_addrs, dst_kinds = self._build_read_plan(
            buffers, transfer_spec
        )
        request_id = self._next_request_id()
        req = OneSidedReadRequest(
            receiver_id=self._infer_receiver_id_for_request(transfer_spec),
            request_id=request_id,
            sender_local_addrs=sender_addrs,
            sizes=sizes,
        )
        stream_ptr = self._get_stream_ptr(transfer_spec)
        stream = self._get_torch_stream(transfer_spec)
        # The connector's ping-pong pool-reuse fence in `_remote_batched_to_gpu`
        # waits on `channel.transport_stream`, so the copy-out whose completion
        # this submit records its event on MUST be transport_stream. A foreign
        # TS_STREAM would record the event on one stream while the fence guards
        # another (and split the copy-out enqueue from the event), silently
        # breaking pool-reuse protection. Reject it loudly.
        if (
            stream is not self.transport_stream
            or stream_ptr != self.transport_stream.npu_stream
        ):
            raise ValueError(
                "submit_batched_read does not support a caller-provided "
                "TS_STREAM: copy-out must run on the channel transport_stream so "
                "the connector's ping-pong pool-reuse fence stays valid."
            )
        peer_id = transfer_spec[TS_RECEIVER_ID]

        with self._recv_staging_lock:
            try:
                self._prepare_recv_staging_for_peer(peer_id, stream)
                with peer.transfer_req_lock:
                    assert peer.transfer_req_socket is not None
                    peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                    ack_bytes = self._wait_for_transfer_ack(peer)
                ack = msgspec.msgpack.decode(ack_bytes, type=OneSidedMsg)
                self._raise_on_ack_failure(ack)
                self._copy_staging_to_dsts(
                    peer.conn_handle,
                    local_dst_addrs,
                    sizes,
                    dst_kinds,
                    stream_ptr,
                )
                event = torch.npu.Event()
                event.record(stream)
                self._recv_inflight_event = event
                self._recv_inflight_peer_id = peer_id
                self._recv_inflight_stream = stream
            except _RecvInflightDrainError:
                raise
            except Exception:
                self._clear_recv_inflight()
                self._teardown_connector_peer(peer_id, peer, stream)
                raise
        return event

    def _assert_not_on_event_loop(self, method: str) -> None:
        """Reject blocking entrypoints invoked on the channel's event loop.

        ``submit_batched_read`` busy-waits for the sender ack and host-drains
        device work, so running it on the asyncio loop that also drives
        init/handshake/recovery (``self.event_loop``) would wedge those
        coroutines. The intended callers are the worker thread (delay-pull
        submit) or a ``run_in_executor`` worker, never the loop thread itself.
        """
        loop = getattr(self, "event_loop", None)
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return
        if running is loop:
            raise RuntimeError(
                f"{method}() is blocking and must not be called on the channel "
                f"event loop; dispatch it to a worker thread or run_in_executor."
            )

    def _get_stream_ptr(self, transfer_spec: dict) -> int:
        stream = transfer_spec.get(TS_STREAM, None)
        if stream is not None:
            if isinstance(stream, int):
                return stream
            return stream.npu_stream
        return self.transport_stream.npu_stream

    def _get_torch_stream(self, transfer_spec: dict) -> torch.npu.Stream:
        stream = transfer_spec.get(TS_STREAM, None)
        if stream is not None and isinstance(stream, torch.npu.Stream):
            return stream
        return self.transport_stream

    @staticmethod
    def _raise_on_ack_failure(ack: OneSidedMsg) -> None:
        if isinstance(ack, OneSidedReadAck):
            if not ack.ok:
                raise RuntimeError(f"OneSided remote returned error: {ack.error}")
            return
        raise RuntimeError(f"Unexpected OneSided ack type: {type(ack).__name__}")

    def _wait_for_transfer_ack(
        self,
        peer: _PeerState,
        deadline_sec: Optional[float] = None,
    ) -> bytes:
        assert peer.transfer_req_socket is not None
        if deadline_sec is None:
            deadline_sec = self._ack_timeout_sec
        start = time.time()
        deadline = start + deadline_sec
        warn_after = start + min(_SLOW_ACK_WARN_SEC, deadline_sec / 2.0)
        warned = False
        while time.time() < deadline:
            try:
                return peer.transfer_req_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                if not warned and time.time() >= warn_after:
                    warned = True
                    logger.warning(
                        "OneSided slow transfer ack: still waiting on the "
                        "sender after %.1fs (hard timeout %.1fs); the sender "
                        "_transfer_loop is likely back-pressured.",
                        time.time() - start,
                        deadline_sec,
                    )
                time.sleep(_ASYNC_EVENT_POLL_SEC)
        raise TimeoutError(
            f"OneSided timed out waiting for transfer ack after {deadline_sec}s"
        )

    def _teardown_connector_peer(
        self,
        peer_id: str,
        peer: _PeerState,
        stream: Optional[torch.npu.Stream],
    ) -> None:
        """Tear down the receiver end of a connection after a read failure.

        Restores the clean protocol state ("all notifies 0 + no dirty slots")
        by dropping the transport so the next pull re-handshakes. Each step is
        guarded: the originating failure may itself be a device/synchronize
        error, and any wait(data_ready) the receiver already queued only drains
        after the (lowered) transport timeout, but close_connection, socket
        recycling, and the peer drop must all still run.
        """
        if stream is not None:
            try:
                stream.synchronize()
            except Exception as e:
                logger.error(
                    "OneSided teardown: stream.synchronize() failed for "
                    "connector %s: %s",
                    peer_id,
                    e,
                )
        try:
            self.agent.close_connection(peer.conn_handle)
        except Exception as e:
            logger.error(
                "OneSided teardown: close_connection failed for connector %s: %s",
                peer_id,
                e,
            )
        # The REQ socket may be poisoned (EFSM) by a send-without-recv on the
        # ack timeout / error path; close it with linger=0 so it is never
        # reused. The re-handshake builds a fresh transfer_req_socket.
        sock = peer.transfer_req_socket
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass
            peer.transfer_req_socket = None
        removed = False
        with self._state_lock:
            if self._connector_peers.get(peer_id) is peer:
                del self._connector_peers[peer_id]
                removed = True
        # Tell the backend to drop its cached "connected" state so the next
        # lookup re-handshakes this peer. Done outside the state lock and fully
        # guarded: the callback crosses into the backend and must never turn a
        # teardown into a crash.
        if removed and self._on_connector_peer_torndown is not None:
            try:
                self._on_connector_peer_torndown(peer_id)
            except Exception as e:
                logger.error(
                    "OneSided teardown: connector teardown callback failed for "
                    "peer %s: %s",
                    peer_id,
                    e,
                )

    def _teardown_listener_peer(self, peer_id: str, peer: _PeerState) -> None:
        """Tear down the sender end of a connection after a write failure.

        Mirror of the receiver teardown: drain the send stream (guarded), drop
        the transport and the listener entry, and clear the in-flight send
        event so the next handshake rebuilds clean state. ``remote_dirty_slots``
        dies with the dropped peer.
        """
        try:
            self._send_stream.synchronize()
        except Exception as e:
            logger.error(
                "OneSided teardown: _send_stream.synchronize() failed for "
                "listener %s: %s",
                peer_id,
                e,
            )
        try:
            self.agent.close_connection(peer.conn_handle)
        except Exception as e:
            logger.error(
                "OneSided teardown: close_connection failed for listener %s: %s",
                peer_id,
                e,
            )
        self._send_inflight_event = None
        with self._state_lock:
            if self._listener_peers.get(peer_id) is peer:
                del self._listener_peers[peer_id]

    def _validate_read_request(self, req: OneSidedReadRequest) -> _PeerState:
        """Cheap, ACK-gating checks for an incoming read request.

        Runs on ``_transfer_loop`` before the ACK, so it must stay fast: a
        length check plus resolving the (already handshaked) listener peer.
        Anything that raises here is reported to the receiver as ``ok=False``;
        the heavier device enqueue is deferred to ``_run_send_job`` after the
        ACK.
        """
        if len(req.sender_local_addrs) != len(req.sizes):
            raise ValueError("OneSidedReadRequest addrs/sizes length mismatch")
        return self._await_peer_ready(req.receiver_id, role="listener")

    def _handle_read_request(self, req: OneSidedReadRequest) -> None:
        """Validate + enqueue a read request synchronously.

        Production decouples these for the ACK: ``_transfer_loop`` calls
        ``_validate_read_request`` (the ACK gate) and the send worker calls
        ``_run_send_job`` (the device enqueue). This combined form is retained
        for the host-only sender unit tests and any synchronous caller; note it
        does NOT send an ACK (the socket layer owns that).
        """
        peer = self._validate_read_request(req)
        self._run_send_job(req, peer)

    def _run_send_job(self, req: OneSidedReadRequest, peer: _PeerState) -> None:
        """Stage source pages, WriteAsync into the receiver, Post(data_ready).

        Runs on the dedicated send worker (never the _transfer_loop), so a
        full ACL stream task queue (e.g. a Wait(consumed) for a dead receiver)
        back-pressures only this worker, not the ACKs. On error the listener
        peer is torn down and the exception is re-raised for the worker to log.
        """
        staging_base = self.agent.get_staging_base()
        enqueue_start = time.time()

        try:
            with self._send_staging_lock:
                if self._send_inflight_event is not None:
                    self._send_stream.wait_event(self._send_inflight_event)
                    self._send_inflight_event = None
                with torch.npu.stream(self._send_stream):
                    for idx, (src_addr, size) in enumerate(
                        zip(req.sender_local_addrs, req.sizes, strict=True)
                    ):
                        local_slot = self._send_slot(idx)
                        remote_slot = self._remote_recv_slot(peer, idx)
                        local_slot_off = self._slot_offset(local_slot)
                        remote_slot_off = self._remote_slot_offset(peer, remote_slot)
                        if remote_slot in peer.remote_dirty_slots:
                            self.agent.wait(
                                peer.conn_handle,
                                notify_idx=hos.consumed_notify(remote_slot),
                                stream=self._send_stream.npu_stream,
                            )
                            peer.remote_dirty_slots.remove(remote_slot)
                        self._copy_src_to_staging(
                            staging_base + local_slot_off,
                            src_addr,
                            size,
                            self._send_stream.npu_stream,
                        )
                        local_md = hos.MemDetails(addr=local_slot_off, size=size, key=0)
                        remote_md = hos.MemDetails(
                            addr=remote_slot_off, size=size, key=0
                        )
                        self.agent.batch_write(
                            peer.conn_handle,
                            [local_md],
                            [remote_md],
                            self._send_stream.npu_stream,
                        )
                        self.agent.post(
                            peer.conn_handle,
                            notify_idx=hos.data_ready_notify(remote_slot),
                            stream=self._send_stream.npu_stream,
                        )
                        peer.remote_dirty_slots.add(remote_slot)
                    self._send_inflight_event = torch.npu.Event()
                    self._send_inflight_event.record(self._send_stream)
            enqueue_elapsed = time.time() - enqueue_start
            if enqueue_elapsed >= _SLOW_ENQUEUE_WARN_SEC:
                # The ACK is now decoupled (sent before this enqueue), so a slow
                # enqueue no longer trips the receiver ack timeout. It still
                # means the ACL stream task queue is back-pressured -- typically
                # a Wait(consumed) for a slow/dead receiver that is not draining
                # -- which delays data delivery and grows the send-job queue.
                logger.warning(
                    "OneSided send worker: enqueue for peer %s took %.1fs across "
                    "%d objects; the ACL stream is back-pressured (ACK already "
                    "sent, so receivers are not failed, but delivery is delayed).",
                    req.receiver_id,
                    enqueue_elapsed,
                    len(req.sizes),
                )
        except Exception:
            # Sender-side error: tear down so the next handshake rebuilds a
            # clean transport. The receiver observes ok=False and tears down
            # its end independently; unique per-handshake tags make the
            # ordering benign.
            self._teardown_listener_peer(req.receiver_id, peer)
            raise

    def close(self) -> None:
        self.running = False
        # Wake the send worker if it is parked on an empty queue so it can exit
        # promptly instead of waiting out its get() timeout.
        send_queue = getattr(self, "_send_job_queue", None)
        if send_queue is not None:
            try:
                send_queue.put_nowait(None)
            except Exception:
                pass
        for thread in self.running_threads:
            thread.join(timeout=2.0)
        # Drop queued handshakes and stop accepting new ones. Don't wait on
        # in-flight connects/accepts: a connect can block on the socket
        # rendezvous for up to the transport timeout, and close() must not hang
        # that long. cancel_futures clears anything still queued.
        for pool in (
            getattr(self, "_connect_pool", None),
            getattr(self, "_accept_pool", None),
            getattr(self, "_recovery_pool", None),
        ):
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
        with self._state_lock:
            for peer in self._connector_peers.values():
                if peer.transfer_req_socket is not None:
                    try:
                        peer.transfer_req_socket.close()
                    except Exception:
                        pass
            self._connector_peers.clear()
            self._listener_peers.clear()
        for sock in self.side_channels:
            try:
                sock.close()
            except Exception:
                pass
        try:
            self.zmq_context.term()
        except Exception:
            pass
        try:
            self._transfer_zmq_context.term()
        except Exception:
            pass
