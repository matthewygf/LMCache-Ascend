# SPDX-License-Identifier: Apache-2.0
"""HCCL ping-pong transfer channel.

A receiver-driven point-to-point KV-cache transfer channel built on the
BatchChannel staging primitive (csrc/hccl_pingpong/). Unlike the existing
``HcclChannel`` which RDMA-reads/writes user buffers directly, this channel
funnels every transfer through a small device-shared ping-pong staging region
managed by the per-device ``PingPongAgent`` singleton.

Wire-level coordination uses two ZMQ socket layers:

  - **Init handshake**: standard REP-on-listener / REQ-on-connector for one-shot
    bring-up, exchanging the C++ ``PingPongClientMeta`` / ``PingPongServerMeta``
    plus per-side buffer infos and the listener's transfer URL.
  - **Transfer time**: each channel binds ONE shared transfer REP socket; every
    peer that wants to read from us issues a ``PingPongReadRequest`` /
    ``PingPongScatterRequest`` against that single URL. A daemon worker thread
    on the listener polls the REP socket and dispatches into the agent.

For the H2D-only v1 of ``BatchChannel.ScatterSend``, this channel exposes
``async_batched_scatter`` for host-source senders. NPU-source senders fall back
to two-step ``async_batched_read`` until ``scatter_send_d2d_followup`` lands.
"""

# Standard
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import pickle
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
import lmcache_ascend.hccl_pingpong_npu_comms as hpp

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
from .hccl_pingpong_protocol import (
    PingPongInitRequest,
    PingPongInitResponse,
    PingPongMsg,
    PingPongReadAck,
    PingPongReadRequest,
    PingPongScatterAck,
    PingPongScatterEntryMsg,
    PingPongScatterRequest,
)
from .transfer_spec import (
    TS_RECEIVER_ID,
    TS_REMOTE_BUFFER_UUIDS,
    TS_REMOTE_MEM_INDEXES,
)

logger = init_logger(__name__)


# How long (seconds) to wait when receiving a transfer-side ack before giving up.
_TRANSFER_ACK_TIMEOUT_SEC = 600
# Polling interval while busy-waiting on stream events from async paths.
_ASYNC_EVENT_POLL_SEC = 0.001


class _PeerState:
    """Per-peer state owned by the channel.

    ``conn_handle`` is the opaque BatchChannel pointer returned from
    ``PingPongAgent.accept`` / ``connect`` (cast to int by pybind).
    ``transfer_url`` is the remote channel's shared transfer REP socket URL.
    ``transfer_req_socket`` is a long-lived REQ socket connected to that URL.
    ``remote_buffers`` lets us resolve (uuid, page_index) -> remote (sender)
    virtual addresses without round-tripping for every page.

    ``ready_event`` is set once the BatchChannel is fully wired up. On the
    connector side this is immediately after ``agent.connect()`` returns
    (the peer is created already-ready by ``_register_peer``). On the
    listener side we register a *placeholder* peer up-front so that a
    transfer request which races ahead of ``agent.accept()`` can block on
    this event instead of seeing a missing peer entry. ``handshake_error``
    is set (and the placeholder is dropped from ``_peers``) if the
    background ``accept()`` raises, so the awaiter sees a clean
    ``RuntimeError`` rather than hanging or operating on a zero
    ``conn_handle``.
    """

    __slots__ = (
        "conn_handle",
        "transfer_url",
        "transfer_req_socket",
        "transfer_req_lock",
        "remote_buffers",
        "ready_event",
        "handshake_error",
    )

    def __init__(
        self,
        conn_handle: int,
        transfer_url: str,
        transfer_req_socket: Optional[zmq.Socket],
        remote_buffers: RemotePeerBufferList,
    ):
        self.conn_handle = conn_handle
        self.transfer_url = transfer_url
        self.transfer_req_socket = transfer_req_socket
        # Single REQ socket per peer with a fairness lock so concurrent
        # batched_read + async_batched_scatter calls don't garble its
        # request/response framing.
        self.transfer_req_lock = threading.Lock()
        self.remote_buffers = remote_buffers
        # Cleared by callers immediately after they finish populating
        # ``conn_handle``. ``_await_peer_ready`` blocks on this before
        # touching the conn handle, so transfer requests that arrive
        # between the init response and ``accept()`` completing block
        # rather than see a half-initialized peer.
        self.ready_event = threading.Event()
        self.handshake_error: Optional[BaseException] = None


class HcclPingPongChannel(BaseTransferChannel):
    """Receiver-driven point-to-point KV-cache transfer channel.

    Recognized ``**kwargs`` (in addition to the abstract base's required
    ``role`` / ``peer_init_url``):

    Required:
      ``role``                   : str  -- caller's role label ("sender" etc.).
      ``peer_init_url``          : str  -- ``host:port`` to bind the init REP
                                          socket. Use ``"0.0.0.0:<port>"`` for
                                          wildcard; downstream peers receive
                                          ``pp_advertised_host`` instead.
      ``buffer_ptr``/``buffer_size``/``align_bytes`` : int -- only required
                                          when ``buffers=None``. ``buffer_size``
                                          MUST be a multiple of ``align_bytes``.

    Optional ping-pong tuning:
      ``pp_chunk_size_bytes``    : int  -- RDMA chunk size (default 2 MiB).
      ``pp_n_chunks_per_buff``   : int  -- chunks per ping-pong buffer.
      ``pp_n_buffs``             : int  -- ping-pong buffer count (must be 2).
      ``pp_wait_recv_done``      : bool -- whether Send waits for the final
                                          recv-done notify.
      ``pp_tc``, ``pp_sl``       : int  -- RoCE TC / SL.

    Optional wire-level:
      ``pp_transfer_bind_addr``  : str  -- bind address for the transfer REP
                                          socket. Defaults to ``"0.0.0.0:0"``
                                          (kernel picks an ephemeral port).
      ``pp_advertised_host``     : str  -- host advertised to peers when the
                                          transfer socket is bound to a
                                          wildcard ``0.0.0.0``. Defaults to
                                          ``lmcache.v1.rpc_utils.get_ip()``.

    Optional misc:
      ``local_id``               : str  -- our own id. Otherwise captured on
                                          the first ``lazy_init_peer_connection``.
      ``event_loop`` : asyncio.AbstractEventLoop -- required when
                                          ``async_mode=True``.
    """

    _channel_name = "hccl_pingpong"

    # Set once per process the first time any pingpong transfer observes an
    # ``ok=False`` ack. After that point every subsequent ``recv_batch`` /
    # ``scatter_recv`` on this device queues behind ~10s of phantom
    # ``Wait`` drain per chunk (see the poisoned-streams note in
    # ``_wait_for_transfer_ack`` and ``para.timeout`` in
    # ``csrc/hccl_pingpong/batch_channel.cc``). Flipping this flag from
    # the puller / sender hot path produces one loud, easy-to-grep marker
    # in the worker logs so the operator can correlate cascade timeouts
    # against their underlying trigger.
    _poisoned_flag_logged: bool = False

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

        # ----- agent + config -----
        device_id = torch.npu.current_device()
        self.handle_device = device_id

        cfg = hpp.PingPongConfig()
        cfg.chunk_size_bytes = int(
            kwargs.get("pp_chunk_size_bytes", cfg.chunk_size_bytes)
        )
        cfg.n_chunks_per_buff = int(
            kwargs.get("pp_n_chunks_per_buff", cfg.n_chunks_per_buff)
        )
        cfg.n_buffs = int(kwargs.get("pp_n_buffs", cfg.n_buffs))
        cfg.wait_recv_done = bool(kwargs.get("pp_wait_recv_done", cfg.wait_recv_done))
        cfg.tc = int(kwargs.get("pp_tc", cfg.tc))
        cfg.sl = int(kwargs.get("pp_sl", cfg.sl))
        self._pp_config = cfg

        self.agent = hpp.PingPongAgent.get_instance(device_id)
        self.agent.init(cfg)

        # ----- logical buffer registration (no MR; uuid/page mapping only) -----
        self.mem_handles: List[MemHandleMeta] = []
        self._uuid_to_handle: Dict[str, MemHandleMeta] = {}
        for buf in buffers:
            # We expose every (uuid, page_index) -> address mapping to the
            # remote side, so the buffer MUST be cleanly tileable by
            # ``align_bytes``. A misaligned trailing region would otherwise
            # advertise an addr that crosses the buffer boundary and silently
            # let the peer write past it.
            if buf.align_bytes <= 0:
                raise ValueError(
                    f"BufferConfig.align_bytes must be positive (got "
                    f"{buf.align_bytes})"
                )
            if buf.size % buf.align_bytes != 0:
                raise ValueError(
                    f"BufferConfig: size {buf.size} must be a multiple of "
                    f"align_bytes {buf.align_bytes} "
                    f"({buf.size % buf.align_bytes} bytes overrun)"
                )
            buffer_addrs = [
                base for base in range(buf.ptr, buf.ptr + buf.size, buf.align_bytes)
            ]
            meta = MemHandleMeta(
                mem_handle=None,
                buffer_ptr=buf.ptr,
                buffer_size=buf.size,
                page_size=buf.align_bytes,
                local_buffer_addrs=buffer_addrs,
                buffer_type=buf.device_type,
            )
            self.mem_handles.append(meta)
            self._uuid_to_handle[meta.uuid] = meta

        # ----- per-peer state -----
        self.running = True
        self._state_lock = threading.Lock()
        # Channel-wide serializer for every Send/Recv/Scatter call on this
        # device. The agent's shared input/output regions are not safe for
        # concurrent use across peers.
        self._transfer_lock = threading.Lock()
        # Bidirectional traffic on the same channel needs *two* per-peer
        # dictionaries, not one. A single dict keyed by the remote process'
        # id would collide because both roles end up using the same key:
        #   - the *connector* role (we initiated the BatchChannel connect to
        #     the remote, so we hold a live ``transfer_req_socket`` and a
        #     ``conn_handle`` from ``agent.connect()``);
        #   - the *listener* role (the remote initiated to us, so we hold a
        #     ``conn_handle`` from ``agent.accept()`` and have
        #     ``transfer_req_socket=None`` because we never originate
        #     transfer requests in this direction — we only receive them).
        # Under DP-induced bidirectional pulls the second-inserted entry
        # would silently overwrite the first, and any later transfer in the
        # losing direction would crash on ``peer.transfer_req_socket.send``
        # (AttributeError on None). Keeping the two roles in separate dicts
        # is the simplest way to keep both peer states alive simultaneously.
        self._connector_peers: Dict[str, _PeerState] = {}
        self._listener_peers: Dict[str, _PeerState] = {}
        # Per-peer handshake serialization for the async path. Mirrors the
        # treatment ``HcclChannel`` got via PR #234: with the p2p_sync
        # ``batched_get_blocking`` path fanning blocking ``ensure_peer_connection``
        # hops onto the P2P loop, several coroutines can race the
        # ``_connector_peers`` check for the same peer, double-fire the init
        # handshake, and call ``self.agent.connect`` twice on the same NPU
        # device. A per-peer ``asyncio.Lock`` keeps each peer's handshake to
        # one in-flight coroutine without blocking other peers.
        self._peer_handshake_locks: Dict[str, asyncio.Lock] = {}
        self.side_channels: List[zmq.Socket] = []
        self.running_threads: List[threading.Thread] = []

        # Channel's own id, used to populate PingPong*Request.receiver_id so
        # the remote can look us up in its peer table. Captured on the first
        # lazy_init_peer_connection call (or supplied via kwargs).
        self.local_id: Optional[str] = kwargs.get("local_id")

        # ----- ZMQ context + init/transfer URLs -----
        self.async_mode = async_mode
        self.zmq_context = get_zmq_context(use_asyncio=async_mode)
        self.event_loop = kwargs.get("event_loop", None)

        # Both the init REP and transfer REP sockets are bound EAGERLY on the
        # foreground thread:
        #   - We want the resolved tcp://<ip>:<port> available before any
        #     worker / loop / consumer runs (so init responses can carry a
        #     real transfer URL, and the parent process can publish our init
        #     URL via a shared dict for the connector to look up).
        #   - bind() itself is synchronous on both zmq.Context and
        #     zmq.asyncio.Context instances, so this works regardless of
        #     ``async_mode``. Only the subsequent recv/send loops touch the
        #     async APIs.
        # get_zmq_socket() prepends "tcp://"; pass host:port only.
        peer_init_url = kwargs["peer_init_url"]
        if isinstance(peer_init_url, str) and peer_init_url.startswith("tcp://"):
            peer_init_url = peer_init_url[len("tcp://") :]
        self.peer_init_url = peer_init_url
        # Required by the upstream abstract.handle_init_side_msg path:
        # when the p2p_backend follows up the pingpong handshake with an
        # empty ``P2PInitSideMsg``, the listener must respond with a
        # ``P2PInitSideRetMsg(peer_lookup_url=self.peer_lookup_url)``. Other
        # channels (nixl, py_socket) read this from kwargs the same way;
        # omitting it makes the assertion at abstract.py:83 fire and falls
        # back to ``_make_error_init_response``, which the connector then
        # fails to decode as ``SideMsg``.
        self.peer_lookup_url = kwargs.get("peer_lookup_url", None)
        self._init_socket = get_zmq_socket(
            self.zmq_context, self.peer_init_url, "tcp", zmq.REP, "bind"
        )
        self.side_channels.append(self._init_socket)
        # peer_init_url_resolved is the host:port a remote process should
        # connect to. Identical to peer_init_url for an explicit bind, but
        # rewrites the wildcard 0.0.0.0[:0] form into the actual listening
        # interface/port (and replaces 0.0.0.0 with a routable host).
        init_last_endpoint = self._init_socket.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(init_last_endpoint, bytes):
            init_last_endpoint = init_last_endpoint.decode("utf-8")
        self.peer_init_url_resolved = self._resolve_transfer_url(
            init_last_endpoint, kwargs
        )

        _pp_bind = kwargs.get("pp_transfer_bind_addr", "0.0.0.0:0")
        if isinstance(_pp_bind, str) and _pp_bind.startswith("tcp://"):
            _pp_bind = _pp_bind[len("tcp://") :]
        self.transfer_bind_addr = _pp_bind
        # The transfer REP socket is consumed exclusively by the synchronous
        # ``_transfer_loop`` daemon thread (calls ``recv()`` / ``send()``
        # directly). When ``async_mode=True`` (e.g. ``AscendP2PBackend``),
        # ``self.zmq_context`` is a ``zmq.asyncio.Context`` and binding a REP
        # socket on it would yield a ``zmq.asyncio.Socket`` whose blocking
        # ``recv()`` is not safe to call from a non-loop thread. Use a
        # dedicated sync context for the transfer socket so it can be polled
        # from the daemon thread regardless of the channel's async mode.
        # Mirrors the per-peer transfer REQ socket setup in ``_register_peer``,
        # which already grabs its own sync context.
        self._transfer_zmq_context = get_zmq_context(use_asyncio=False)
        self._transfer_socket = get_zmq_socket(
            self._transfer_zmq_context,
            self.transfer_bind_addr,
            "tcp",
            zmq.REP,
            "bind",
        )
        self.side_channels.append(self._transfer_socket)
        last_endpoint = self._transfer_socket.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(last_endpoint, bytes):
            last_endpoint = last_endpoint.decode("utf-8")
        self.transfer_url = self._resolve_transfer_url(last_endpoint, kwargs)

        # Outbound-pull stream: every connector-side recv_batch /
        # scatter_recv (batched_read, async_batched_read,
        # submit_batched_read, async_batched_scatter) is queued here so
        # the caller can pipeline reads behind a recorded event (see
        # ``NPUConnector._batched_to_gpu_proxy`` which calls
        # ``transport_stream.wait_event(...)``).
        self.transport_stream = torch.npu.Stream(torch.npu.current_device())
        # Inbound-serve stream: listener-side send_batch / scatter_send
        # MUST run on a separate stream from ``transport_stream``.
        # Reusing one stream for both directions deadlocks under
        # symmetric cross-rank pulls: each side's listener-side
        # ``send_batch`` is FIFO-queued behind its OWN main-thread
        # ``recv_batch``, that ``recv_batch`` can only complete when
        # the peer's ``send_batch`` drains, and the peer's
        # ``send_batch`` is similarly blocked behind its own pending
        # recv. Symptom (DP=2, TP=4 Qwen3-30B): DPx_TPi and DPy_TPi
        # both wedge in ``_wait_for_transfer_ack`` while the other 6
        # workers stall at the next TP-AllReduce. Splitting the
        # streams keeps the two FIFOs independent so inbound sends
        # always drain regardless of pending outbound recvs.
        self._send_stream = torch.npu.Stream(torch.npu.current_device())

        self._init_side_channels()
        self._start_transfer_worker()

    # ------------------------------------------------------------------
    # init / transfer URL helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_transfer_url(bound_url: str, kwargs: dict) -> str:
        """Normalize the bound transfer URL into a bare ``host:port`` form.

        ZMQ's ``LAST_ENDPOINT`` returns a URL prefixed with the scheme
        (``tcp://host:port``), but everywhere downstream we feed this string
        back into :func:`get_zmq_socket`, which prepends ``tcp://`` itself.
        Returning the raw ``LAST_ENDPOINT`` results in a doubled scheme
        (``tcp://tcp://host:port``) and a ZMQ ``Invalid argument`` error.

        We also rewrite a wildcard ``0.0.0.0`` bind into a routable host: the
        explicit ``pp_advertised_host`` kwarg if provided, otherwise the local
        IP returned by :func:`get_ip` (which itself falls back to
        ``127.0.0.1``). Without this rewrite the URL we publish over the init
        handshake is not reachable from a peer process.
        """
        url = bound_url
        if url.startswith("tcp://"):
            url = url[len("tcp://") :]
        if url.startswith("0.0.0.0"):
            advertised_host: Optional[str] = kwargs.get("pp_advertised_host")
            if not advertised_host:
                advertised_host = get_ip()
            url = url.replace("0.0.0.0", advertised_host, 1)
        return url

    def _init_side_channels(self):
        if self.peer_init_url is None:
            raise ValueError("Peer init URL is not set")

        if self.async_mode:
            asyncio.run_coroutine_threadsafe(self._async_init_loop(), self.event_loop)
        else:
            t = threading.Thread(target=self._init_loop, daemon=True)
            t.start()
            self.running_threads.append(t)

    def _start_transfer_worker(self):
        t = threading.Thread(target=self._transfer_loop, daemon=True)
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

    # ------------------------------------------------------------------
    # Init handshake (REP socket on listener / REQ on connector)
    # ------------------------------------------------------------------
    def _capture_local_id(self, local_id: str) -> None:
        if self.local_id is None:
            self.local_id = local_id
        elif self.local_id != local_id:
            raise ValueError(
                f"HcclPingPongChannel: local_id mismatch (have={self.local_id!r}, "
                f"got={local_id!r}). Reusing one channel across multiple local_ids "
                f"is not supported."
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
            # Only the connector dict matters for "have I already initiated
            # to this peer?". A pre-existing entry in ``_listener_peers``
            # means they initiated to us, which does NOT give us a
            # transfer_req_socket, so we must still run our own handshake.
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
            req = PingPongInitRequest(
                local_id=local_id,
                client_meta_bytes=pickle.dumps(self.agent.get_client_meta()),
                buffer_infos=self._make_buffer_infos(),
            )
            init_tmp_socket.send(msgspec.msgpack.encode(req))
            resp_bytes = init_tmp_socket.recv()
            resp = msgspec.msgpack.decode(resp_bytes, type=PingPongMsg)
            if not isinstance(resp, PingPongInitResponse) or not resp.server_meta_bytes:
                raise ConnectionError(
                    f"PingPong init handshake failed for peer {peer_id}: "
                    f"{type(resp).__name__}"
                )
            server_meta = pickle.loads(resp.server_meta_bytes)

            conn_handle = self.agent.connect(server_meta)
            self._register_peer(peer_id, conn_handle, resp)

            if init_side_msg is not None:
                return self.send_init_side_msg(init_tmp_socket, init_side_msg)
            return None
        finally:
            init_tmp_socket.close()

    def _get_peer_handshake_lock(self, peer_id: str) -> asyncio.Lock:
        # The lock dict itself only mutates on the loop thread (every caller
        # of ``async_lazy_init_peer_connection`` already runs on the P2P
        # loop), so a plain check/insert is safe without ``_state_lock``.
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
        # Serialize per-peer handshakes so concurrent ``_ensure_peer_connection``
        # fan-out (notably from the new sync ``batched_get_blocking`` bridge
        # in ``AscendP2PBackend``) cannot race the ``_connector_peers`` check
        # and call ``self.agent.connect`` twice on the same NPU device.
        # Mirrors ``HcclChannel.async_lazy_init_peer_connection`` (PR #234).
        async with self._get_peer_handshake_lock(peer_id):
            return await self._async_lazy_init_peer_connection_locked(
                local_id,
                peer_id,
                peer_init_url,
                init_side_msg,
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
            # See sync variant: connector dict only.
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
            req = PingPongInitRequest(
                local_id=local_id,
                client_meta_bytes=pickle.dumps(self.agent.get_client_meta()),
                buffer_infos=self._make_buffer_infos(),
            )
            await init_tmp_socket.send(msgspec.msgpack.encode(req))
            resp_bytes = await init_tmp_socket.recv()
            resp = msgspec.msgpack.decode(resp_bytes, type=PingPongMsg)
            if not isinstance(resp, PingPongInitResponse) or not resp.server_meta_bytes:
                raise ConnectionError(
                    f"PingPong init handshake failed for peer {peer_id}: "
                    f"{type(resp).__name__}"
                )
            server_meta = pickle.loads(resp.server_meta_bytes)

            loop = asyncio.get_running_loop()
            device = self.handle_device

            def _connect_blocking() -> int:
                torch.npu.set_device(device)
                return self.agent.connect(server_meta)

            conn_handle = await loop.run_in_executor(None, _connect_blocking)
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
        resp: PingPongInitResponse,
    ) -> None:
        # Open a long-lived REQ socket connected to the listener's transfer URL.
        # Sync vs async REQ: the transfer-side socket is always sync because
        # the C++ blocking calls (send_batch / scatter_send) we issue on the
        # other end are GIL-released anyway, and a sync REQ keeps the
        # request/response framing trivial to lock around. The sync context
        # is a process-wide singleton (rpc_utils.get_zmq_context returns
        # ``zmq.Context.instance()`` per use_asyncio mode), so mixing sync
        # transfer sockets with the async init socket here doesn't leak a
        # second context.
        sync_ctx = get_zmq_context(use_asyncio=False)
        transfer_req = get_zmq_socket(
            sync_ctx, resp.transfer_url, "tcp", zmq.REQ, "connect"
        )
        transfer_req.setsockopt(zmq.RCVTIMEO, _TRANSFER_ACK_TIMEOUT_SEC * 1000)
        transfer_req.setsockopt(zmq.SNDTIMEO, _TRANSFER_ACK_TIMEOUT_SEC * 1000)
        peer = _PeerState(
            conn_handle=conn_handle,
            transfer_url=resp.transfer_url,
            transfer_req_socket=transfer_req,
            remote_buffers=RemotePeerBufferList(resp.buffer_infos),
        )
        # Connector-side: ``agent.connect()`` has already returned, so the
        # BatchChannel is fully wired. Publish before the caller can issue
        # a transfer.
        peer.ready_event.set()
        with self._state_lock:
            self._connector_peers[peer_id] = peer

    def remote_xfer_handler_exists(self, receiver_or_sender_id: str) -> bool:
        # Called by the sender-side P2P backend handler to confirm that the
        # receiver has previously initiated the BatchChannel handshake to us
        # (so we can satisfy its lookup-and-get). That handshake lives in
        # ``_listener_peers``; the connector dict tracks the *opposite*
        # direction (us pulling from them) and is irrelevant here.
        with self._state_lock:
            return receiver_or_sender_id in self._listener_peers

    # ------------------------------------------------------------------
    # _init_loop / _async_init_loop
    # ------------------------------------------------------------------
    def _make_error_init_response(self) -> PingPongInitResponse:
        return PingPongInitResponse(
            server_meta_bytes=b"",
            buffer_infos=[],
            transfer_url="",
        )

    def _handle_init_msg(
        self, req: Union[PingPongMsg, InitSideMsgBase]
    ) -> Union[PingPongMsg, InitSideRetMsgBase]:
        if isinstance(req, PingPongInitRequest):
            logger.info("PingPong: init request from %s", req.local_id)
            server_meta = self.agent.get_server_meta()
            client_meta = pickle.loads(req.client_meta_bytes)

            # Reserve the peer entry BEFORE we send the init response. We
            # cannot wait for ``accept()`` to return on this thread without
            # deadlocking: ``accept()`` blocks until the connector calls
            # ``connect()``, and the connector won't call ``connect()`` until
            # it receives our response. So we register a not-yet-ready
            # placeholder, kick off ``accept()`` in the background, and let
            # any transfer request that races ahead of the handshake block on
            # ``ready_event`` via ``_await_peer_ready``.
            peer = _PeerState(
                conn_handle=0,
                transfer_url="",  # listener side: no remote transfer url
                transfer_req_socket=None,
                remote_buffers=RemotePeerBufferList(req.buffer_infos),
            )
            local_id = req.local_id
            with self._state_lock:
                # Note: we intentionally do NOT touch ``_connector_peers``
                # here. If the local channel is also pulling from this same
                # remote (bidirectional traffic under DP), that direction
                # has its own peer state with a live transfer_req_socket
                # which must survive this insert.
                self._listener_peers[local_id] = peer

            accept_started = threading.Event()

            def _complete_handshake():
                torch.npu.set_device(self.handle_device)
                try:
                    accept_started.set()
                    conn_handle = self.agent.accept(client_meta, server_meta)
                    peer.conn_handle = conn_handle
                    logger.info(
                        "PingPong: accepted connection from %s", local_id
                    )
                except Exception as e:
                    peer.handshake_error = e
                    # Drop the failed entry so a subsequent init handshake
                    # from the same local_id is allowed to retry instead of
                    # being short-circuited by ``remote_xfer_handler_exists``.
                    with self._state_lock:
                        if self._listener_peers.get(local_id) is peer:
                            del self._listener_peers[local_id]
                    logger.error(
                        "PingPong handshake failed for %s: %s", local_id, e
                    )
                finally:
                    # MUST come after conn_handle / handshake_error are
                    # written: ``_await_peer_ready`` treats ``ready_event``
                    # firing as the publish barrier.
                    peer.ready_event.set()

            t = threading.Thread(target=_complete_handshake, daemon=True)
            t.start()
            if not accept_started.wait(timeout=10.0):
                raise TimeoutError(
                    "Timed out waiting for handshake thread to start accept()"
                )

            return PingPongInitResponse(
                server_meta_bytes=pickle.dumps(server_meta),
                buffer_infos=self._make_buffer_infos(),
                transfer_url=self.transfer_url,
            )

        if isinstance(req, InitSideMsgBase):
            return self.handle_init_side_msg(req)

        raise ValueError(f"Unsupported init message type: {type(req)}")

    def _init_loop(self):
        # Socket was bound eagerly in __init__ so peer_init_url_resolved is
        # already publishable; the loop just owns the recv/send lifetime.
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
                    req_bytes, type=Union[PingPongMsg, SideMsg]
                )
                resp = self._handle_init_msg(req)
                sock.send(msgspec.msgpack.encode(resp))
            except Exception as e:
                logger.error("PingPong init loop failure: %s", e)
                try:
                    sock.send(
                        msgspec.msgpack.encode(self._make_error_init_response())
                    )
                except Exception:
                    pass
                if self.running:
                    time.sleep(0.01)
        sock.close()

    async def _async_init_loop(self):
        # Socket was bound eagerly in __init__ (see comment there about
        # ``bind()`` being sync on both context flavours).
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
                    req_bytes, type=Union[PingPongMsg, SideMsg]
                )
                resp = await loop.run_in_executor(None, self._handle_init_msg, req)
                await sock.send(msgspec.msgpack.encode(resp))
            except Exception as e:
                logger.error("PingPong async init loop failure: %s", e)
                try:
                    await sock.send(
                        msgspec.msgpack.encode(self._make_error_init_response())
                    )
                except Exception:
                    pass
                if self.running:
                    await asyncio.sleep(0.01)
        sock.close()

    # ------------------------------------------------------------------
    # Transfer worker (REP) — serves remote read / scatter requests
    # ------------------------------------------------------------------
    def _transfer_loop(self):
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
            try:
                req = msgspec.msgpack.decode(req_bytes, type=PingPongMsg)
            except Exception as e:
                logger.error("PingPong transfer decode failure: %s", e)
                self._send_transfer_ack(
                    sock, PingPongReadAck(ok=False, error=f"decode_error:{e}")
                )
                continue

            try:
                if isinstance(req, PingPongReadRequest):
                    self._handle_read_request(req)
                    self._send_transfer_ack(sock, PingPongReadAck(ok=True))
                elif isinstance(req, PingPongScatterRequest):
                    self._handle_scatter_request(req)
                    self._send_transfer_ack(sock, PingPongScatterAck(ok=True))
                else:
                    raise ValueError(
                        f"Unsupported transfer message: {type(req).__name__}"
                    )
            except Exception as e:
                logger.error("PingPong transfer dispatch failure: %s", e)
                if isinstance(req, PingPongScatterRequest):
                    ack: Union[PingPongReadAck, PingPongScatterAck] = (
                        PingPongScatterAck(ok=False, error=str(e))
                    )
                else:
                    ack = PingPongReadAck(ok=False, error=str(e))
                self._send_transfer_ack(sock, ack)

        sock.close()

    @staticmethod
    def _send_transfer_ack(
        sock: zmq.Socket, ack: Union[PingPongReadAck, PingPongScatterAck]
    ) -> None:
        try:
            sock.send(msgspec.msgpack.encode(ack))
        except Exception as e:
            logger.error("PingPong: failed to send transfer ack: %s", e)

    def _handle_read_request(self, req: PingPongReadRequest) -> None:
        if len(req.sender_local_addrs) != len(req.sizes):
            raise ValueError("PingPongReadRequest sizes/addrs length mismatch")
        # We are the sender side here — the receiver (req.receiver_id)
        # initiated the BatchChannel handshake to us, so its peer entry
        # lives in the listener registry.
        t0 = time.monotonic()
        n_chunks = len(req.sender_local_addrs)
        try:
            peer = self._await_peer_ready(req.receiver_id, role="listener")
        except Exception as e:
            logger.error(
                "PingPong sender: _await_peer_ready FAILED receiver_id=%s "
                "chunks=%d await_ms=%.1f exc=%r. ok=False will be sent - "
                "receiver stream will be POISONED on its device.",
                req.receiver_id,
                n_chunks,
                (time.monotonic() - t0) * 1000,
                e,
            )
            raise
        t1 = time.monotonic()
        conn_handle = peer.conn_handle

        ops = [
            hpp.PingPongOp(local_addr=addr, size=size)
            for addr, size in zip(
                req.sender_local_addrs, req.sizes, strict=True
            )
        ]
        # Submit send_batch under ``_transfer_lock`` (briefly serializes
        # access to the shared ``agent`` object), then release the lock
        # BEFORE blocking on ``_send_stream.synchronize()``. Mirrors the
        # connector-side discipline in ``batched_read`` — holding the
        # lock through synchronize would block the main thread from
        # acquiring it to issue a concurrent ``recv_batch``, defeating
        # half the point of splitting send/recv onto separate streams.
        with self._transfer_lock:
            t2 = time.monotonic()
            try:
                self.agent.send_batch(
                    conn_handle, ops, self._send_stream.npu_stream
                )
            except Exception as e:
                logger.error(
                    "PingPong sender: agent.send_batch FAILED receiver_id=%s "
                    "chunks=%d exc=%r. ok=False will be sent - receiver "
                    "stream will be POISONED on its device.",
                    req.receiver_id,
                    n_chunks,
                    e,
                )
                raise
            t3 = time.monotonic()
        self._send_stream.synchronize()
        t4 = time.monotonic()
        self._maybe_log_sender_phases(
            "read", req.receiver_id, n_chunks, t0, t1, t2, t3, t4
        )

    def _handle_scatter_request(self, req: PingPongScatterRequest) -> None:
        # Sender side: see ``_handle_read_request`` for the role rationale.
        t0 = time.monotonic()
        n_chunks = sum(len(e.counts) for e in req.entries)
        try:
            peer = self._await_peer_ready(req.receiver_id, role="listener")
        except Exception as e:
            logger.error(
                "PingPong sender: _await_peer_ready FAILED (scatter) "
                "receiver_id=%s chunks=%d await_ms=%.1f exc=%r. ok=False "
                "will be sent - receiver stream will be POISONED on its "
                "device.",
                req.receiver_id,
                n_chunks,
                (time.monotonic() - t0) * 1000,
                e,
            )
            raise
        t1 = time.monotonic()
        conn_handle = peer.conn_handle

        entries = []
        for entry in req.entries:
            cpp_entry = hpp.PingPongScatterEntry()
            cpp_entry.ddr_buf = entry.sender_local_addr
            cpp_entry.dst_bufs = []  # sender side: empty
            cpp_entry.counts = list(entry.counts)
            cpp_entry.data_type = hpp.HcclDataType(entry.data_type)
            entries.append(cpp_entry)

        # See ``_handle_read_request`` for the lock/stream discipline.
        with self._transfer_lock:
            t2 = time.monotonic()
            try:
                self.agent.scatter_send(
                    conn_handle, entries, self._send_stream.npu_stream
                )
            except Exception as e:
                logger.error(
                    "PingPong sender: agent.scatter_send FAILED "
                    "receiver_id=%s chunks=%d exc=%r. ok=False will be "
                    "sent - receiver stream will be POISONED on its "
                    "device.",
                    req.receiver_id,
                    n_chunks,
                    e,
                )
                raise
            t3 = time.monotonic()
        self._send_stream.synchronize()
        t4 = time.monotonic()
        self._maybe_log_sender_phases(
            "scatter", req.receiver_id, n_chunks, t0, t1, t2, t3, t4
        )

    # ------------------------------------------------------------------
    # Local helpers
    # ------------------------------------------------------------------
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
        buffer_uuids: List[str] = []
        mem_indexes: List[int] = []
        if isinstance(objects[0], MemoryObj):
            for mem_obj in objects:
                assert isinstance(mem_obj, MemoryObj)
                buf_uuid, mem_idx = resolve_buffer_ref(
                    self.mem_handles, mem_obj.data_ptr, mem_obj.meta.address
                )
                buffer_uuids.append(buf_uuid)
                mem_indexes.append(mem_idx)
            return buffer_uuids, mem_indexes
        raise NotImplementedError("Sending raw bytes is not supported")

    def _get_local_addr(self, ptr: int, idx: int) -> int:
        return resolve_local_addr(self.mem_handles, ptr, idx)

    def _resolve_remote_addrs(self, transfer_spec: dict) -> List[int]:
        if (
            TS_REMOTE_BUFFER_UUIDS in transfer_spec
            and TS_REMOTE_MEM_INDEXES in transfer_spec
        ):
            peer_id = transfer_spec[TS_RECEIVER_ID]
            with self._state_lock:
                # Only the connector-side entry carries the remote sender's
                # buffer descriptors that the receiver-driven pull needs.
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

    # Default wait when a transfer-time lookup races the listener-side
    # ``accept()``. Generous because ``accept()`` does a full HCCL transport
    # init (NIC setup, NotifyPool allocation, etc.) on first use.
    _PEER_READY_TIMEOUT_SEC = 30.0

    def _await_peer_ready(
        self,
        peer_id: str,
        timeout: float = _PEER_READY_TIMEOUT_SEC,
        role: str = "connector",
    ) -> _PeerState:
        """Look up ``peer_id`` in the dict for ``role`` and block until its
        BatchChannel handshake has completed (or surface the handshake error).

        ``role`` must be either ``"connector"`` (the caller wants the peer
        entry we created when WE initiated the BatchChannel handshake to
        the remote — has a live ``transfer_req_socket``) or ``"listener"``
        (the caller is a sender-side handler servicing a transfer request
        from a remote that initiated to us — no ``transfer_req_socket``).
        Passing the wrong role will surface as either an ``AttributeError``
        on a ``None`` socket (connector op on a listener entry) or a
        ``KeyError`` (the directions just didn't share a peer at the time
        of the call).

        Closes the race where the listener-side ``_handle_init_msg`` has
        already sent ``PingPongInitResponse`` (so the connector now knows
        our transfer URL) but the background ``accept()`` thread hasn't
        populated ``conn_handle`` yet. Without this wait, a transfer
        request landing on our transfer worker between those two events
        would call ``agent.send_batch`` with a zero ``conn_handle``.

        Connector-side peers are created already-ready by
        ``_register_peer``, so this is effectively a same-thread
        ``Event.is_set()`` check on that path.
        """
        if role == "connector":
            registry = self._connector_peers
        elif role == "listener":
            registry = self._listener_peers
        else:
            raise ValueError(
                f"_await_peer_ready: unknown role {role!r}; "
                "expected 'connector' or 'listener'"
            )
        with self._state_lock:
            peer = registry.get(peer_id)
        if peer is None:
            raise KeyError(
                f"Unknown peer {peer_id!r} for role {role!r}"
            )
        if not peer.ready_event.wait(timeout=timeout):
            raise TimeoutError(
                f"PingPong: peer {peer_id!r} handshake did not complete "
                f"within {timeout}s"
            )
        if peer.handshake_error is not None:
            raise RuntimeError(
                f"PingPong: peer {peer_id!r} handshake failed: "
                f"{peer.handshake_error}"
            )
        return peer

    # ------------------------------------------------------------------
    # Send/Recv: not supported (use read / scatter instead)
    # ------------------------------------------------------------------
    def batched_send(self, *args, **kwargs):
        raise NotImplementedError("HcclPingPongChannel uses receiver-driven reads")

    def batched_recv(self, *args, **kwargs):
        raise NotImplementedError("HcclPingPongChannel uses receiver-driven reads")

    async def async_batched_send(self, *args, **kwargs):
        raise NotImplementedError("HcclPingPongChannel uses receiver-driven reads")

    async def async_batched_recv(self, *args, **kwargs):
        raise NotImplementedError("HcclPingPongChannel uses receiver-driven reads")

    def batched_write(self, *args, **kwargs):
        raise NotImplementedError("HcclPingPongChannel does not support write")

    async def async_batched_write(self, *args, **kwargs):
        raise NotImplementedError("HcclPingPongChannel does not support write")

    # ------------------------------------------------------------------
    # batched_read / async_batched_read
    # ------------------------------------------------------------------
    def _build_read_plan(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: dict,
    ) -> Tuple[_PeerState, List[int], List[int], List[int]]:
        peer_id = transfer_spec[TS_RECEIVER_ID]
        # batched_read / submit_batched_read are the receiver-driven pull
        # path: we are the connector to the remote sender, so we need the
        # peer entry that carries our ``transfer_req_socket``.
        peer = self._await_peer_ready(peer_id, role="connector")
        sender_local_addrs = self._resolve_remote_addrs(transfer_spec)

        local_dst_addrs: List[int] = []
        sizes: List[int] = []
        for mem_obj in buffers:
            if not isinstance(mem_obj, MemoryObj):
                raise NotImplementedError(
                    "Sending raw bytes is not supported in HcclPingPongChannel"
                )
            local_dst_addrs.append(
                self._get_local_addr(mem_obj.data_ptr, mem_obj.meta.address)
            )
            sizes.append(self.page_size)
        if len(local_dst_addrs) != len(sender_local_addrs):
            raise ValueError(
                f"local dst count ({len(local_dst_addrs)}) != sender addrs "
                f"count ({len(sender_local_addrs)})"
            )
        return peer, sender_local_addrs, sizes, local_dst_addrs

    def batched_read(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        if not buffers:
            return 0
        peer, sender_local_addrs, sizes, local_dst_addrs = self._build_read_plan(
            buffers, transfer_spec
        )
        receiver_id = self._infer_receiver_id_for_request(transfer_spec)

        req = PingPongReadRequest(
            receiver_id=receiver_id,
            sender_local_addrs=sender_local_addrs,
            sizes=sizes,
        )

        ops = [
            hpp.PingPongOp(local_addr=addr, size=size)
            for addr, size in zip(local_dst_addrs, sizes, strict=True)
        ]
        # Lock acquisition order is peer.transfer_req_lock -> _transfer_lock:
        # the per-peer lock serializes REQ/REP framing on this socket (only
        # one outstanding request at a time), the channel-wide lock guards
        # the agent's shared input region during recv_batch submission. We
        # MUST release ``_transfer_lock`` before blocking on the sender's
        # ack -- if we hold it through the ack wait, the channel's own
        # ``_transfer_loop`` cannot acquire it to service an incoming
        # ``PingPongReadRequest`` from the symmetric peer, and under DP
        # bidirectional cross-rank pulls A and B mutually deadlock waiting
        # for each other's ack.
        t0 = time.monotonic()
        n_chunks = len(ops)
        with peer.transfer_req_lock:
            t1 = time.monotonic()
            with self._transfer_lock:
                t2 = time.monotonic()
                peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                t3 = time.monotonic()
                # Submit the local recv concurrently with the remote send so
                # ping-pong wraparound (which needs receiver-side recvReady
                # credits) can make progress.
                self.agent.recv_batch(
                    peer.conn_handle, ops, self.transport_stream.npu_stream
                )
                t4 = time.monotonic()
            # NOBLOCK-poll the REQ socket. We must NOT pre-emptively
            # synchronize() the stream here -- if the sender rejects the
            # request it will never post the matching sendDoneSlot
            # notifies and synchronize() would deadlock. The poll runs
            # WITHOUT _transfer_lock so the listener side stays responsive.
            ack_bytes = self._wait_for_transfer_ack(peer)
            t5 = time.monotonic()
        ack = msgspec.msgpack.decode(ack_bytes, type=PingPongMsg)
        self._maybe_log_ack_failure(
            ack, peer, n_chunks, "batched_read", t0, t1, t2, t3, t4, t5
        )
        # Raises on ok=False, leaving the stream queue intact for the
        # HCCL transport timeout to drain (see poisoned-streams note in
        # _wait_for_transfer_ack).
        self._raise_on_ack_failure(ack)
        # Happy path only: the sender synchronized its own stream
        # BEFORE sending the ok ack, so every sendDoneSlot Post is
        # already on the wire. Our queued Waits clear in microseconds
        # and synchronize() returns immediately once the receiver-side
        # memcpys finish populating the user buffer. Listener-side
        # send_batch lives on ``_send_stream`` (NOT this stream), so
        # there is no cross-direction FIFO interference — see the
        # stream-split rationale in ``__init__``.
        t6 = time.monotonic()
        self.transport_stream.synchronize()
        t7 = time.monotonic()
        self._maybe_log_transfer_phases(
            "batched_read", peer, n_chunks,
            t0, t1, t2, t3, t4, t5, t6, t7,
        )
        return len(buffers)

    async def async_batched_read(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        assert transfer_spec is not None
        if not buffers:
            return 0
        peer, sender_local_addrs, sizes, local_dst_addrs = self._build_read_plan(
            buffers, transfer_spec
        )
        receiver_id = self._infer_receiver_id_for_request(transfer_spec)

        req = PingPongReadRequest(
            receiver_id=receiver_id,
            sender_local_addrs=sender_local_addrs,
            sizes=sizes,
        )

        ops = [
            hpp.PingPongOp(local_addr=addr, size=size)
            for addr, size in zip(local_dst_addrs, sizes, strict=True)
        ]

        loop = asyncio.get_running_loop()

        def _run_blocking() -> int:
            # run_in_executor dispatches onto a default ThreadPoolExecutor
            # worker that has NO NPU device set; without this call HCCL
            # transport ops fail synchronously with HCCL_E_RUNTIME because
            # the current ACL device id doesn't match transport_stream's
            # device. Mirrors async_lazy_init_peer_connection._connect_blocking.
            torch.npu.set_device(self.handle_device)
            # See ``batched_read`` for the locking rationale: ack-wait must
            # happen WITHOUT ``_transfer_lock`` so the listener side can
            # service symmetric peer reads concurrently.
            t0 = time.monotonic()
            n_chunks = len(ops)
            with peer.transfer_req_lock:
                t1 = time.monotonic()
                with self._transfer_lock:
                    t2 = time.monotonic()
                    peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                    t3 = time.monotonic()
                    self.agent.recv_batch(
                        peer.conn_handle, ops, self.transport_stream.npu_stream
                    )
                    t4 = time.monotonic()
                ack_bytes = self._wait_for_transfer_ack(peer)
                t5 = time.monotonic()
            ack = msgspec.msgpack.decode(ack_bytes, type=PingPongMsg)
            self._maybe_log_ack_failure(
                ack, peer, n_chunks, "async_batched_read",
                t0, t1, t2, t3, t4, t5,
            )
            self._raise_on_ack_failure(ack)
            # Success: drain our stream so the user buffer is populated
            # by the time this coroutine resumes. Listener-side send_batch
            # lives on ``_send_stream``, so no cross-direction FIFO
            # interference here (see ``__init__`` for the stream split
            # rationale).
            t6 = time.monotonic()
            self.transport_stream.synchronize()
            t7 = time.monotonic()
            self._maybe_log_transfer_phases(
                "async_batched_read", peer, n_chunks,
                t0, t1, t2, t3, t4, t5, t6, t7,
            )
            return len(buffers)

        return await loop.run_in_executor(None, _run_blocking)

    def submit_batched_read(
        self,
        buffers: Union[List[bytes], List[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> Optional[torch.npu.Event]:
        """Submit a batched read without draining ``transport_stream``.

        Mirrors the contract of :meth:`HcclChannel.submit_batched_read` /
        :meth:`HcommOneSidedChannel.submit_batched_read` so that
        :meth:`ProxyMemoryObj.submit_resolve_batch` can take the pipelined
        path (instead of falling back to the synchronous ``batched_read``)
        and the NPU connector can overlap RDMA reads with the previous
        micro-batch's scatter via ``load_stream.wait_event(event)``.

        Behaviour differs from the HCCL/HCOMM variants in one important way:
        ping-pong is receiver-driven over a control REQ/REP plane, so we
        must wait for the sender's ``PingPongReadAck`` before returning.
        We block on the ack (under the same locks as ``batched_read``) but
        deliberately skip ``transport_stream.synchronize()``; instead we
        record an :class:`torch.npu.Event` on ``transport_stream`` and let
        the caller schedule a cross-stream wait. Returning before the ack
        would let the caller ``wait_event`` on a stream that may still
        carry the remote-rejection sequence (see the ``ok=False`` /
        poisoned-stream notes in :meth:`_wait_for_transfer_ack`).
        """
        assert transfer_spec is not None
        if not buffers:
            return None
        peer, sender_local_addrs, sizes, local_dst_addrs = self._build_read_plan(
            buffers, transfer_spec
        )
        receiver_id = self._infer_receiver_id_for_request(transfer_spec)

        req = PingPongReadRequest(
            receiver_id=receiver_id,
            sender_local_addrs=sender_local_addrs,
            sizes=sizes,
        )

        ops = [
            hpp.PingPongOp(local_addr=addr, size=size)
            for addr, size in zip(local_dst_addrs, sizes, strict=True)
        ]

        # See ``batched_read`` for the full locking rationale. The event
        # is recorded under ``_transfer_lock`` so it captures exactly our
        # queued recv ops (listener-side ``send_batch`` lives on
        # ``_send_stream`` and cannot race onto this stream — the lock
        # only protects shared ``agent`` state during submission). The
        # ack wait must happen WITHOUT ``_transfer_lock`` to avoid the
        # bidirectional pull deadlock against symmetric peer reads.
        with peer.transfer_req_lock:
            with self._transfer_lock:
                peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                self.agent.recv_batch(
                    peer.conn_handle, ops, self.transport_stream.npu_stream
                )
                # Happy path: the sender has already synchronized its own
                # stream before sending ``ok=True``, so every
                # ``sendDoneSlot`` notify is on the wire and our queued
                # ``recv_batch`` waits will clear in microseconds. Record
                # an event the caller can wait on instead of blocking the
                # current thread on ``synchronize()``.
                event = torch.npu.Event()
                event.record(self.transport_stream)
            ack_bytes = self._wait_for_transfer_ack(peer)
        ack = msgspec.msgpack.decode(ack_bytes, type=PingPongMsg)
        self._raise_on_ack_failure(ack)
        return event

    @staticmethod
    def _raise_on_ack_failure(ack: PingPongMsg) -> None:
        if isinstance(ack, PingPongReadAck) or isinstance(ack, PingPongScatterAck):
            if not ack.ok:
                raise RuntimeError(f"PingPong remote returned error: {ack.error}")
        else:
            raise RuntimeError(
                f"Unexpected transfer ack type: {type(ack).__name__}"
            )

    # ------------------------------------------------------------------
    # Diagnostic helpers (puller-side phase timing).
    #
    # These exist so the three transfer entry points (``batched_read``,
    # ``async_batched_read``, ``async_batched_scatter``) emit logs in a
    # uniform, greppable shape. They are deliberately log-only with no
    # control-flow effect; the eventual fix plan keeps the
    # poisoning-detected line and removes the rest once we've identified
    # the trigger.
    # ------------------------------------------------------------------
    @classmethod
    def _maybe_log_poisoning(
        cls,
        peer: _PeerState,
        device_id: int,
        ack_error: str,
    ) -> None:
        if cls._poisoned_flag_logged:
            return
        cls._poisoned_flag_logged = True
        logger.error(
            "PingPong: STREAM POISONING DETECTED on device %d (peer=%s). "
            "First ok=False ack observed: %r. Every subsequent transfer on "
            "this device will queue behind ~10s/chunk of phantom Wait drain "
            "(see para.timeout in csrc/hccl_pingpong/batch_channel.cc) until "
            "the device is reset. Expect cascading sync-get timeouts.",
            device_id,
            peer.transfer_url,
            ack_error,
        )

    def _maybe_log_ack_failure(
        self,
        ack: PingPongMsg,
        peer: _PeerState,
        n_chunks: int,
        op: str,
        t0: float,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
        t5: float,
    ) -> None:
        if isinstance(ack, (PingPongReadAck, PingPongScatterAck)) and not ack.ok:
            err = ack.error or "<no error>"
            logger.error(
                "PingPong puller %s ok=False peer=%s chunks=%d "
                "phase_ms[lock1=%.1f lock2=%.1f zmq_send=%.1f recv_submit=%.1f "
                "lock2_rel+ack_wait=%.1f] error=%r",
                op,
                peer.transfer_url,
                n_chunks,
                (t1 - t0) * 1000,
                (t2 - t1) * 1000,
                (t3 - t2) * 1000,
                (t4 - t3) * 1000,
                (t5 - t4) * 1000,
                err,
            )
            self._maybe_log_poisoning(peer, self.handle_device, err)

    @staticmethod
    def _maybe_log_sender_phases(
        op: str,
        receiver_id: str,
        n_chunks: int,
        t0: float,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
    ) -> None:
        total_ms = (t4 - t0) * 1000
        # Same thresholds as the puller path. Healthy sends complete in
        # microseconds; anything over 100ms total is worth surfacing.
        if total_ms <= 100.0:
            return
        logger.warning(
            "PingPong sender %s SLOW receiver_id=%s chunks=%d total_ms=%.1f "
            "phase_ms[await=%.1f lock_acq=%.1f send_submit=%.1f sync=%.1f]",
            op,
            receiver_id,
            n_chunks,
            total_ms,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t4 - t3) * 1000,
        )

    @staticmethod
    def _maybe_log_transfer_phases(
        op: str,
        peer: _PeerState,
        n_chunks: int,
        t0: float,
        t1: float,
        t2: float,
        t3: float,
        t4: float,
        t5: float,
        t6: float,
        t7: float,
    ) -> None:
        total_ms = (t7 - t0) * 1000
        sync_ms = (t7 - t6) * 1000
        # Skip the common fast path (sub-100ms total, sub-50ms sync).
        # These bounds are intentionally tight so any cascade-triggering
        # slowness surfaces while a healthy DP=8 ring stays quiet.
        if total_ms <= 100.0 and sync_ms <= 50.0:
            return
        logger.warning(
            "PingPong puller %s SLOW peer=%s chunks=%d total_ms=%.1f "
            "phase_ms[lock1=%.1f lock2=%.1f zmq_send=%.1f recv_submit=%.1f "
            "lock2_rel+ack_wait=%.1f sync=%.1f]",
            op,
            peer.transfer_url,
            n_chunks,
            total_ms,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t4 - t3) * 1000,
            (t5 - t4) * 1000,
            sync_ms,
        )

    def _wait_for_transfer_ack(
        self,
        peer: _PeerState,
        deadline_sec: float = _TRANSFER_ACK_TIMEOUT_SEC,
    ) -> bytes:
        """Non-blocking poll on the per-peer transfer REQ socket until the
        sender's ``PingPong{Read,Scatter}Ack`` arrives, bounded by
        ``deadline_sec``.

        Why poll instead of plain blocking ``recv()``:

        * Happy path -- sender posts every chunk's ``sendDoneSlot``,
          synchronizes its own stream, then sends ``ok=True``. The ack
          arrives shortly after the matching notify traffic so we return
          quickly and the caller can safely run
          ``transport_stream.synchronize()`` knowing the queued waits
          will clear in microseconds.

        * Failure path -- sender rejects the request (e.g. unknown
          ``receiver_id``) and sends ``ok=False`` WITHOUT posting any
          ``sendDoneSlot``. A naive ``stream.synchronize()`` would
          deadlock waiting for those posts; polling the REQ socket
          surfaces the error ack promptly so the caller can raise.

        * Sender-crash path -- if the sender disappears after the
          handshake, we DON'T want to block for the socket's
          ``RCVTIMEO`` (10 minutes). The NOBLOCK loop respects
          ``deadline_sec`` and raises ``TimeoutError`` cleanly.

        NOTE on POISONED STREAMS: when the caller gets ``ok=False``
        back, its ``transport_stream`` still has ~1 ``Wait`` queued
        per chunk from the ``recv_batch`` / ``scatter_recv`` that just
        ran -- those waits will only drain when the HCCL transport
        timeout fires (see ``para.timeout`` in
        ``csrc/hccl_pingpong/batch_channel.cc``, currently 10 s).
        Subsequent transfers on the same channel will queue behind those
        timing-out waits. The cleanest production fix would be a
        sender-side ``AbortSend`` that posts the matching
        ``sendDoneSlot`` notifies WITHOUT performing the data copy
        before emitting the error ack, restoring credit balance and
        letting the receiver stream drain immediately. We deliberately
        do not implement that yet: the only current trigger for an
        error ack is a corrupt / spoofed ``receiver_id`` (a
        configuration bug), and callers are expected to ``close()`` the
        channel on failure. Wire the AbortSend up when we add transfer-
        time validation that can fail with a known peer in hand
        (e.g. unknown buffer uuid, out-of-range page index).
        """
        start = time.time()
        deadline = start + deadline_sec
        # Emit a WARNING heartbeat every 5s so a wedged sender no longer
        # presents as a silent hang for up to ``_TRANSFER_ACK_TIMEOUT_SEC``
        # (currently 10 minutes). Cheap because the heartbeat only fires on
        # the slow path; healthy transfers return in microseconds.
        next_heartbeat = start + 5.0
        while True:
            now = time.time()
            if now >= deadline:
                break
            try:
                return peer.transfer_req_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            if now >= next_heartbeat:
                logger.warning(
                    "PingPong: still waiting for transfer ack peer=%s "
                    "elapsed=%.1fs deadline=%.1fs",
                    peer.transfer_url,
                    now - start,
                    deadline_sec,
                )
                next_heartbeat = now + 5.0
            time.sleep(_ASYNC_EVENT_POLL_SEC)
        raise TimeoutError(
            f"PingPong: timed out waiting for transfer ack "
            f"after {deadline_sec}s"
        )

    def _infer_receiver_id_for_request(self, transfer_spec: dict) -> str:
        """Wire-level ``receiver_id`` is OUR own local_id (we are the receiver
        in the data plane). The remote uses it to look up the peer entry it
        registered for us during the init handshake.
        """
        if self.local_id is not None:
            return self.local_id
        explicit = transfer_spec.get("local_id")
        if explicit:
            self._capture_local_id(explicit)
            return explicit
        raise RuntimeError(
            "HcclPingPongChannel: local_id not set. Call lazy_init_peer_connection "
            "first, or pass local_id=... when constructing the channel."
        )

    # ------------------------------------------------------------------
    # async_batched_scatter
    # ------------------------------------------------------------------
    async def async_batched_scatter(
        self,
        scatter_entries: List[Dict[str, Any]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Receiver-driven scatter from a remote host-resident sender buffer.

        Each entry is a dict with:
          - ``sender_local_addr`` (int): sender-side host buffer pointer.
          - ``counts`` (List[int]): per-destination element counts.
          - ``data_type`` (hpp.HcclDataType): element type.
          - ``dst_addrs`` (List[int]): receiver-side device addresses.

        We send one ZMQ scatter request, locally call ``scatter_recv`` against
        the same chunk-packed entries, sync the transport stream, and only
        return after the remote ack.

        NOTE: gated on ``scatter_send_d2d_followup`` for NPU-backed senders;
        v1 only supports host-resident sender ddr buffers. The connector layer
        is responsible for translating slot_mapping -> dst_addrs before calling
        in.
        """
        assert transfer_spec is not None
        if not scatter_entries:
            return 0
        # We are the receiver here, scatter-pulling from a remote sender we
        # initiated to — connector role.
        peer = self._await_peer_ready(
            transfer_spec[TS_RECEIVER_ID], role="connector"
        )
        receiver_id = self._infer_receiver_id_for_request(transfer_spec)

        wire_entries: List[PingPongScatterEntryMsg] = []
        cpp_recv_entries: List[hpp.PingPongScatterEntry] = []
        for e in scatter_entries:
            sender_addr = int(e["sender_local_addr"])
            counts = list(e["counts"])
            # Accept either a raw int or the pybind11 hpp.HcclDataType enum;
            # both convert via int() since the C++ enum is bound with int cast.
            data_type_int = int(e["data_type"])
            dst_addrs = [int(a) for a in e["dst_addrs"]]
            if len(dst_addrs) != len(counts):
                raise ValueError(
                    "async_batched_scatter: dst_addrs and counts must align"
                )

            wire_entries.append(
                PingPongScatterEntryMsg(
                    sender_local_addr=sender_addr,
                    counts=counts,
                    data_type=data_type_int,
                )
            )
            cpp_recv = hpp.PingPongScatterEntry()
            cpp_recv.ddr_buf = 0
            cpp_recv.dst_bufs = dst_addrs
            cpp_recv.counts = counts
            cpp_recv.data_type = hpp.HcclDataType(data_type_int)
            cpp_recv_entries.append(cpp_recv)

        req = PingPongScatterRequest(
            receiver_id=receiver_id,
            entries=wire_entries,
        )

        loop = asyncio.get_running_loop()
        total_count = sum(len(e["counts"]) for e in scatter_entries)

        def _run_blocking() -> int:
            # Set the NPU device on the executor worker thread; see the
            # equivalent comment in async_batched_read for the rationale.
            torch.npu.set_device(self.handle_device)
            # See ``batched_read`` for the locking rationale: ack-wait must
            # happen WITHOUT ``_transfer_lock`` so the listener side can
            # service symmetric peer reads concurrently.
            t0 = time.monotonic()
            n_chunks = sum(len(e.counts) for e in cpp_recv_entries)
            with peer.transfer_req_lock:
                t1 = time.monotonic()
                with self._transfer_lock:
                    t2 = time.monotonic()
                    peer.transfer_req_socket.send(msgspec.msgpack.encode(req))
                    t3 = time.monotonic()
                    self.agent.scatter_recv(
                        peer.conn_handle,
                        cpp_recv_entries,
                        self.transport_stream.npu_stream,
                    )
                    t4 = time.monotonic()
                ack_bytes = self._wait_for_transfer_ack(peer)
                t5 = time.monotonic()
            ack = msgspec.msgpack.decode(ack_bytes, type=PingPongMsg)
            self._maybe_log_ack_failure(
                ack, peer, n_chunks, "async_batched_scatter",
                t0, t1, t2, t3, t4, t5,
            )
            self._raise_on_ack_failure(ack)
            # Success: drain stream so dst_addrs are populated when the
            # coroutine resumes. Listener-side scatter_send lives on
            # ``_send_stream`` (see ``__init__``), so no cross-direction
            # FIFO interference on this stream.
            t6 = time.monotonic()
            self.transport_stream.synchronize()
            t7 = time.monotonic()
            self._maybe_log_transfer_phases(
                "async_batched_scatter", peer, n_chunks,
                t0, t1, t2, t3, t4, t5, t6, t7,
            )
            return total_count

        return await loop.run_in_executor(None, _run_blocking)

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def close(self):
        self.running = False
        for thread in self.running_threads:
            thread.join(timeout=2.0)
        with self._state_lock:
            # Only the connector dict owns transfer_req_sockets; the
            # listener dict's entries always have ``transfer_req_socket=None``
            # so iterating it just to clear() is fine.
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
        self.zmq_context.term()
        # The dedicated sync context used by ``_transfer_socket`` is separate
        # from ``self.zmq_context`` so we have to tear it down explicitly.
        try:
            self._transfer_zmq_context.term()
        except Exception:
            pass
