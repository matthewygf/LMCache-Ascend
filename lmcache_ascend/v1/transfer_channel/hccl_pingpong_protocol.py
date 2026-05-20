# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Union

# Third Party
import msgspec

# Local
from .buffer_config import PeerBufferInfo


class PingPongMsgBase(msgspec.Struct, tag=True):
    """Base for all ping-pong control messages (init handshake + transfer)."""

    pass


# ---------------------------------------------------------------------------
# Init handshake (REP socket on listener side)
# ---------------------------------------------------------------------------


class PingPongInitRequest(PingPongMsgBase):
    """Sent by the connector (REQ side) to the listener.

    Carries the connector's PingPongClientMeta and the list of buffer infos
    the listener will need to logically resolve uuid+page indexes back to
    sender-side virtual addresses. The pingpong agent does NOT register user
    buffers with HCCL — only the agent's shared input/output ping-pong regions
    are registered — so we just exchange (uuid, ptr, size, page_size) here.
    """

    local_id: str
    client_meta_bytes: bytes
    buffer_infos: List[PeerBufferInfo]


class PingPongInitResponse(PingPongMsgBase):
    """Reply from the listener carrying its server meta, its own buffer infos,
    and the URL of its shared transfer REP socket.

    The connector saves transfer_url and uses a single REQ socket per peer to
    issue read/scatter requests against that URL during normal operation.
    """

    server_meta_bytes: bytes
    buffer_infos: List[PeerBufferInfo]
    transfer_url: str


# ---------------------------------------------------------------------------
# Transfer-time messages (REP socket per channel; senders/listeners poll, peers
# REQ to that single URL)
# ---------------------------------------------------------------------------


class PingPongReadRequest(PingPongMsgBase):
    """Receiver -> remote sender: please send these flat pages over the
    BatchChannel.

    The receiver pre-resolves the sender-side virtual addresses from
    (buffer_uuid, page_index) pairs against its local copy of the sender's
    PeerBufferInfo list, so the sender does not have to repeat that lookup.
    """

    receiver_id: str
    sender_local_addrs: List[int]
    sizes: List[int]


class PingPongReadAck(PingPongMsgBase):
    """Sender -> receiver: ``request-accepted`` ack for a PingPongReadRequest.

    Semantics (NOT a "transfer complete" ack):
      - ``ok=True``: the sender validated the request synchronously
        (receiver_id known, addrs/sizes shape sane) and is about to enqueue
        the corresponding ``send_batch``. It does NOT mean any data has
        moved yet — the receiver MUST still wait on its own
        ``transport_stream`` synchronize() to know when the bytes have
        landed in its destination buffers.
      - ``ok=False``: validation failed; the sender will not run the
        transfer. The receiver propagates ``error`` as an exception
        WITHOUT submitting any local recv work.

    Always sending exactly one ack (success or error) keeps the REP socket
    in lockstep, matching HcclChannel._init_loop's contract.
    """

    ok: bool
    error: str = ""


class PingPongScatterEntryMsg(msgspec.Struct):
    """One entry in a PingPongScatterRequest.

    sender_local_addr: pointer (sender-virtual) to the host-resident contiguous
        source buffer for this entry. Per-slice byte offsets are computed by
        cumsum-ing counts * sizeof(data_type), matching the receiver side
        deterministically.
    counts: per-destination element counts (length = number of destinations).
    data_type: HcclDataType enum value (matches the C++ HcclDataType enum).
    """

    sender_local_addr: int
    counts: List[int]
    data_type: int


class PingPongScatterRequest(PingPongMsgBase):
    """Receiver -> remote sender: please scatter-send this batch of entries.

    The receiver supplies its destination addresses to its own scatter_recv
    locally — the wire message only needs the sender-side source layout plus
    per-entry counts/dtype so chunk packing aligns on both ends.
    """

    receiver_id: str
    entries: List[PingPongScatterEntryMsg]


class PingPongScatterAck(PingPongMsgBase):
    ok: bool
    error: str = ""


# Union of every wire message we may decode on either init or transfer paths.
# Includes init for the REP socket on the init side; the transfer worker uses
# only the transfer-time members.
PingPongMsg = Union[
    PingPongInitRequest,
    PingPongInitResponse,
    PingPongReadRequest,
    PingPongReadAck,
    PingPongScatterRequest,
    PingPongScatterAck,
]
