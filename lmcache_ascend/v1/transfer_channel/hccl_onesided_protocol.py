# SPDX-License-Identifier: Apache-2.0
"""Wire messages for the HCCL write-driven one-sided transfer channel."""

# Standard
from typing import List, Union

# Third Party
import msgspec

# Local
from .buffer_config import PeerBufferInfo


class OneSidedMsgBase(msgspec.Struct, tag=True):
    pass


class OneSidedInitRequest(OneSidedMsgBase):
    local_id: str
    client_meta_bytes: bytes
    buffer_infos: List[PeerBufferInfo]
    slot_bytes: int
    recv_slots: int
    send_slots: int


class OneSidedInitResponse(OneSidedMsgBase):
    server_meta_bytes: bytes
    buffer_infos: List[PeerBufferInfo]
    transfer_url: str
    slot_bytes: int
    recv_slots: int
    send_slots: int


class OneSidedReadRequest(OneSidedMsgBase):
    receiver_id: str
    request_id: int
    sender_local_addrs: List[int]
    sizes: List[int]


class OneSidedReadAck(OneSidedMsgBase):
    ok: bool
    error: str = ""


OneSidedMsg = Union[
    OneSidedInitRequest,
    OneSidedInitResponse,
    OneSidedReadRequest,
    OneSidedReadAck,
]
