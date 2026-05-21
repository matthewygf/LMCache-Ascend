# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# Standard
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import asyncio
import multiprocessing as mp
import os
import sys
import time
import warnings

# First Party TODO Amory - Re enable, temporarily disabled to let local tests run
#from tests.bootstrap import prepare_environment

#prepare_environment()

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, PagedCpuGpuMemoryAllocator
import pytest
import torch

# First Party
from lmcache_ascend import _build_info

_cann_ver = _build_info.cann_version_tuple()

# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------


@dataclass
class PingPongTestConfig:
    num_objs: int
    kv_shape: Tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    send_device_id: int = 0
    recv_device_id: int = 1
    timeout: int = 120
    use_host_sender: bool = False  # only used by scatter test


def _calc_tensor_bytes(kv_shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for d in kv_shape:
        n *= d
    return n * torch.tensor([], dtype=dtype).itemsize


def _make_npu_allocator(
    device_id: int, kv_shape: Tuple[int, ...], dtype: torch.dtype, num_objs: int
) -> PagedCpuGpuMemoryAllocator:
    allocator = PagedCpuGpuMemoryAllocator()
    page_bytes = _calc_tensor_bytes(kv_shape, dtype)
    # Grossly oversize so allocations don't fragment.
    buffer_size = page_bytes * max(num_objs, 1) * 2
    allocator.init_gpu_memory_allocator(
        buffer_size,
        [torch.Size(kv_shape)],
        [dtype],
        MemoryFormat.KV_2LTD,
        device_id,
    )
    return allocator


def _make_host_allocator(
    kv_shape: Tuple[int, ...], dtype: torch.dtype, num_objs: int
) -> PagedCpuGpuMemoryAllocator:
    allocator = PagedCpuGpuMemoryAllocator()
    page_bytes = _calc_tensor_bytes(kv_shape, dtype)
    buffer_size = page_bytes * max(num_objs, 1) * 2
    allocator.init_cpu_memory_allocator(
        buffer_size,
        [torch.Size(kv_shape)],
        [dtype],
        MemoryFormat.KV_2LTD,
    )
    return allocator


def _channel_buffer_args(
    allocator: PagedCpuGpuMemoryAllocator,
    page_bytes: int,
    use_host: bool,
) -> Tuple[List[int], List[int], List[str], List[int]]:
    if use_host:
        return (
            [allocator.cpu_allocator.buffer_ptr],
            [allocator.cpu_allocator.buffer_size],
            ["cpu"],
            [page_bytes],
        )
    return (
        [allocator.gpu_allocator.buffer_ptr],
        [allocator.gpu_allocator.buffer_size],
        ["npu"],
        [page_bytes],
    )


# ---------------------------------------------------------------------------
# batched_read worker processes
# ---------------------------------------------------------------------------

# read sender
def _read_provider_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.send_device_id)

        # Local import so prepare_environment() runs first in subprocess.
        # First Party
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        allocator = _make_npu_allocator(
            config.send_device_id, config.kv_shape, config.dtype, config.num_objs
        )
        objs = []
        expected = []
        for i in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            fill = float(i) + 0.5
            obj.tensor.fill_(fill)
            objs.append(obj)
            expected.append(fill)

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        # Wildcard + ephemeral port. The channel resolves the actual bound
        # port via zmq LAST_ENDPOINT and exposes it on
        # ``channel.peer_init_url_resolved``; we publish that for the reader
        # so parallel CI runs can't collide on a hard-coded port.
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=False)

        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="sender",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=str(config.send_device_id),
        )

        # We are the LISTENER (sender side). The reader will initiate the
        # handshake against our peer_init_url; we just publish our buffer refs
        # and wait. The transfer-side REQ socket only flows connector ->
        # listener, which matches the receiver-driven read direction.
        uuids, idxs = channel.get_local_buffer_refs(objs)
        shared["provider_uuids"] = list(uuids)
        shared["provider_idxs"] = list(idxs)
        shared["provider_expected"] = list(expected)
        shared["provider_init_url"] = channel.peer_init_url_resolved
        shared["provider_ready"] = True

        deadline = time.time() + config.timeout
        while "read_complete" not in shared:
            time.sleep(0.1)
            if time.time() > deadline:
                raise TimeoutError("provider: read never completed")

        channel.close()
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("provider failed: %s", e)
        sys.exit(1)


def _read_reader_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)

        # First Party
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        allocator = _make_npu_allocator(
            config.recv_device_id, config.kv_shape, config.dtype, config.num_objs
        )
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

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        # Ephemeral; we don't accept inbound init traffic on the reader, so
        # any free port is fine.
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=False)

        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="receiver",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=str(config.recv_device_id),
        )

        deadline = time.time() + 30
        while "provider_init_url" not in shared:
            time.sleep(0.05)
            if time.time() > deadline:
                raise TimeoutError("reader: provider never reported ready")

        # The reader is the CONNECTOR (it will REQ-issue transfer requests),
        # so it initiates the handshake against the provider's REP url.
        # The provider published its resolved tcp host:port via
        # ``shared["provider_init_url"]`` after the eager bind, so we
        # connect to that real interface (not 0.0.0.0).
        remote_url = shared["provider_init_url"]
        channel.lazy_init_peer_connection(
            local_id=str(config.recv_device_id),
            peer_id=str(config.send_device_id),
            peer_init_url=remote_url,
        )

        provider_uuids = list(shared["provider_uuids"])
        provider_idxs = list(shared["provider_idxs"])
        expected = list(shared["provider_expected"])

        transfer_spec = {
            "receiver_id": str(config.send_device_id),
            "remote_buffer_uuids": provider_uuids,
            "remote_mem_indexes": provider_idxs,
        }

        logger.info(
            "reader: starting batched_read of %d pages of %d bytes",
            config.num_objs,
            page_bytes,
        )
        t0 = time.time()
        channel.batched_read(buffers=objs, transfer_spec=transfer_spec)
        logger.info("reader: batched_read finished in %.4fs", time.time() - t0)

        for i, obj in enumerate(objs):
            host = obj.tensor.cpu()
            ev = float(expected[i])
            if not (host == ev).all():
                sample = host.flatten()[:5].float().numpy()
                raise AssertionError(
                    f"page {i}: expected {ev}, got sample {sample}"
                )

        shared["read_complete"] = True
        channel.close()
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("reader failed: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# submit_batched_read reader (uses event-based completion; same provider)
# ---------------------------------------------------------------------------


def _submit_read_reader_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    """Same wire as ``_read_reader_process`` but exercises
    ``submit_batched_read`` + ``event.synchronize()``.

    Validates the new pipelined contract used by ``ProxyMemoryObj.
    submit_resolve_batch``: the channel must return a recorded
    ``torch.npu.Event`` after the sender ack arrives, and the recorded
    event must be on ``transport_stream`` so a cross-stream
    ``wait_event`` (or a synchronous ``synchronize()``) is enough to
    guarantee the destination buffers are populated.
    """
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)

        # First Party
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        allocator = _make_npu_allocator(
            config.recv_device_id, config.kv_shape, config.dtype, config.num_objs
        )
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

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=False)

        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="receiver",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=str(config.recv_device_id),
        )

        deadline = time.time() + 30
        while "provider_init_url" not in shared:
            time.sleep(0.05)
            if time.time() > deadline:
                raise TimeoutError("submit reader: provider never reported ready")

        remote_url = shared["provider_init_url"]
        channel.lazy_init_peer_connection(
            local_id=str(config.recv_device_id),
            peer_id=str(config.send_device_id),
            peer_init_url=remote_url,
        )

        provider_uuids = list(shared["provider_uuids"])
        provider_idxs = list(shared["provider_idxs"])
        expected = list(shared["provider_expected"])

        transfer_spec = {
            "receiver_id": str(config.send_device_id),
            "remote_buffer_uuids": provider_uuids,
            "remote_mem_indexes": provider_idxs,
        }

        logger.info(
            "submit reader: starting submit_batched_read of %d pages of %d bytes",
            config.num_objs,
            page_bytes,
        )
        t0 = time.time()
        event = channel.submit_batched_read(buffers=objs, transfer_spec=transfer_spec)
        assert event is not None, (
            "submit_batched_read should return a non-None event when buffers"
            " is non-empty"
        )
        # The event MUST be recorded on transport_stream; synchronize()
        # blocks until every queued recv_batch op completes. After this
        # returns the destination buffers must hold the sender's data.
        event.synchronize()
        logger.info(
            "submit reader: submit_batched_read finished in %.4fs",
            time.time() - t0,
        )

        for i, obj in enumerate(objs):
            host = obj.tensor.cpu()
            ev = float(expected[i])
            if not (host == ev).all():
                sample = host.flatten()[:5].float().numpy()
                raise AssertionError(
                    f"submit page {i}: expected {ev}, got sample {sample}"
                )

        # Empty-buffer fast path must short-circuit and return None.
        empty_event = channel.submit_batched_read(
            buffers=[], transfer_spec=transfer_spec
        )
        if empty_event is not None:
            raise AssertionError(
                "submit_batched_read([], ...) must return None, got an event"
            )

        shared["read_complete"] = True
        channel.close()
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("submit reader failed: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# scatter worker processes
# ---------------------------------------------------------------------------


def _scatter_provider_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    """Sender side: register a host-resident contiguous buffer with a single
    UUID. The receiver requests scatter by sending PingPongScatterRequest, the
    sender's transfer worker handles it via PingPongAgent.scatter_send.
    """
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.send_device_id)

        # First Party
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        allocator = _make_host_allocator(
            config.kv_shape, config.dtype, config.num_objs
        )
        objs = []
        expected = []
        for i in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="cpu",
            )
            fill = float(i) + 0.25
            obj.tensor.fill_(fill)
            objs.append(obj)
            expected.append(fill)

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=True)

        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="sender",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=str(config.send_device_id),
        )

        # Listener side: just publish host-buffer addrs and wait. Reader will
        # connect.
        sender_local_addrs = [obj.data_ptr for obj in objs]
        shared["sender_local_addrs"] = list(sender_local_addrs)
        shared["scatter_expected"] = list(expected)
        shared["provider_init_url"] = channel.peer_init_url_resolved
        shared["provider_ready"] = True

        deadline = time.time() + config.timeout
        while "scatter_complete" not in shared:
            time.sleep(0.1)
            if time.time() > deadline:
                raise TimeoutError("scatter provider: never completed")
        channel.close()
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("scatter provider failed: %s", e)
        sys.exit(1)


def _scatter_reader_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    try:
        warnings.filterwarnings("ignore", message=".*torch.Tensor.cuda.*")
        logger = init_logger(__name__)
        torch.npu.set_device(config.recv_device_id)

        # First Party
        import lmcache_ascend.hccl_pingpong_npu_comms as hpp
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        allocator = _make_npu_allocator(
            config.recv_device_id, config.kv_shape, config.dtype, config.num_objs
        )
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

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=False)

        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="receiver",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=str(config.recv_device_id),
        )

        deadline = time.time() + 30
        while "provider_init_url" not in shared:
            time.sleep(0.05)
            if time.time() > deadline:
                raise TimeoutError("scatter reader: provider never ready")

        # Reader is the connector: REQ-side flows from us to the listener.
        remote_url = shared["provider_init_url"]
        channel.lazy_init_peer_connection(
            local_id=str(config.recv_device_id),
            peer_id=str(config.send_device_id),
            peer_init_url=remote_url,
        )

        sender_local_addrs = list(shared["sender_local_addrs"])
        expected = list(shared["scatter_expected"])

        # Map torch dtype -> hpp.HcclDataType.
        dtype_map = {
            torch.bfloat16: hpp.HcclDataType.HCCL_DATA_TYPE_BFP16,
            torch.float16: hpp.HcclDataType.HCCL_DATA_TYPE_FP16,
            torch.float32: hpp.HcclDataType.HCCL_DATA_TYPE_FP32,
            torch.int8: hpp.HcclDataType.HCCL_DATA_TYPE_INT8,
            torch.uint8: hpp.HcclDataType.HCCL_DATA_TYPE_UINT8,
        }
        if config.dtype not in dtype_map:
            raise ValueError(
                f"unsupported dtype {config.dtype} for scatter test"
            )
        data_type = dtype_map[config.dtype]
        elem_count = page_bytes // torch.tensor([], dtype=config.dtype).itemsize

        # One scatter entry per sender page, with a single destination each.
        # This is the simplest layout that exercises ScatterSend's per-entry
        # PushSegments call boundary.
        scatter_entries: List[Dict[str, Any]] = []
        for sender_addr, recv_obj in zip(sender_local_addrs, objs, strict=True):
            dst_addr = recv_obj.data_ptr
            scatter_entries.append(
                {
                    "sender_local_addr": int(sender_addr),
                    "counts": [int(elem_count)],
                    "data_type": data_type,
                    "dst_addrs": [int(dst_addr)],
                }
            )

        transfer_spec = {"receiver_id": str(config.send_device_id)}

        logger.info(
            "scatter reader: invoking async_batched_scatter (%d entries)",
            len(scatter_entries),
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                channel.async_batched_scatter(scatter_entries, transfer_spec)
            )
        finally:
            loop.close()

        for i, obj in enumerate(objs):
            host = obj.tensor.cpu()
            ev = float(expected[i])
            if not (host == ev).all():
                sample = host.flatten()[:5].float().numpy()
                raise AssertionError(
                    f"scatter page {i}: expected {ev}, got {sample}"
                )

        shared["scatter_complete"] = True
        channel.close()
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("scatter reader failed: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Failure-injection worker
# ---------------------------------------------------------------------------


def _failure_reader_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    """Issue a batched_read with a deliberately wrong receiver_id (we set our
    local_id to a value the sender doesn't know). Sender's transfer worker must
    return an error ack and our batched_read must raise — not deadlock.
    """
    try:
        # First Party
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        torch.npu.set_device(config.recv_device_id)

        allocator = _make_npu_allocator(
            config.recv_device_id, config.kv_shape, config.dtype, config.num_objs
        )
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

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=False)

        wrong_local_id = "definitely_not_registered"
        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="receiver",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=wrong_local_id,
        )

        deadline = time.time() + 30
        while "provider_init_url" not in shared:
            time.sleep(0.05)
            if time.time() > deadline:
                raise TimeoutError("failure reader: provider never ready")

        # We connect using wrong_local_id (which the provider WILL register
        # under that key during accept). To force a genuine sender-side
        # lookup miss at transfer time we then override channel.local_id to
        # something the sender never saw. This simulates a spoofed/corrupted
        # local_id on the transfer request.
        remote_url = shared["provider_init_url"]
        channel.lazy_init_peer_connection(
            local_id=wrong_local_id,
            peer_id=str(config.send_device_id),
            peer_init_url=remote_url,
        )
        channel.local_id = "spoofed_unknown_id"

        provider_uuids = list(shared["provider_uuids"])
        provider_idxs = list(shared["provider_idxs"])
        transfer_spec = {
            "receiver_id": str(config.send_device_id),
            "remote_buffer_uuids": provider_uuids,
            "remote_mem_indexes": provider_idxs,
        }

        try:
            channel.batched_read(buffers=objs, transfer_spec=transfer_spec)
            shared["failure_outcome"] = "no_error_raised"
        except RuntimeError as e:
            shared["failure_outcome"] = f"raised:{e}"
        except Exception as e:
            shared["failure_outcome"] = f"raised_other:{type(e).__name__}:{e}"

        shared["failure_done"] = True
        # batched_read intentionally rejects the spoofed receiver_id BEFORE
        # ever posting the sendDoneSlot notifies the receiver queued on its
        # transport_stream. Those queued waits will sit until the HCCL
        # transport-level timeout fires (see batch_channel.cc), so a normal
        # Python shutdown would block on Stream.__del__ ->
        # aclrtDestroyStream waiting for the poisoned stream to drain.
        # The test only cares that batched_read raised; bypass interpreter
        # cleanup so this process can terminate within the test budget
        # regardless of how long the device-side waits take to time out.
        try:
            channel.close()
        except Exception:
            pass
        os._exit(0)
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("failure reader unexpected exception: %s", e)
        shared["failure_outcome"] = f"setup_error:{e}"
        shared["failure_done"] = True
        os._exit(0)


def _failure_provider_process(
    config: PingPongTestConfig, shared: Dict[str, Any]
) -> None:
    try:
        # First Party
        from lmcache_ascend.v1.transfer_channel import CreateTransferChannel

        torch.npu.set_device(config.send_device_id)

        allocator = _make_npu_allocator(
            config.send_device_id, config.kv_shape, config.dtype, config.num_objs
        )
        objs = []
        for _ in range(config.num_objs):
            obj = allocator.allocate(
                torch.Size(config.kv_shape),
                config.dtype,
                fmt=MemoryFormat.KV_2LTD,
                allocator_type="gpu",
            )
            obj.tensor.fill_(1.0)
            objs.append(obj)

        page_bytes = _calc_tensor_bytes(config.kv_shape, config.dtype)
        local_url = "0.0.0.0:0"
        bp, bs, bt, ab = _channel_buffer_args(allocator, page_bytes, use_host=False)

        channel = CreateTransferChannel(
            channel_type="hccl_pingpong",
            async_mode=False,
            role="sender",
            buffer_ptr=bp,
            buffer_size=bs,
            buffer_type=bt,
            align_bytes=ab,
            tp_rank=0,
            peer_init_url=local_url,
            local_id=str(config.send_device_id),
        )

        # Listener side; reader connects.
        uuids, idxs = channel.get_local_buffer_refs(objs)
        shared["provider_uuids"] = list(uuids)
        shared["provider_idxs"] = list(idxs)
        shared["provider_init_url"] = channel.peer_init_url_resolved
        shared["provider_ready"] = True

        deadline = time.time() + config.timeout
        while "failure_done" not in shared:
            time.sleep(0.1)
            if time.time() > deadline:
                raise TimeoutError("failure provider: reader never finished")
        channel.close()
    except Exception as e:
        logger = init_logger(__name__)
        logger.error("failure provider failed: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


def _run_two_process(
    config: PingPongTestConfig, sender_fn, receiver_fn
) -> Dict[str, Any]:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with mp.Manager() as manager:
        shared = manager.dict()

        p_recv = mp.Process(
            target=receiver_fn, args=(config, shared), name="ReceiverProcess"
        )
        p_send = mp.Process(
            target=sender_fn, args=(config, shared), name="SenderProcess"
        )
        p_recv.start()
        p_send.start()

        p_send.join(timeout=config.timeout)
        p_recv.join(timeout=config.timeout)

        snapshot = dict(shared)

        errs = []
        if p_send.is_alive():
            p_send.terminate()
            errs.append("Sender timed out")
        elif p_send.exitcode != 0:
            errs.append(f"Sender exitcode={p_send.exitcode}")
        if p_recv.is_alive():
            p_recv.terminate()
            errs.append("Receiver timed out")
        elif p_recv.exitcode != 0:
            errs.append(f"Receiver exitcode={p_recv.exitcode}")
        if errs:
            pytest.fail("\n".join(errs))
        return snapshot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

"""
Receiver-driven batched read of many 256 KiB pages on NPU buffers. 
Needs at least 3 ping-pong wraparound cycles to exercise the trailing-partial-epoch 
``recvReady`` Post.
"""
@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    # 256 KiB pages, 200 entries -> 50 MiB total at default 16 MiB ping-pong
    # cycle => 3 wraps + 2 MiB tail (mirrors the C++ main.cc demo).
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (200, 31, 256, 8, 128),
    ],
)
def test_pingpong_batched_read(
    num_objs, num_layer, chunk_size, num_kv_head, head_size
):
    config = PingPongTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=180,
    )
    _run_two_process(config, _read_provider_process, _read_reader_process)


"""
Same wire as test_pingpong_batched_read but exercises submit_batched_read +
event.synchronize() so ProxyMemoryObj.submit_resolve_batch can take the
event-driven cross-stream pipelining path used by the NPU connector for
delay-pull (instead of falling back to synchronous batched_read).
"""
@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (200, 31, 256, 8, 128),
    ],
)
def test_pingpong_submit_batched_read(
    num_objs, num_layer, chunk_size, num_kv_head, head_size
):
    config = PingPongTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=180,
    )
    _run_two_process(config, _read_provider_process, _submit_read_reader_process)


"""
Async scatter from a host-resident contiguous buffer on the sender to per-page
NPU destinations on the receiver. Validates the H2D-only v1 of
BatchChannel.ScatterSend end-to-end.
"""
@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
@pytest.mark.parametrize(
    "num_objs, num_layer, chunk_size, num_kv_head, head_size",
    [
        (16, 31, 256, 8, 128),
    ],
)
def test_pingpong_async_batched_scatter_host_source(
    num_objs, num_layer, chunk_size, num_kv_head, head_size
):
    config = PingPongTestConfig(
        num_objs=num_objs,
        kv_shape=(num_layer, 2, chunk_size, num_kv_head, head_size),
        timeout=120,
        use_host_sender=True,
    )
    _run_two_process(config, _scatter_provider_process, _scatter_reader_process)

"""
Exercise the REQ/REP error contract by issuing a read with an invalid ``receiver_id`` 
and asserting the channel raises (rather than deadlocking).
"""

@pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Requires at least 2 NPU devices",
)
def test_pingpong_read_failure_propagates():
    config = PingPongTestConfig(
        num_objs=4,
        kv_shape=(31, 2, 256, 8, 128),
        timeout=60,
    )
    snap = _run_two_process(
        config, _failure_provider_process, _failure_reader_process
    )
    outcome = snap.get("failure_outcome", "")
    assert outcome.startswith("raised"), (
        f"expected failure to raise, got outcome={outcome!r}"
    )


# ---------------------------------------------------------------------------
# Bidirectional peer-registry collision regression test.
#
# When DP=2 (or any topology where two LMCache engines pull from each other
# over the same pingpong channel) the channel ends up acting as both
# connector ("I'm pulling from peer X") and listener ("peer X is pulling
# from me") for the SAME remote id X at the same time. Historically both
# roles shared a single ``self._peers`` dict keyed by X, so whichever role
# registered second would silently overwrite the other:
#
#   - connector entry: ``transfer_req_socket`` -> real REQ socket
#   - listener entry:  ``transfer_req_socket`` -> None
#
# If the listener-side insert won, the next connector-side transfer would
# crash on ``peer.transfer_req_socket.send(...)`` with
# ``AttributeError: 'NoneType' object has no attribute 'send'``. This test
# pins the split-dict invariant introduced to fix that race.
#
# The test deliberately bypasses ``HcclPingPongChannel.__init__`` (which
# binds ZMQ sockets, opens NPU agents, and spawns background loops) and
# instead constructs a bare instance via ``__new__``, populating only the
# fields touched by the registry methods under test. This keeps the test
# CI-friendly (no NPU, no network) and pinpoints the specific failure mode.
# ---------------------------------------------------------------------------


def _make_bare_channel():
    """Build an HcclPingPongChannel skeleton sufficient to exercise the
    peer-registry methods without touching the NPU agent, sockets, or any
    other heavy init."""
    # Local import: pulls in the module under test only here so we don't
    # disturb collection-time imports for tests that legitimately need NPU.
    # First Party
    import threading as _threading  # noqa: PLR0402

    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        HcclPingPongChannel,
    )

    chan = HcclPingPongChannel.__new__(HcclPingPongChannel)
    chan._state_lock = _threading.Lock()
    chan._connector_peers = {}
    chan._listener_peers = {}
    return chan


def _make_ready_peer(transfer_req_socket):
    """Build a minimal _PeerState pre-marked ready so _await_peer_ready
    returns immediately. We only assert on the *identity* of the returned
    entry, never invoke its socket or conn_handle, so dummy values are OK."""
    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        _PeerState,
    )

    peer = _PeerState(
        conn_handle=0,
        transfer_url="",
        transfer_req_socket=transfer_req_socket,
        remote_buffers=None,
    )
    peer.ready_event.set()
    return peer


def test_pingpong_bidirectional_peer_registry_does_not_collide():
    """Repro: a connector entry and a listener entry for the SAME peer_id
    must coexist; neither role should clobber the other."""
    chan = _make_bare_channel()

    fake_socket = object()  # stand-in for a real zmq.REQ socket
    connector_peer = _make_ready_peer(fake_socket)
    listener_peer = _make_ready_peer(None)  # listener never has a socket

    peer_id = "remote_dp1_tp0"
    # Simulate the connector path running first (we initiated to remote).
    chan._connector_peers[peer_id] = connector_peer
    # Then the listener path fires concurrently (remote initiated to us).
    chan._listener_peers[peer_id] = listener_peer

    # Pre-fix the listener insert above would have *overwritten* the
    # connector entry. With split dicts, both must survive.
    assert chan._connector_peers[peer_id] is connector_peer
    assert chan._listener_peers[peer_id] is listener_peer

    # And _await_peer_ready must route role->dict correctly.
    assert (
        chan._await_peer_ready(peer_id, role="connector")
        is connector_peer
    )
    assert (
        chan._await_peer_ready(peer_id, role="listener")
        is listener_peer
    )

    # The connector entry must still be the one with the real socket
    # (this is the exact field the pre-fix AttributeError tripped on).
    returned = chan._await_peer_ready(peer_id, role="connector")
    assert returned.transfer_req_socket is fake_socket


def test_pingpong_remote_xfer_handler_exists_only_checks_listener():
    """``remote_xfer_handler_exists`` answers a sender-side question:
    'has the receiver previously initiated the BatchChannel handshake to
    me?'. That bookkeeping lives in ``_listener_peers``, never in
    ``_connector_peers``."""
    chan = _make_bare_channel()
    peer_id = "remote_dp1_tp0"

    # Only the connector dict knows about this peer -> sender-side handler
    # should refuse to serve a lookup from it (no handshake from them yet).
    chan._connector_peers[peer_id] = _make_ready_peer(object())
    assert chan.remote_xfer_handler_exists(peer_id) is False

    # Once the listener-side entry exists, the answer flips.
    chan._listener_peers[peer_id] = _make_ready_peer(None)
    assert chan.remote_xfer_handler_exists(peer_id) is True


def test_pingpong_await_peer_ready_rejects_unknown_role():
    chan = _make_bare_channel()
    with pytest.raises(ValueError, match="unknown role"):
        chan._await_peer_ready("anything", role="bogus")


def test_pingpong_await_peer_ready_missing_peer_is_role_scoped():
    """A KeyError for one role must NOT short-circuit a lookup for the
    other role on the same peer id."""
    chan = _make_bare_channel()
    peer_id = "remote_dp1_tp0"
    chan._connector_peers[peer_id] = _make_ready_peer(object())

    # The listener side never registered this peer -> should KeyError.
    with pytest.raises(KeyError):
        chan._await_peer_ready(peer_id, role="listener")
    # But the connector side still works.
    assert chan._await_peer_ready(peer_id, role="connector") is not None


# ---------------------------------------------------------------------------
# Bidirectional pull deadlock regression
#
# Under DP with TP-aware controller routing, channel A and channel B can
# concurrently pull from each other on the same-TP-rank channel pair. The
# receive path's ``_transfer_lock`` MUST be released before blocking on the
# sender's ack, otherwise each side holds its own channel lock while waiting
# for the other's ack and neither side's ``_transfer_loop`` can acquire its
# own ``_transfer_lock`` to service the symmetric incoming request. Both
# sides then deadlock indefinitely; vLLM's ``shm_broadcast`` 60s timeouts
# appear on both EngineCores and the benchmark wedges with zero throughput.
#
# The unit test below directly asserts the locking invariant on the
# synchronous ``batched_read`` path using mocked I/O so it runs without an
# NPU or a real network. The same fix applies to ``async_batched_read``,
# ``submit_batched_read``, and ``async_batched_scatter``; structural review
# is sufficient since the lock-discipline change is identical.
# ---------------------------------------------------------------------------


def test_pingpong_transfer_lock_released_during_ack_wait():
    """Repro: bidirectional symmetric pulls would deadlock if
    ``_transfer_lock`` is held across ``_wait_for_transfer_ack``.

    Approach:
        1. Drive ``batched_read`` from a background thread with every heavy
           dependency mocked, gating ``_wait_for_transfer_ack`` on an event
           so the call parks deterministically.
        2. From the main thread, try to acquire ``_transfer_lock``. This
           simulates the channel's own ``_transfer_loop`` attempting to
           enter ``_handle_read_request`` to service an incoming peer
           ``PingPongReadRequest`` (the symmetric direction).
        3. Without the fix the acquire blocks forever (deadlock). With the
           fix the acquire succeeds promptly.
    """
    # Standard
    import threading
    from unittest.mock import MagicMock

    # Third Party
    import msgspec

    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        _PeerState,
    )
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_protocol import (
        PingPongReadAck,
    )
    from lmcache_ascend.v1.transfer_channel.transfer_spec import (
        TS_RECEIVER_ID,
    )

    chan = _make_bare_channel()
    chan._transfer_lock = threading.Lock()
    chan.local_id = "test_local"
    chan.handle_device = 0

    # Build a single ready peer with a mocked REQ socket. ``batched_read``
    # only calls ``.send(bytes)`` on it; we don't model recv() because we
    # patch ``_wait_for_transfer_ack`` directly.
    peer = _PeerState(
        conn_handle=1,
        transfer_url="tcp://fake",
        transfer_req_socket=MagicMock(),
        remote_buffers=None,
    )
    peer.ready_event.set()

    # Mock the agent / stream so no NPU is touched.
    chan.agent = MagicMock()
    chan.transport_stream = MagicMock()
    chan.transport_stream.npu_stream = None

    # Patch the heavyweight helpers so the call body is exercised end-to-end
    # but with no real plan resolution.
    chan._build_read_plan = lambda buffers, spec: (
        peer,
        [0xDEADBEEF],
        [4096],
        [0xBEEFDEAD],
    )
    chan._infer_receiver_id_for_request = lambda spec: "test_remote"
    chan._raise_on_ack_failure = lambda ack: None

    # Gating: the patched ack-wait sets ``waiting_started`` then blocks
    # until ``may_complete`` is set, returning a valid ack on release.
    waiting_started = threading.Event()
    may_complete = threading.Event()
    captured_lock_state: dict = {}

    def gated_wait(peer_arg, deadline_sec: float = 600.0) -> bytes:
        # Snapshot lock-held state at the moment we enter the wait. The
        # bug holds ``_transfer_lock`` here; the fix releases it before
        # calling us.
        # ``acquire(blocking=False)`` returns True only if the lock was
        # free at the time of the call.
        acquired_during_wait = chan._transfer_lock.acquire(blocking=False)
        captured_lock_state["acquired_during_wait"] = acquired_during_wait
        if acquired_during_wait:
            chan._transfer_lock.release()
        waiting_started.set()
        if not may_complete.wait(timeout=10.0):
            raise TimeoutError("test scaffolding: may_complete never set")
        return msgspec.msgpack.encode(PingPongReadAck(ok=True))

    chan._wait_for_transfer_ack = gated_wait

    fake_buffers = [object()]
    fake_spec = {TS_RECEIVER_ID: "test_remote"}

    pull_exc: list = []

    def _run_pull():
        try:
            chan.batched_read(fake_buffers, fake_spec)
        except BaseException as e:  # pragma: no cover - surfaced via assert
            pull_exc.append(e)

    pull_thread = threading.Thread(target=_run_pull, daemon=True)
    pull_thread.start()

    assert waiting_started.wait(timeout=5.0), (
        "batched_read never reached _wait_for_transfer_ack (test scaffolding "
        "failure, unrelated to the lock-discipline bug)"
    )

    # Independent acquire from the main thread, also simulating the
    # listener-side _transfer_loop attempting to service a symmetric
    # incoming PingPongReadRequest while our outgoing pull is in flight.
    acquired_externally = chan._transfer_lock.acquire(timeout=2.0)

    # Let the mocked pull complete regardless of the assertion outcome so
    # the test never hangs on cleanup.
    may_complete.set()
    pull_thread.join(timeout=5.0)
    if acquired_externally:
        chan._transfer_lock.release()

    assert pull_exc == [], f"batched_read raised unexpectedly: {pull_exc[0]!r}"
    assert not pull_thread.is_alive(), "pull thread did not finish in time"

    # The two flavours of the same invariant: at the moment the receive
    # path is parked waiting for the sender's ack, _transfer_lock must
    # be acquirable both from within the wait callback and from an
    # independent thread.
    assert captured_lock_state.get("acquired_during_wait") is True, (
        "REGRESSION: _transfer_lock was held WHILE inside "
        "_wait_for_transfer_ack. Under DP this causes A and B to mutually "
        "deadlock waiting for each other's ack while neither side's "
        "_transfer_loop can take its own lock to deliver one."
    )
    assert acquired_externally, (
        "REGRESSION: _transfer_lock was not acquirable from an independent "
        "thread during ack-wait; the listener-side _transfer_loop would be "
        "starved and the symmetric bidirectional pull would deadlock."
    )


# ---------------------------------------------------------------------------
# Stream-split regression
#
# Under DP with symmetric cross-rank pulls (e.g. DP0_TP2 and DP1_TP2 each
# pulling from each other concurrently), a SINGLE NPU stream for both
# inbound serves (``_handle_read_request`` -> ``agent.send_batch``) and
# outbound pulls (``batched_read`` -> ``agent.recv_batch``) FIFO-queues
# the listener's send_batch BEHIND the connector's already-pending
# recv_batch. The recv_batch can only drain once the peer's send_batch
# fires, but the peer's send_batch is similarly blocked behind its own
# recv -> mutual stall in ``_wait_for_transfer_ack`` while the rest of
# the TP group stalls at the next AllReduce.
#
# The fix is to use SEPARATE NPU streams per direction. These tests
# pin both halves of that contract: (a) ``_send_stream`` is a distinct
# attribute from ``transport_stream``, and (b) the listener methods
# actually use it (and never touch ``transport_stream``).
# ---------------------------------------------------------------------------


def _install_listener_stream_mocks(chan):
    """Wire ``transport_stream`` / ``_send_stream`` / ``agent`` as
    distinguishable MagicMocks so we can assert *which* stream the
    listener handler picks."""
    # Third Party
    from unittest.mock import MagicMock

    chan.transport_stream = MagicMock(name="transport_stream")
    chan.transport_stream.npu_stream = "RECV_STREAM_HANDLE"
    chan._send_stream = MagicMock(name="_send_stream")
    chan._send_stream.npu_stream = "SEND_STREAM_HANDLE"
    chan.agent = MagicMock(name="agent")
    return chan


def _patch_hpp_value_types(monkeypatch):
    """Stub out the C++ ``hpp.PingPongOp`` / ``hpp.PingPongScatterEntry``
    / ``hpp.HcclDataType`` constructors with lightweight Python stand-ins.

    The listener handlers construct these as value objects right before
    handing them to ``agent.send_batch`` / ``agent.scatter_send`` — both
    of which are mocked here — so a plain ``object()`` is sufficient and
    avoids any dependency on the NPU extension being importable in CI."""
    # First Party
    from lmcache_ascend.v1.transfer_channel import (
        hccl_pingpong_channel as _mod,
    )

    class _PPOp:
        def __init__(self, local_addr=0, size=0):
            self.local_addr = local_addr
            self.size = size

    class _PPScatter:
        def __init__(self):
            self.ddr_buf = 0
            self.dst_bufs = []
            self.counts = []
            self.data_type = 0

    class _HcclDataType(int):
        def __new__(cls, v):
            return int.__new__(cls, v)

    monkeypatch.setattr(_mod.hpp, "PingPongOp", _PPOp, raising=False)
    monkeypatch.setattr(_mod.hpp, "PingPongScatterEntry", _PPScatter, raising=False)
    monkeypatch.setattr(_mod.hpp, "HcclDataType", _HcclDataType, raising=False)


def test_pingpong_handle_read_request_uses_send_stream_not_transport(monkeypatch):
    """REGRESSION: ``_handle_read_request`` must submit to ``_send_stream``,
    NOT ``transport_stream``. Reusing ``transport_stream`` puts the
    listener-side send_batch on the same FIFO as the connector-side
    recv_batch, which is the exact stream-FIFO deadlock that wedges DP
    bidirectional pulls (see py-spy: DPx_TPi and DPy_TPi both park in
    ``_wait_for_transfer_ack`` until the test rig hits its TTL)."""
    # First Party
    import threading as _threading

    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        _PeerState,
    )
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_protocol import (
        PingPongReadRequest,
    )

    _patch_hpp_value_types(monkeypatch)
    chan = _make_bare_channel()
    chan._transfer_lock = _threading.Lock()
    _install_listener_stream_mocks(chan)

    # Pre-register a ready listener-side peer.
    peer = _PeerState(
        conn_handle=42,
        transfer_url="",
        transfer_req_socket=None,
        remote_buffers=None,
    )
    peer.ready_event.set()
    chan._listener_peers["remote_dp0_tp2"] = peer

    req = PingPongReadRequest(
        receiver_id="remote_dp0_tp2",
        sender_local_addrs=[0x1000],
        sizes=[4096],
    )

    chan._handle_read_request(req)

    # send_batch must target the SEND stream handle.
    assert chan.agent.send_batch.call_count == 1
    call_args = chan.agent.send_batch.call_args
    assert call_args.args[2] == "SEND_STREAM_HANDLE", (
        "REGRESSION: _handle_read_request submitted agent.send_batch to "
        f"{call_args.args[2]!r} instead of the dedicated _send_stream. "
        "Inbound serves must NOT share transport_stream with outbound pulls "
        "or symmetric DP cross-rank pulls will deadlock on stream FIFO."
    )
    # And the synchronize must drain the SEND stream, not the recv one.
    assert chan._send_stream.synchronize.called
    assert not chan.transport_stream.synchronize.called, (
        "REGRESSION: _handle_read_request called transport_stream.synchronize, "
        "blocking the outbound-pull stream's drain on inbound-send completion."
    )


def test_pingpong_handle_scatter_request_uses_send_stream_not_transport(monkeypatch):
    """Mirror of the read-request test for scatter: ``_handle_scatter_request``
    must route ``agent.scatter_send`` onto ``_send_stream``."""
    # First Party
    import threading as _threading

    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        _PeerState,
    )
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_protocol import (
        PingPongScatterEntryMsg,
        PingPongScatterRequest,
    )

    _patch_hpp_value_types(monkeypatch)
    chan = _make_bare_channel()
    chan._transfer_lock = _threading.Lock()
    _install_listener_stream_mocks(chan)

    peer = _PeerState(
        conn_handle=7,
        transfer_url="",
        transfer_req_socket=None,
        remote_buffers=None,
    )
    peer.ready_event.set()
    chan._listener_peers["remote_dp0_tp2"] = peer

    req = PingPongScatterRequest(
        receiver_id="remote_dp0_tp2",
        entries=[
            PingPongScatterEntryMsg(
                sender_local_addr=0x2000,
                counts=[64],
                data_type=0,
            )
        ],
    )

    chan._handle_scatter_request(req)

    assert chan.agent.scatter_send.call_count == 1
    call_args = chan.agent.scatter_send.call_args
    assert call_args.args[2] == "SEND_STREAM_HANDLE", (
        "REGRESSION: _handle_scatter_request submitted agent.scatter_send to "
        f"{call_args.args[2]!r} instead of _send_stream."
    )
    assert chan._send_stream.synchronize.called
    assert not chan.transport_stream.synchronize.called


def test_pingpong_handle_read_releases_transfer_lock_before_synchronize(monkeypatch):
    """The listener-side synchronize() MUST happen OUTSIDE _transfer_lock.

    The connector-side fix already moved ``_wait_for_transfer_ack`` out
    of the lock. The symmetric listener-side rule is: once ``send_batch``
    is queued on ``_send_stream``, the lock can (and must) be released
    before blocking on the stream drain — otherwise the lock is held for
    the duration of an HCCL transfer, starving any concurrent connector
    submission and re-creating the same bidirectional deadlock by a
    different code path.
    """
    # Third Party
    from unittest.mock import MagicMock

    # First Party
    import threading as _threading

    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        _PeerState,
    )
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_protocol import (
        PingPongReadRequest,
    )

    _patch_hpp_value_types(monkeypatch)
    chan = _make_bare_channel()
    chan._transfer_lock = _threading.Lock()
    _install_listener_stream_mocks(chan)

    peer = _PeerState(
        conn_handle=1,
        transfer_url="",
        transfer_req_socket=None,
        remote_buffers=None,
    )
    peer.ready_event.set()
    chan._listener_peers["remote"] = peer

    captured: dict = {}

    def _record_lock_state():
        # blocking=False acquire reflects whether _transfer_lock was held
        # at the moment synchronize() ran. We immediately release if we
        # grabbed it so the rest of the call body can complete normally.
        acquired = chan._transfer_lock.acquire(blocking=False)
        captured["lock_held_during_synchronize"] = not acquired
        if acquired:
            chan._transfer_lock.release()

    chan._send_stream.synchronize = MagicMock(side_effect=_record_lock_state)

    chan._handle_read_request(
        PingPongReadRequest(
            receiver_id="remote",
            sender_local_addrs=[0x4000],
            sizes=[128],
        )
    )

    assert "lock_held_during_synchronize" in captured, (
        "test scaffolding: synchronize() was never invoked"
    )
    assert captured["lock_held_during_synchronize"] is False, (
        "REGRESSION: _handle_read_request held _transfer_lock across "
        "_send_stream.synchronize(). Under DP this starves the main thread's "
        "concurrent recv_batch acquisition and re-creates the bidirectional "
        "deadlock the stream split was meant to break."
    )


# ---------------------------------------------------------------------------
# Per-peer handshake serialization (NPU-free)
#
# After the p2p_sync rebase, ``AscendP2PBackend.batched_get_blocking`` bridges
# sync callers onto the P2P loop via ``_run_coroutine_threadsafe_blocking``
# and fans out ``_ensure_peer_connection`` for the same peer from several
# coroutines. Without a per-peer ``asyncio.Lock``, two of those coroutines
# can both see ``peer_id not in _connector_peers`` and double-fire the init
# handshake -> ``self.agent.connect`` on the same NPU device. Mirrors the
# fix PR #234 already applied to ``HcclChannel``.
# ---------------------------------------------------------------------------


def test_pingpong_get_peer_handshake_lock_returns_same_lock_per_peer():
    """Repeated calls for the same peer return one shared asyncio.Lock,
    distinct peers get distinct locks."""
    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        HcclPingPongChannel,
    )

    chan = HcclPingPongChannel.__new__(HcclPingPongChannel)
    chan._peer_handshake_locks = {}

    lock_a1 = chan._get_peer_handshake_lock("peer_a")
    lock_a2 = chan._get_peer_handshake_lock("peer_a")
    lock_b = chan._get_peer_handshake_lock("peer_b")

    assert isinstance(lock_a1, asyncio.Lock)
    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b


def test_pingpong_async_lazy_init_serializes_per_peer_handshakes():
    """Two concurrent ``async_lazy_init_peer_connection`` calls for the same
    peer must run their inner ``_async_lazy_init_peer_connection_locked``
    bodies one-at-a-time; calls for different peers must overlap freely."""
    # First Party
    from lmcache_ascend.v1.transfer_channel.hccl_pingpong_channel import (
        HcclPingPongChannel,
    )

    chan = HcclPingPongChannel.__new__(HcclPingPongChannel)
    chan._peer_handshake_locks = {}

    in_flight: Dict[str, int] = {}
    peak: Dict[str, int] = {}

    async def fake_locked(
        local_id, peer_id, peer_init_url, init_side_msg=None
    ):
        in_flight[peer_id] = in_flight.get(peer_id, 0) + 1
        peak[peer_id] = max(peak.get(peer_id, 0), in_flight[peer_id])
        # Yield twice so a sibling coroutine for the same peer gets a real
        # chance to enter without the lock if serialization is broken.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight[peer_id] -= 1
        return None

    chan._async_lazy_init_peer_connection_locked = fake_locked

    async def run():
        await asyncio.gather(
            HcclPingPongChannel.async_lazy_init_peer_connection(
                chan, "me", "peer_a", "url_a", None
            ),
            HcclPingPongChannel.async_lazy_init_peer_connection(
                chan, "me", "peer_a", "url_a", None
            ),
            HcclPingPongChannel.async_lazy_init_peer_connection(
                chan, "me", "peer_b", "url_b", None
            ),
            HcclPingPongChannel.async_lazy_init_peer_connection(
                chan, "me", "peer_b", "url_b", None
            ),
        )

    asyncio.run(run())

    assert peak.get("peer_a", 0) == 1, (
        "REGRESSION: two coroutines entered "
        "_async_lazy_init_peer_connection_locked for peer_a concurrently; "
        "per-peer handshake lock is missing or not held across the inner "
        "call."
    )
    assert peak.get("peer_b", 0) == 1, (
        "Same regression as above, observed via peer_b."
    )
    # Sanity: both peers were actually exercised (avoids a vacuously passing
    # test if the wiring drops calls entirely).
    assert "peer_a" in peak and "peer_b" in peak
