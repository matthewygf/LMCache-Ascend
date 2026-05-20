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
Async scatter from a host-resident contiguous buffer on the sender to per-page NPU destinations
on the receiver. Validates the H2D-only v1 of BatchChannel.ScatterSend end-to-end.
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
