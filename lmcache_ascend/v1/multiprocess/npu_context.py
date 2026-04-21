# SPDX-License-Identifier: Apache-2.0
""""""

# Standard
from typing import Any, Callable

# Third Party
import torch
import torch_npu  # noqa: F401 — registers torch.npu namespace

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    discover_gpu_kv_format,
    get_attention_backend,
    get_block_size,
    get_concrete_gpu_kv_shape,
    get_dtype,
    get_gpu_kv_shape_description,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    is_mla,
)
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.gpu_context import (
    GPUCacheContext,
    list_to_gpu_tensor,        # device-agnostic: array.array("l") + .to(device)
    unwrap_kv_cache_tensors,   # device-agnostic: calls wrapper.to_tensor()
)
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_HAS_LAUNCH_HOST_FUNC = (
    hasattr(torch, "npu")
    and hasattr(torch.npu, "_launch_host_func")
    and hasattr(torch.npu, "_subscribe_report")
    and hasattr(torch.npu, "_unsubscribe_report")
)


class NPUStreamHostFuncAdapter:
    """
    Duck-type replacement for cupy.cuda.ExternalStream as consumed by
    MPCacheEngine.  Exposes launch_host_func(callback, user_data) backed by
    torch.npu._launch_host_func → c10_npu::launch_callback → AclrtLaunchCallback.

    Lifecycle:
    - Construction: torch.npu._subscribe_report(stream) registers the ACL
      report thread so queued callbacks are dispatched.
    - close(): torch.npu._unsubscribe_report(stream) deletes the PyFuncStruct
      entries in torch_npu's callbacks[stream] map and must be called when
      the owning NPUCacheContext is torn down.

    Fallback:
    - If the subscribe/launch APIs are absent on a build, falls back to
      stream.synchronize() + direct call (correct but blocking).
    """

    def __init__(self, npu_stream: torch.npu.Stream) -> None:
        self._stream = npu_stream
        self._subscribed = False
        if _HAS_LAUNCH_HOST_FUNC:
            logger.debug("Subscribing to ACL report callbacks on NPU stream.")
            torch.npu._subscribe_report(self._stream)
            self._subscribed = True
        else:
            logger.warning(
                "torch.npu._launch_host_func not available; "
                "falling back to synchronous host callbacks on NPU stream."
            )

    def launch_host_func(self, callback: Callable, user_data: Any) -> None:
        """
        Enqueue callback(user_data) to run after all work on the stream
        completes.  Matches the CuPy stream.launch_host_func(fn, arg) signature
        used by upstream MPCacheEngine.
        """
        if _HAS_LAUNCH_HOST_FUNC and self._subscribed:
            torch.npu._launch_host_func(self._stream, callback, user_data)
        else:
            self._stream.synchronize()
            callback(user_data)

    def close(self) -> None:
        """Unsubscribe from ACL report callbacks and release PyFuncStruct memory."""
        if self._subscribed:
            try:
                torch.npu._unsubscribe_report(self._stream)
            except Exception as exc:
                logger.warning("_unsubscribe_report failed: %s", exc)
            finally:
                self._subscribed = False

    def __del__(self) -> None:
        self.close()


class NPUCacheContext(GPUCacheContext):
    """
    NPU analogue of upstream GPUCacheContext for Ascend.

    Inherits all device-agnostic methods unchanged:
        stage_block_ids, get_kv_buffer_shape, get_tmp_gpu_buffer,
        get_tmp_gpu_buffer_batched, get_slot_mapping_tensor,
        cache_size_per_token, and all layout @property accessors.

    Overrides:
        __init__      — NPU streams + lmcache_ascend.c_ops for PageBufferShapeDesc.
        stream        — returns torch.npu.Stream
        cupy_stream   — returns NPUStreamHostFuncAdapter
        high_priority_stream        — returns torch.npu.Stream
        high_priority_cupy_stream   — returns NPUStreamHostFuncAdapter
        close / __del__ — unsubscribe ACL report callbacks.
    """

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_chunk_size: int = 256,
        layout_hints: LayoutHints | None = None,
    ) -> None:
        self.kv_caches_ = unwrap_kv_cache_tensors(kv_caches)
        self.device_ = self.kv_caches_[0].device

        pointers_list = [t.data_ptr() for t in self.kv_caches_]
        self.kv_cache_pointers_ = list_to_gpu_tensor(pointers_list, self.device_)

        self.gpu_kv_format_ = discover_gpu_kv_format(
            self.kv_caches_,
            EngineType.VLLM,
            layout_hints=layout_hints,
        )
        self.is_mla_ = is_mla(self.gpu_kv_format_)
        self.num_layers_ = get_num_layers(self.kv_caches_, self.gpu_kv_format_)
        self.num_blocks_ = get_num_blocks(self.kv_caches_, self.gpu_kv_format_)
        self.block_size_ = get_block_size(self.kv_caches_, self.gpu_kv_format_)
        self.hidden_dim_size_ = get_hidden_dim_size(
            self.kv_caches_, self.gpu_kv_format_
        )
        self.num_heads_ = get_num_heads(self.kv_caches_, self.gpu_kv_format_)
        self.head_size_ = get_head_size(self.kv_caches_, self.gpu_kv_format_)

        self.shape_desc_ = lmc_ops.PageBufferShapeDesc()
        self.shape_desc_.kv_size = 1 if self.is_mla_ else 2
        self.shape_desc_.nl = self.num_layers_
        self.shape_desc_.nb = self.num_blocks_
        self.shape_desc_.bs = self.block_size_
        self.shape_desc_.nh = self.num_heads_
        self.shape_desc_.hs = self.head_size_
        self.shape_desc_.element_size = self.kv_caches_[0].element_size()

        _MAX_BLOCK_IDS = 1_000_000
        self.block_ids_buffer_ = torch.empty(
            _MAX_BLOCK_IDS, dtype=torch.long, device=self.device_
        )

        block_ids = torch.arange(
            0, self.num_blocks_, dtype=torch.long, device=self.device_
        ).unsqueeze(1)
        offsets = torch.arange(
            0, self.block_size_, dtype=torch.long, device=self.device_
        ).unsqueeze(0)
        self.slot_mapping_tensor_ = (offsets + block_ids * self.block_size_).reshape(
            (self.num_blocks_, self.block_size_)
        )

        self.max_batch_size = 4
        tmp_buffer_shape = self.get_kv_buffer_shape(
            lmcache_chunk_size * self.max_batch_size
        )
        self.tmp_gpu_buffer_ = torch.empty(
            tmp_buffer_shape, dtype=self.dtype, device=self.device_
        )

        self.npu_stream_ = torch.npu.Stream(self.device_.index, priority=0)
        self.npu_cupy_stream_ = NPUStreamHostFuncAdapter(self.npu_stream_)

        # Second stream for high-priority path (distinct from primary).
        # torch_npu uses negative priority for the high-priority pool (see
        # NPUStream.cpp: isHighPriority ? -kMaxStreamPriorities + 1 : 0).
        self.high_priority_npu_stream_ = torch.npu.Stream(
            self.device_.index, priority=-1
        )
        self.high_priority_npu_cupy_stream_ = NPUStreamHostFuncAdapter(
            self.high_priority_npu_stream_
        )

        logger.info(
            "Initialized NPU streams on device %s (layers=%d, block_size=%d)",
            str(self.device_),
            self.num_layers_,
            self.block_size_,
        )

    @property
    def stream(self) -> torch.npu.Stream:
        """Primary NPU stream for KV cache operations."""
        return self.npu_stream_

    @property
    def cupy_stream(self) -> NPUStreamHostFuncAdapter:
        """NPU host-func adapter with the same launch_host_func interface as CuPy."""
        return self.npu_cupy_stream_

    @property
    def high_priority_stream(self) -> torch.npu.Stream:
        return self.high_priority_npu_stream_

    @property
    def high_priority_cupy_stream(self) -> NPUStreamHostFuncAdapter:
        return self.high_priority_npu_cupy_stream_

    def close(self) -> None:
        """Unsubscribe both streams from ACL report callbacks."""
        # Partial contexts (e.g. tests using object.__new__ + minimal fields)
        # may not have stream adapters; skip in that case.
        if hasattr(self, "npu_cupy_stream_"):
            self.npu_cupy_stream_.close()
        if hasattr(self, "high_priority_npu_cupy_stream_"):
            self.high_priority_npu_cupy_stream_.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
