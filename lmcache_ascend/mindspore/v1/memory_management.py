# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
# Copyright 2025 Ilya Yanok, Serapheim Dimitropoulos.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Optional, Union
import ctypes

# Third Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.memory_management import (
    AddressManager,
    TensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.system_detection import NUMAMapping
import mindspore as ms
import numpy as np
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

# Local
from ._tensor import (
    get_data_ptr,
    get_dtype_compat,
    get_element_size,
    get_numel,
    view_and_shape,
)

logger = init_logger(__name__)


def _allocate_cpu_memory(
    size: int,
    numa_mapping: Optional[NUMAMapping] = None,
    shm_name: Optional[str] = None,
) -> torch.Tensor:
    # shm_name is accepted for API compatibility with lmcache>=0.4.3 PinMemoryAllocator.
    # MindSpore CPU buffers are pinned host allocations, not shared-memory segments.
    if shm_name is not None:
        logger.warning(
            "shm_name=%s is ignored by MindSpore _allocate_cpu_memory; "
            "falling back to pinned host allocation.",
            shm_name,
        )

    if numa_mapping:
        if torch.cuda.is_available():
            current_device_id = ms.get_current_device().device_id
        else:
            current_device_id = 0
        gpu_to_numa_mapping = numa_mapping.gpu_to_numa_mapping
        assert current_device_id in gpu_to_numa_mapping, (
            f"Current device {current_device_id} is not in the GPU NUMA mapping."
        )
        numa_id = gpu_to_numa_mapping[current_device_id]
        ptr = lmc_ops.alloc_pinned_numa_ptr(size, numa_id)
    else:
        ptr = lmc_ops.alloc_pinned_ptr(size, 0)

    array_type = ctypes.c_uint8 * size
    buf = array_type.from_address(ptr)
    buffer = np.frombuffer(buf, dtype=np.uint8)

    return buffer


class NumpyAndTensorMemoryObj(TensorMemoryObj):
    @property
    def tensor(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        if not self.valid:
            logger.warning("Trying to access an invalidated MemoryObj")
            return None
        assert self.meta.dtype is not None
        # AddressManager may return an aligned phy_size larger than the logical
        # tensor; trim to get_size() before dtype/shape views (matches upstream).
        return view_and_shape(
            self.raw_data[: self.get_size()], self.meta.dtype, self.meta.shape
        )

    @property
    def byte_array(self) -> bytes:
        kv_chunk = self.tensor
        assert kv_chunk is not None
        num_bytes = get_numel(kv_chunk) * get_element_size(kv_chunk)
        ptr = get_data_ptr(kv_chunk)
        ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
        byte_array = (ctypes.c_ubyte * num_bytes).from_address(
            ctypes.addressof(ubyte_ptr.contents)
        )
        return memoryview(byte_array)


class NumpyAndTensorMemoryAllocator(TensorMemoryAllocator):
    """TensorMemoryAllocator variant that can back the pool with a NumPy buffer.

    lmcache>=0.3.15 moved free-list management into AddressManager and dropped
    TensorMemoryAllocator.ALIGN_BYTES / explicit_list. This class must stay on
    that API so MindSpore's TensorMemoryAllocator monkeypatch can import and
    free successfully against lmcache==0.4.3.
    """

    def __init__(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        align_bytes: int = AddressManager.ALIGN_BYTES,
        init_address_space: int | None = None,
    ):
        # NOTE (Gingfung:) use reshape so NumPy uint8 pools work; torch.view()
        # is not available on ndarray.
        assert self._is_uint8_type(tensor)
        self.buffer = tensor.reshape(-1)

        self.address_manager = AddressManager(
            get_numel(self.buffer) if init_address_space is None else init_address_space,
            align_bytes,
        )

        # For debugging purposes
        self.num_active_allocations = 0

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    def _is_uint8_type(self, tensor: Union[torch.Tensor, np.ndarray]):
        if isinstance(tensor, np.ndarray):
            return tensor.dtype == np.uint8
        elif isinstance(tensor, torch.Tensor):
            return tensor.dtype == torch.uint8
        else:
            raise ValueError(f"tensor of type: {type(tensor)} not supported.")

    def _get_buffer_slice(
        self, start: int, size: int
    ) -> Union[torch.Tensor, np.ndarray]:
        """Hook: Get buffer slice for torch or NumPy pools."""
        return self.buffer[start : start + size]

    def _adapt_shapes_and_dtypes(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ):
        shapes, dtypes = super()._adapt_shapes_and_dtypes(shapes, dtypes)
        # NumPy views cannot interpret torch.dtype directly; normalize MindSpore
        # / torch dtypes to a NumPy-compatible dtype when the pool is ndarray.
        if isinstance(self.buffer, np.ndarray):
            dtypes = [self._dtype_for_numpy_buffer(dtype) for dtype in dtypes]
        return shapes, dtypes

    @staticmethod
    def _dtype_for_numpy_buffer(dtype):
        dtype = get_dtype_compat(dtype)
        if isinstance(dtype, torch.dtype):
            # torch.bfloat16 has no stable numpy dtype without MindSpore helpers.
            if dtype == torch.bfloat16:
                return np.float16
            return np.dtype(str(dtype).replace("torch.", ""))
        return dtype
