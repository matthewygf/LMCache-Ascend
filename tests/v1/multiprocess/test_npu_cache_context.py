# SPDX-License-Identifier: Apache-2.0

"""NPU-side buffer tests for NPUCacheContext.

Runs the **same** test classes and methods as upstream
``LMCache/tests/v1/multiprocess/test_gpu_context.py`` (via ``lmcache_tests``),
by patching that module's ``_make_context`` to build ``NPUCacheContext`` on
``npu:0`` instead of ``GPUCacheContext`` on ``cuda``.

This keeps assertions identical to upstream so any divergence is caught when
LMCache bumps.  Full ``__init__`` / KVCache / stream coverage lives in
``test_npu_cache_context_init.py``.
"""

# First Party — register ``lmcache_tests`` alias before importing from it
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
import pytest
import torch
import torch_npu  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="NPU not available"
)

# First Party
from lmcache_ascend.v1.multiprocess.npu_context import NPUCacheContext  # noqa: E402


def _make_context(
    num_layers: int = 4,
    num_heads: int = 8,
    head_size: int = 128,
    is_mla: bool = False,
    chunk_size: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> NPUCacheContext:
    """Same contract as upstream ``test_gpu_context._make_context``, NPU device."""
    ctx = object.__new__(NPUCacheContext)
    ctx.is_mla_ = is_mla
    ctx.num_layers_ = num_layers
    ctx.hidden_dim_size_ = num_heads * head_size
    ctx.max_batch_size = 4

    kv_dim = 1 if is_mla else 2
    total_tokens = chunk_size * ctx.max_batch_size
    shape = torch.Size((kv_dim, num_layers, total_tokens, num_heads * head_size))
    ctx.tmp_gpu_buffer_ = torch.empty(shape, dtype=dtype, device="npu:0")
    return ctx


# Load upstream test module and swap the factory so imported test methods
# resolve ``_make_context`` in that module's globals at runtime.
import lmcache_tests.v1.multiprocess.test_gpu_context as _gpu_context_test_mod  # noqa: E402

_gpu_context_test_mod._make_context = _make_context

from lmcache_tests.v1.multiprocess.test_gpu_context import (  # noqa: E402
    TestGetTmpGpuBuffer,
    TestGetTmpGpuBufferBatched,
)

__all__ = [
    "TestGetTmpGpuBuffer",
    "TestGetTmpGpuBufferBatched",
]
