# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MindSpore NumpyAndTensorMemoryAllocator vs lmcache 0.4.3."""

# Standard
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys

# Third Party
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MS_V1 = REPO_ROOT / "lmcache_ascend" / "mindspore" / "v1"


def _ensure_pkg(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    mod = ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def _load_module(fullname: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(fullname, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_import_stubs():
    """Import the MindSpore allocator without a real MindSpore / Ascend runtime."""
    build_info = ModuleType("lmcache_ascend._build_info")
    build_info.__framework_name__ = "mindspore"
    sys.modules["lmcache_ascend._build_info"] = build_info

    if "mindspore" not in sys.modules:
        mindspore = ModuleType("mindspore")
        mindspore.get_current_device = lambda: SimpleNamespace(device_id=0)
        mindspore.dtype = SimpleNamespace(Type=type("MSType", (), {}))
        mindspore.dtype_to_nptype = lambda dtype: np.float16
        sys.modules["mindspore"] = mindspore
        sys.modules["mindspore.common"] = ModuleType("mindspore.common")
        np_dtype = ModuleType("mindspore.common.np_dtype")
        np_dtype.bfloat16 = np.float16
        sys.modules["mindspore.common.np_dtype"] = np_dtype

    if "lmcache_ascend.c_ops" not in sys.modules:
        c_ops = ModuleType("lmcache_ascend.c_ops")
        c_ops.alloc_pinned_ptr = lambda size, _flags: 0
        c_ops.alloc_pinned_numa_ptr = lambda size, _numa: 0
        sys.modules["lmcache_ascend.c_ops"] = c_ops

    # Avoid executing lmcache_ascend/__init__.py (framework-wide patches).
    _ensure_pkg("lmcache_ascend", REPO_ROOT / "lmcache_ascend")
    _ensure_pkg("lmcache_ascend.mindspore", REPO_ROOT / "lmcache_ascend" / "mindspore")
    _ensure_pkg("lmcache_ascend.mindspore.v1", MS_V1)

    if "lmcache_ascend.mindspore.v1._tensor" not in sys.modules:
        _load_module("lmcache_ascend.mindspore.v1._tensor", MS_V1 / "_tensor.py")

    if "lmcache_ascend.mindspore.v1.memory_management" not in sys.modules:
        return _load_module(
            "lmcache_ascend.mindspore.v1.memory_management",
            MS_V1 / "memory_management.py",
        )
    return sys.modules["lmcache_ascend.mindspore.v1.memory_management"]


@pytest.fixture(scope="module")
def allocator_cls():
    mm = _install_import_stubs()
    # Mirror mindspore/__init__._patch_memory_management so inherited
    # allocate()/batched_allocate() construct NumpyAndTensorMemoryObj.
    # Third Party
    import lmcache.v1.memory_management as upstream_mm

    upstream_mm.TensorMemoryObj = mm.NumpyAndTensorMemoryObj
    return mm.NumpyAndTensorMemoryAllocator


def test_allocator_constructs_with_address_manager(allocator_cls):
    # Third Party
    from lmcache.v1.memory_management import AddressManager, TensorMemoryAllocator

    # Upstream lmcache 0.4.3 moved ALIGN_BYTES off TensorMemoryAllocator.
    assert not hasattr(TensorMemoryAllocator, "ALIGN_BYTES")

    buf = np.zeros(AddressManager.ALIGN_BYTES * 2, dtype=np.uint8)
    alloc = allocator_cls(buf)
    assert hasattr(alloc, "address_manager")
    assert alloc.address_manager.get_heap_size() == buf.size
    assert alloc.memcheck()


def test_numpy_pool_allocate_free_roundtrip(allocator_cls):
    # Third Party
    from lmcache.v1.memory_management import MemoryFormat

    buf = np.zeros(1024 * 1024, dtype=np.uint8)
    alloc = allocator_cls(buf)
    shape = torch.Size([2, 8, 16])
    dtype = torch.float16

    obj = alloc.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
    assert obj is not None
    assert obj.is_valid()
    assert obj.get_shape() == shape
    assert obj.meta.address >= 0
    assert obj.meta.phy_size >= shape.numel() * dtype.itemsize
    assert alloc.num_active_allocations == 1

    view = obj.tensor
    assert view is not None
    assert tuple(view.shape) == tuple(shape)
    assert view.dtype == np.float16

    obj.ref_count_down()
    assert not obj.is_valid()
    assert alloc.num_active_allocations == 0
    assert alloc.memcheck()


def test_numpy_pool_batched_allocate_free(allocator_cls):
    buf = np.zeros(1024 * 1024, dtype=np.uint8)
    alloc = allocator_cls(buf)
    shape = torch.Size([4, 8])
    dtype = torch.float16

    objs = alloc.batched_allocate(shape, dtype, batch_size=3)
    assert objs is not None
    assert len(objs) == 3
    assert alloc.num_active_allocations == 3

    alloc.batched_free(objs)
    assert alloc.num_active_allocations == 0
    assert alloc.memcheck()


def test_old_align_bytes_default_would_fail_on_upstream():
    """Document the pre-fix import crash: TensorMemoryAllocator.ALIGN_BYTES is gone."""
    # Third Party
    from lmcache.v1.memory_management import TensorMemoryAllocator

    with pytest.raises(AttributeError):
        _ = TensorMemoryAllocator.ALIGN_BYTES
