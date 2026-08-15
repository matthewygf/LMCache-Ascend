# SPDX-License-Identifier: Apache-2.0
"""Regression tests for CacheGen remote D2H host relocation.

Without a device synchronize after ``copy_(..., non_blocking=True)``, the
subsequent ``load_stream`` H2D in retrieve can race the D2H and load wrong KV.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, call

# First Party
from lmcache_ascend.v1.storage_backend.remote_cachegen import (
    relocate_cachegen_bufs_to_host,
)


def _source(shape=(2, 4), dtype="fake-dtype", fmt="KV_2LTD"):
    tensor = MagicMock()
    tensor.shape = shape
    tensor.dtype = dtype
    src = SimpleNamespace(tensor=tensor, meta=SimpleNamespace(fmt=fmt))
    return src


def test_relocate_synchronizes_after_non_blocking_copies():
    src0 = _source()
    src1 = _source(shape=(2, 8))
    dst0_tensor = MagicMock()
    dst1_tensor = MagicMock()
    dst0 = SimpleNamespace(tensor=dst0_tensor)
    dst1 = SimpleNamespace(tensor=dst1_tensor)

    allocator = MagicMock()
    allocator.allocate.side_effect = [dst0, dst1]
    sync = MagicMock()

    result = relocate_cachegen_bufs_to_host(allocator, [src0, src1], sync)

    assert result == [dst0, dst1]
    assert allocator.allocate.call_args_list == [
        call(src0.tensor.shape, src0.tensor.dtype, fmt="KV_2LTD"),
        call(src1.tensor.shape, src1.tensor.dtype, fmt="KV_2LTD"),
    ]
    dst0_tensor.copy_.assert_called_once_with(src0.tensor, non_blocking=True)
    dst1_tensor.copy_.assert_called_once_with(src1.tensor, non_blocking=True)
    sync.assert_called_once_with()


def test_relocate_preserves_none_and_oom_slots_but_still_syncs():
    src_ok = _source()
    dst_tensor = MagicMock()
    dst = SimpleNamespace(tensor=dst_tensor)

    allocator = MagicMock()
    # Second allocate fails (OOM); third source is a miss (None).
    allocator.allocate.side_effect = [dst, None]
    sync = MagicMock()

    result = relocate_cachegen_bufs_to_host(
        allocator, [src_ok, _source(), None], sync
    )

    assert result == [dst, None, None]
    dst_tensor.copy_.assert_called_once_with(src_ok.tensor, non_blocking=True)
    # Sync even when some slots failed so any successful D2H is visible.
    sync.assert_called_once_with()


def test_relocate_skips_source_with_none_tensor():
    bad = SimpleNamespace(tensor=None, meta=SimpleNamespace(fmt="KV_2LTD"))
    allocator = MagicMock()
    sync = MagicMock()

    result = relocate_cachegen_bufs_to_host(allocator, [bad], sync)

    assert result == [None]
    allocator.allocate.assert_not_called()
    sync.assert_called_once_with()
