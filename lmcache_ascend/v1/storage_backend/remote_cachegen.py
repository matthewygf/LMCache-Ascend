# SPDX-License-Identifier: Apache-2.0
"""Host relocation for CacheGen remote gets on Ascend.

``CacheGenDeserializer`` returns unmanaged NPU tensors. Remote retrieve must
copy them into allocator-managed host (CPU) buffers before the NPU connector
H2Ds them on ``load_stream``. The D2H must be synchronized first; otherwise
``copy_(..., non_blocking=True)`` races the subsequent H2D and can load
partial / stale KV into the serving cache.
"""

# Standard
from typing import Any, Callable, List, Optional, Sequence


def relocate_cachegen_bufs_to_host(
    allocator,
    source_bufs: Sequence[Optional[Any]],
    synchronize_fn: Callable[[], None],
) -> List[Optional[Any]]:
    """Copy CacheGen device tensors into host buffers, then synchronize.

    Args:
        allocator: Backend that provides ``allocate(shape, dtype, fmt=...)``.
        source_bufs: Deserialized CacheGen memory objects (may include ``None``
            misses). Kept alive by the caller until this returns so the async
            D2H cannot race freelist / GC of the sources.
        synchronize_fn: Device synchronize (e.g. ``torch.npu.synchronize`` /
            ``torch.cuda.synchronize``).

    Returns:
        Host-resident memory objects aligned 1:1 with ``source_bufs``. Entries
        are ``None`` when the source was missing or allocation failed.
    """
    target_bufs: List[Optional[Any]] = []
    for source_buf in source_bufs:
        if source_buf is None or source_buf.tensor is None:
            target_bufs.append(None)
            continue

        fmt = getattr(getattr(source_buf, "meta", None), "fmt", None)
        target_buf = allocator.allocate(
            source_buf.tensor.shape,
            source_buf.tensor.dtype,
            fmt=fmt,
        )
        if target_buf is None or target_buf.tensor is None:
            target_bufs.append(None)
            continue

        target_buf.tensor.copy_(source_buf.tensor, non_blocking=True)
        target_bufs.append(target_buf)

    # Callers immediately H2D these host buffers on a different stream.
    # Upstream ``allocate_and_copy_objects`` and MindSpore's 310P path both
    # synchronize after the same non_blocking pattern; omit that and the
    # load stream can read incomplete pages → silent wrong KV.
    synchronize_fn()
    return target_bufs
