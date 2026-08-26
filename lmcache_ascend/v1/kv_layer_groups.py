# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
import torch

logger = init_logger(__name__)


def _get_tuple_storage_shape(kv_cache: tuple[torch.Tensor, ...]) -> torch.Size:
    """Return the flattened LMCache storage shape for tuple-based KV caches.

    For MLA/DSA, LMCache stores multiple KV tensors as a single contiguous
    hidden dimension, so we must derive the flattened hidden size from the
    whole tuple instead of only looking at the first tensor.
    """
    first_shape = kv_cache[0].shape

    for tensor in kv_cache[1:]:
        if tensor.shape[:2] != first_shape[:2]:
            raise ValueError(
                "All KV tensors in a tuple must share [num_blocks, block_size], "
                f"got {first_shape} and {tensor.shape}"
            )

    if len(kv_cache) == 2 and kv_cache[0].shape == kv_cache[1].shape:
        return first_shape

    total_hidden_dim = sum(tensor.shape[-2] * tensor.shape[-1] for tensor in kv_cache)
    return torch.Size([first_shape[0], first_shape[1], total_hidden_dim])


def _get_kv_cache_group_key_and_info(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[tuple[object, ...], torch.Size, torch.dtype]:
    """Build a stable grouping key plus the LMCache storage shape/dtype."""
    if isinstance(kv_cache, tuple):
        dtypes = tuple(tensor.dtype for tensor in kv_cache)
        if len(set(dtypes)) != 1:
            raise ValueError(
                "Tuple-based KV caches with mixed dtypes are not supported by "
                "LMCache-Ascend."
            )

        shapes = tuple(tensor.shape for tensor in kv_cache)
        storage_shape = _get_tuple_storage_shape(kv_cache)
        return (shapes, dtypes), storage_shape, dtypes[0]

    if isinstance(kv_cache, torch.Tensor):
        return ((kv_cache.shape,), (kv_cache.dtype,)), kv_cache.shape, kv_cache.dtype

    raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")


def _is_310p_nz_layout(shape: torch.Size) -> bool:
    """True when *shape* is Ascend 310P NZ-packed KV (innermost tile == 16).

    310P + LMCache forces NZ via the vllm-ascend adapt patch, so physical
    layer shapes look like:

    * MERGED  ``[2, num_blocks, packed, block_size, 16]``
    * SEPARATE ``[num_blocks, packed, block_size, 16]``

    rather than the 910B ND layouts

    * MERGED  ``[2, num_blocks, block_size, heads, head_dim]``
    * SEPARATE ``[num_blocks, block_size, heads, head_dim]``

    Combining the SOC check with ``shape[-1] == 16`` avoids mis-classifying
    ND caches on 310P (if NZ was not applied) and rare 910B ``head_dim=16``.
    """
    if len(shape) < 4 or int(shape[-1]) != 16:
        return False
    try:
        # First Party
        from lmcache_ascend.v1.npu_connector.npu_connectors import is_310p
    except Exception:
        return False
    return bool(is_310p())


def patched_hidden_dim_size(self) -> int:
    """Return the size of the hidden dimension in this group.

    ``metadata.get_shapes()`` stores LMCache chunks as
    ``[kv_size, num_layers, num_tokens, hidden_dim]``. The transfer kernels
    then take ``hidden_dims = key_value.size(-1)``, so a wrong value here
    silently corrupts store/retrieve.
    """
    # hidden_dim_size = num_heads * head_size
    if len(self.shape) == 5:
        # MERGED: ND [2, nb, bs, heads, hd] or NZ [2, nb, packed, bs, 16]
        if _is_310p_nz_layout(self.shape):
            return self.shape[2] * self.shape[4]
        return self.shape[3] * self.shape[4]
    elif len(self.shape) == 4:
        # SEPARATE: ND (nb, bs, heads, hd) or NZ (nb, packed, bs, 16)
        # shape[0]==1 would be an invalid single-block / MLA-confused layout.
        if self.shape[0] == 1:
            raise ValueError(f"Invalid shape for hidden dim size: {self.shape}")

        if _is_310p_nz_layout(self.shape):
            return self.shape[1] * self.shape[3]
        return self.shape[2] * self.shape[3]
    elif len(self.shape) == 3:
        # MLA / DSA flattened storage shape [nb, bs, total_hidden]
        return self.shape[2]
    else:
        raise ValueError(f"Invalid shape: {self.shape}")


def build_kv_layer_groups(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """Build KV layer groups structure by analyzing each layer's shape and dtype.

    Layers with the same shape and dtype are grouped together. This is useful
    because different layers may have different structures (especially the
    last dimension head_size may differ between groups), and different groups
    may have different dtypes.

    If layer groups are already built (non-empty list), this method does nothing.

    Args:
        kv_caches: Dictionary mapping layer names to KV cache tensors.
    """
    # Skip if already built (non-empty list)
    if len(self.kv_layer_groups) > 0:
        return

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    # Group layers by (shape, dtype) in a single loop
    groups_dict: dict[tuple[object, ...], list[tuple[str, int]]] = defaultdict(list)
    group_infos: dict[tuple[object, ...], tuple[torch.Size, torch.dtype]] = {}

    for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
        key, shape, dtype = _get_kv_cache_group_key_and_info(kv_cache)
        groups_dict[key].append((layer_name, idx))
        group_infos[key] = (shape, dtype)

    # Build KVLayerGroupInfo list
    # Sort groups by the first layer index to maintain order
    def _get_first_layer_index(shape_dtype_key):
        """Get the index of the first layer in a layer group."""
        layer_group = groups_dict[
            shape_dtype_key
        ]  # list of (layer_name, layer_index) tuples
        first_layer_info = layer_group[0]  # first (layer_name, layer_index) tuple
        layer_index = first_layer_info[1]  # extract the layer index
        return layer_index

    sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

    kv_layer_groups: list[KVLayerGroupInfo] = []
    for key in sorted_keys:
        shape, dtype = group_infos[key]
        layers = groups_dict[key]
        layer_names, layer_indices = zip(*layers, strict=False)

        group_info = KVLayerGroupInfo(
            layer_names=list(layer_names),
            layer_indices=list(layer_indices),
            shape=shape,
            dtype=dtype,
        )
        kv_layer_groups.append(group_info)

    # Store the built groups
    self.kv_layer_groups = kv_layer_groups

    # Print the group structure
    logger.info("KV layer groups: %s", kv_layer_groups)
