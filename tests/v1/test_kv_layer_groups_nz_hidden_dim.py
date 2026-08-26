# SPDX-License-Identifier: Apache-2.0
"""NZ vs ND hidden_dim derivation for Ascend KV layer groups.

On 310P, physical KV tensors use NZ packing (innermost dim == 16).
``patched_hidden_dim_size`` must return ``num_heads * head_dim`` for both
ND (910B) and NZ (310P) layouts, because ``metadata.get_shapes()`` feeds
the LMCache chunk last-dim straight into ``multi_layer_kv_transfer*`` as
``hidden_dims = key_value.size(-1)``.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.kv_layer_groups import (
    _is_310p_nz_layout,
    patched_hidden_dim_size,
)


def _group(shape):
    return SimpleNamespace(shape=torch.Size(shape))


class TestPatchedHiddenDimSize:
    def test_910b_separate_nd(self):
        # [num_blocks, block_size, heads, head_dim]
        shape = (1024, 128, 8, 128)
        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
            return_value=False,
        ):
            assert patched_hidden_dim_size(_group(shape)) == 8 * 128
            assert not _is_310p_nz_layout(torch.Size(shape))

    def test_910b_merged_nd(self):
        # [2, num_blocks, block_size, heads, head_dim]
        shape = (2, 1024, 128, 8, 128)
        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
            return_value=False,
        ):
            assert patched_hidden_dim_size(_group(shape)) == 8 * 128

    def test_310p_separate_nz(self):
        heads, head_dim, block_size = 8, 128, 128
        packed = heads * head_dim // 16
        # [num_blocks, packed, block_size, 16]
        shape = (1024, packed, block_size, 16)
        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
            return_value=True,
        ):
            assert _is_310p_nz_layout(torch.Size(shape))
            assert patched_hidden_dim_size(_group(shape)) == heads * head_dim
            # Pre-fix ND formula would return block_size * 16 (== 2048 here).
            assert patched_hidden_dim_size(_group(shape)) != block_size * 16

    def test_310p_merged_nz(self):
        heads, head_dim, block_size = 8, 128, 128
        packed = heads * head_dim // 16
        # [2, num_blocks, packed, block_size, 16]
        shape = (2, 1024, packed, block_size, 16)
        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
            return_value=True,
        ):
            assert patched_hidden_dim_size(_group(shape)) == heads * head_dim
            assert patched_hidden_dim_size(_group(shape)) != block_size * 16

    def test_310p_nd_not_misclassified_as_nz(self):
        # If NZ adapt patch is absent, 310P may still expose ND shapes.
        shape = (1024, 128, 8, 128)
        with patch(
            "lmcache_ascend.v1.npu_connector.npu_connectors.is_310p",
            return_value=True,
        ):
            assert not _is_310p_nz_layout(torch.Size(shape))
            assert patched_hidden_dim_size(_group(shape)) == 8 * 128

    def test_mla_flattened_3d(self):
        shape = (1024, 128, 576)
        assert patched_hidden_dim_size(_group(shape)) == 576

    def test_invalid_leading_one_raises(self):
        with pytest.raises(ValueError):
            patched_hidden_dim_size(_group((1, 128, 8, 128)))
