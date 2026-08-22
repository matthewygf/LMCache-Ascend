# SPDX-License-Identifier: Apache-2.0
"""Regression: MindSpore connector 310P NZ page_buffer_size must match #66.

PyTorch ``VLLMPagedMemNPUConnectorV2`` fixed NZ slot counting in #66. The
MindSpore fork kept ``shape[1] * shape[2]`` (packed dim) which under-counts
slots (``num_blocks * packed`` instead of ``num_blocks * block_size``) and
feeds the wrong ``page_buffer_size`` into ``multi_layer_kv_transfer_ms``.
"""

# Standard
from pathlib import Path

# Third Party
import pytest
import torch

CONNECTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "lmcache_ascend"
    / "mindspore"
    / "v1"
    / "npu_connector.py"
)


def _page_buffer_size_merged(shape: torch.Size, is_310p: bool) -> int:
    """Mirror MindSpore connector MERGED_KV size math after the fix."""
    if is_310p:
        block_size = shape[-2]
        return shape[1] * block_size
    return shape[1] * shape[2]


def _page_buffer_size_separate(shape: torch.Size, is_310p: bool) -> int:
    """Mirror MindSpore connector SEPARATE_KV size math after the fix."""
    if is_310p:
        block_size = shape[-2]
        return shape[0] * block_size
    return shape[0] * shape[1]


@pytest.mark.parametrize(
    "num_blocks,packed,block_size",
    [
        (100, 64, 128),
        (32, 32, 16),
        (8, 48, 64),
    ],
)
def test_310p_merged_nz_uses_block_size_not_packed(num_blocks, packed, block_size):
    # NZ MERGED: [2, num_blocks, packed, block_size, 16]
    shape = torch.Size([2, num_blocks, packed, block_size, 16])
    got = _page_buffer_size_merged(shape, is_310p=True)
    assert got == num_blocks * block_size
    # Pre-fix bug: shape[1] * shape[2] == num_blocks * packed
    assert shape[1] * shape[2] == num_blocks * packed
    assert got != num_blocks * packed


@pytest.mark.parametrize(
    "num_blocks,packed,block_size",
    [
        (100, 64, 128),
        (32, 32, 16),
    ],
)
def test_310p_separate_nz_uses_block_size_not_packed(num_blocks, packed, block_size):
    # NZ SEPARATE: [num_blocks, packed, block_size, 16]
    shape = torch.Size([num_blocks, packed, block_size, 16])
    got = _page_buffer_size_separate(shape, is_310p=True)
    assert got == num_blocks * block_size
    assert shape[0] * shape[1] == num_blocks * packed
    assert got != num_blocks * packed


def test_910b_merged_unchanged():
    num_blocks, block_size, num_heads, head_dim = 100, 128, 8, 128
    shape = torch.Size([2, num_blocks, block_size, num_heads, head_dim])
    assert _page_buffer_size_merged(shape, is_310p=False) == num_blocks * block_size


def test_mindspore_connector_source_has_310p_page_buffer_branch():
    """Guard against regressing to the pre-#66 MindSpore formula."""
    src = CONNECTOR_PATH.read_text(encoding="utf-8")
    assert "if self.is_310p:" in src
    assert "self.page_buffer_size = first_tensor.shape[1] * self.block_size" in src
    assert "self.page_buffer_size = first_tensor.shape[0] * self.block_size" in src
    # Ensure we did not leave the unconditional pre-fix assignment.
    assert "self.block_size = first_tensor.shape[-2]\n            if self.kv_format" not in src
