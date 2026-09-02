# SPDX-License-Identifier: Apache-2.0
"""Guard: layerwise/blend connectors must not be created on Ascend 310P.

single_layer_kv_transfer assumes ND [blocks, block_size, heads, head_dim]
layout. 310P vLLM caches are NZ; using layerwise there mis-addresses slots
and over-copies (OOB / silent wrong KV).
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
import pytest
import torch

# First Party
from lmcache_ascend.v1.npu_connector import CreateNPUConnector


def _fake_metadata():
    meta = MagicMock()
    meta.worker_id = 0
    meta.use_mla = False
    meta.kv_shape = (2, 2, 256, 8, 128)
    meta.kv_dtype = torch.bfloat16
    return meta


@patch("lmcache_ascend.v1.npu_connector.is_310p", return_value=True)
def test_create_npu_connector_rejects_layerwise_on_310p(_is_310p):
    config = LMCacheEngineConfig.from_defaults()
    config.use_layerwise = True
    config.enable_blending = False

    with pytest.raises(ValueError, match="310P"):
        CreateNPUConnector(config, _fake_metadata(), EngineType.VLLM)


@patch("lmcache_ascend.v1.npu_connector.is_310p", return_value=True)
def test_create_npu_connector_rejects_blending_on_310p(_is_310p):
    config = LMCacheEngineConfig.from_defaults()
    config.use_layerwise = True
    config.enable_blending = True

    with pytest.raises(ValueError, match="310P"):
        CreateNPUConnector(config, _fake_metadata(), EngineType.VLLM)


@patch("lmcache_ascend.v1.npu_connector.is_310p", return_value=True)
def test_create_npu_connector_rejects_sglang_layerwise_on_310p(_is_310p):
    config = LMCacheEngineConfig.from_defaults()
    config.use_layerwise = True

    with pytest.raises(ValueError, match="310P"):
        CreateNPUConnector(config, _fake_metadata(), EngineType.SGLANG)
