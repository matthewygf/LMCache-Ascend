# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import LayoutHints, need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata
import torch

# First Party
from lmcache_ascend import _build_info
from lmcache_ascend.v1.npu_connector.npu_connectors import is_310p

if _build_info.__framework_name__ == "pytorch":
    # First Party
    from lmcache_ascend.v1.npu_connector.npu_connectors import (
        VLLMBufferLayerwiseNPUConnector,
        VLLMPagedMemLayerwiseNPUConnector,
        VLLMPagedMemNPUConnectorV2,
    )
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    from lmcache_ascend.mindspore.v1.npu_connector import (
        VLLMBufferLayerwiseNPUConnector,
        VLLMPagedMemLayerwiseNPUConnector,
        VLLMPagedMemNPUConnectorV2,
    )

logger = init_logger(__name__)


def CreateNPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
    layout_hints: Optional[LayoutHints] = None,
) -> GPUConnectorInterface:
    """Factory function to create NPU connectors on Ascend.

    Replaces upstream CreateGPUConnector to return Ascend NPU-specific
    connector implementations.
    """
    use_gpu = need_gpu_interm_buffer(config)

    # 310P vLLM KV caches use NZ (fractal) layout. Layerwise / CacheBlend
    # connectors call single_layer_kv_transfer, which derives
    # block_size/num_heads/head_dims from ND trailing dims and copies
    # contiguous head*dim spans. On NZ that mis-addresses slots and
    # over-copies (OOB / silent wrong KV). Multi-layer 310P has a
    # dedicated NZ kernel; layerwise does not yet. Fail closed before
    # touching NPU runtime.
    if config.use_layerwise and is_310p():
        raise ValueError(
            "use_layerwise / enable_blending is not supported on "
            "Ascend 310P: NZ KV layout is incompatible with "
            "single_layer_kv_transfer. Disable use_layerwise (and "
            "blending), or use a 910B device."
        )

    num_gpus = torch.npu.device_count()
    local_rank = metadata.worker_id % num_gpus
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    if engine == EngineType.VLLM:
        if metadata.use_mla and config.use_layerwise and config.enable_blending:
            raise ValueError(
                "We haven't supported MLA with Cacheblend yet. Please disable blending."
            )

        if config.use_layerwise:
            if config.enable_blending:
                conn = VLLMBufferLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            else:
                conn = VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            return conn

        if config.use_gpu_connector_v3:
            raise NotImplementedError(
                "GPU Connector v3 is not supported yet. Please contact LMCache-Ascend."
            )
        else:
            conn = VLLMPagedMemNPUConnectorV2.from_metadata(
                metadata, use_gpu, device, layout_hints=layout_hints
            )
            return conn
    elif engine == EngineType.SGLANG:
        # First Party
        from lmcache_ascend.v1.npu_connector.npu_connectors import (
            SGLangLayerwiseNPUConnector,
            SGLangNPUConnector,
        )

        num_layer, _, chunk_size, num_kv_head, head_dim = metadata.kv_shape
        hidden_dim_size = num_kv_head * head_dim
        kv_dtype = metadata.kv_dtype

        if config.use_layerwise:
            conn = SGLangLayerwiseNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        else:
            conn = SGLangNPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
            )
        return conn
    else:
        raise RuntimeError(f"Unsupported engine type for Ascend: {engine}")
