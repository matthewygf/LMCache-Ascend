# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch
import torch_npu  # noqa: F401

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.server import MPCacheEngine, parse_args, run_cache_server
from lmcache_ascend.v1.multiprocess.npu_context import NPUCacheContext

logger = init_logger(__name__)


def register_npu_kv_cache(
    self,
    instance_id: int,
    kv_caches: KVCache,
    model_name: str,
    world_size: int,
    layout_hints: LayoutHints,
) -> None:
    """
    Registers the KV cache tensors for a given NPU instance ID.

    Args:
        instance_id: The NPU instance ID (typically the worker PID).
        kv_caches: KV cache tensor wrappers from vLLM (list of AscendIPCWrapper).
        model_name: Name of the model associated with this KV cache.
        world_size: World size for this KV cache (used for MLA lock accounting).
        layout_hints: KV layout hints forwarded to NPUCacheContext.
    """
    npu_context = NPUCacheContext(
        kv_caches,
        self.chunk_size,
        layout_hints=layout_hints or None,
    )
    self.gpu_contexts[instance_id] = npu_context
    self.gpu_context_meta[instance_id] = (model_name, world_size)
    logger.info(
        "Registered KV cache for NPU ID %d with %d layers",
        instance_id,
        npu_context.num_layers,
    )


def unregister_npu_kv_cache(self, instance_id: int) -> None:
    """
    Unregisters the KV cache tensors for a given NPU instance ID.

    Replaces upstream MPCacheEngine.unregister_kv_cache to call
    NPUCacheContext.close() (which unsubscribes ACL report streams) and
    torch.npu.empty_cache() instead of the CUDA equivalents.

    Args:
        instance_id: The NPU instance ID (typically the worker PID).
    """
    if instance_id in self.gpu_contexts:
        npu_context = self.gpu_contexts.pop(instance_id)
        del self.gpu_context_meta[instance_id]
        npu_context.close()
        torch.npu.empty_cache()
        logger.info("Unregistered KV cache for NPU ID %d", instance_id)
    else:
        logger.warning(
            "No KV cache found for NPU ID %d to unregister", instance_id
        )


MPCacheEngine.register_kv_cache = register_npu_kv_cache
MPCacheEngine.unregister_kv_cache = unregister_npu_kv_cache

if __name__ == "__main__":
    args = parse_args()
    run_cache_server(
        host=args.host,
        port=args.port,
        chunk_size=args.chunk_size,
        cpu_buffer_size=args.cpu_buffer_size,
        max_workers=args.max_workers,
    )
