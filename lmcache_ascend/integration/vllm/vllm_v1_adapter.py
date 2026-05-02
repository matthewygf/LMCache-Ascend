# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_pp_group
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent)
        self.store_async = self.config.store_async
        self._wait_for_save_done = True
        self._finished_req_ids_waiting_for_save: set[str] = set()
        self._late_finished_sending: set[str] = set()
        logger.debug("store_async: %s", self.store_async)

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self._wait_for_save_done = False
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self.layerwise_retrievers = []

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            # lmcache-ascend start ---------------------
            slot_mapping_cpu = request.slot_mapping
            if self.store_async:
                slot_mapping_cpu = slot_mapping_cpu.pin_memory()
                with torch.npu.stream(self.lmcache_engine.gpu_connector.load_stream):
                    slot_mapping_npu = slot_mapping_cpu.to(
                        device="npu", dtype=torch.long, non_blocking=True
                    )
            else:
                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping_npu = slot_mapping_cpu.to(self.device)
            # lmcache-ascend end ---------------------
            assert len(tokens) == len(slot_mapping_cpu)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping_npu[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping_npu[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping_cpu=slot_mapping_cpu[:lmcache_cached_tokens],
                    slot_mapping_npu=slot_mapping_npu[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping_cpu[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            self._wait_for_save_done = True
            return

        if self.use_layerwise:
            assert not self.store_async, (
                "Layerwise storing is not supported with async store"
            )
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            self._replay_finished_stores_after_save()
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # lmcache-ascend start ---------------------
        ordering_event = torch.npu.Event()
        ordering_event.record()
        # lmcache-ascend end ---------------------

        for request in connector_metadata.requests:
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # lmcache-ascend start ---------------------
            slot_mapping = slot_mapping.pin_memory()
            with torch.npu.stream(self.lmcache_engine.gpu_connector.store_stream):
                slot_mapping_npu = slot_mapping.to(
                    device="npu", dtype=torch.long, non_blocking=True
                )
            # lmcache-ascend end ---------------------

            skip_leading_tokens = save_spec.skip_leading_tokens

            if skip_leading_tokens == len(token_ids):
                continue
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
                ordering_event=ordering_event,
                slot_mapping_npu=slot_mapping_npu,
            )

            if get_pp_group().is_last_rank:
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

        self._wait_for_save_done = True
        self._replay_finished_stores_after_save()

    def _may_register_store_after_wait_for_save(self, request: "Request") -> bool:
        if self.kv_role == "kv_consumer":
            return False
        save_spec = request.save_spec
        if save_spec is None:
            return False
        if not save_spec.can_save and self.kv_role != "kv_producer":
            return False
        return save_spec.skip_leading_tokens != len(request.token_ids)

    def _replay_finished_stores_after_save(self) -> None:
        if not self._finished_req_ids_waiting_for_save or self.lmcache_engine is None:
            return

        finished_sending = self.lmcache_engine.get_finished_stores(
            self._finished_req_ids_waiting_for_save
        )
        if finished_sending:
            self._late_finished_sending |= finished_sending
        self._finished_req_ids_waiting_for_save = set()

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if self.lmcache_engine is None:
            return None, None
        query_req_ids = set(finished_req_ids)
        if not self._wait_for_save_done:
            # NOTE (gingfung): The is a workaround logic for the case
            # where the requests is deferred (i.e. spec_decode or MTP)
            # and the model_runner call get_finished before wait_for_save.
            connector_metadata = self._parent._get_connector_metadata()
            assert isinstance(connector_metadata, LMCacheConnectorMetadata)

            waiting_for_save = {
                request.req_id
                for request in connector_metadata.requests
                if request.req_id in finished_req_ids
                and self._may_register_store_after_wait_for_save(request)
            }
            if waiting_for_save:
                self._finished_req_ids_waiting_for_save |= waiting_for_save
                query_req_ids -= waiting_for_save

        finished_sending = self.lmcache_engine.get_finished_stores(query_req_ids)
        if self._late_finished_sending:
            finished_sending |= self._late_finished_sending
            self._late_finished_sending = set()
        return (
            finished_sending if finished_sending else None,
            None,
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if (
            not self.store_async
            or self.kv_role == "kv_consumer"
            or self.lmcache_engine is None
        ):
            return

        logger.debug(
            "LMCache-Ascend handling preemptions: req_ids=%s",
            sorted(preempted_req_ids),
        )
        waited_req_ids = self.lmcache_engine.wait_for_pending_stores(preempted_req_ids)
        if waited_req_ids:
            logger.info(
                "Handled preemptions after draining async stores: req_ids=%s",
                sorted(waited_req_ids),
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        _, return_params = super().request_finished(request, block_ids)
        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
