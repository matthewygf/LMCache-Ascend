# SPDX-License-Identifier: Apache-2.0
"""Regression: delay-pull cleanup must drain HCCL transport before Done."""

# Standard
from unittest.mock import MagicMock, Mock, patch

# Third Party
from lmcache.v1.memory_management import MemoryFormat
import torch

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemNPUConnectorV2,
)
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.transfer_context import LeaseExpiredError


def _make_real_proxy(ctx, channel, chunk_index: int = 0) -> ProxyMemoryObj:
    return ProxyMemoryObj(
        backing_obj=None,
        transfer_channel=channel,
        target_peer_url="peer",
        remote_buffer_uuid=f"uuid-{chunk_index}",
        remote_mem_index=chunk_index,
        transfer_context=ctx,
        chunk_index=chunk_index,
        shapes=[torch.Size([2, 2, 16, 16])],
        dtypes=[torch.bfloat16],
        fmt=MemoryFormat.KV_2LTD,
    )


class TestRemoteBatchedToGpuCleanup:
    def test_mid_pipeline_lease_expiry_drains_transport_before_release(self):
        """Lease expiry after batch-0 submit must not UAF in-flight DMA pages.

        Trigger: use_host_staging + delay_pull, >=2 micro-batches; batch 0's
        ``submit_batched_read`` enqueues HCCL DMA, then batch 1's
        ``check_lease`` raises. Cleanup used to only sync ``load_stream``,
        then ``release_buffers`` + ``send_done_now`` while transport DMA
        could still be writing the ping-pong pool / producer arena.
        """
        connector = object.__new__(VLLMPagedMemNPUConnectorV2)
        call_order: list[str] = []

        transport_stream = MagicMock()
        transport_stream.synchronize.side_effect = lambda: call_order.append(
            "transport_sync"
        )

        load_stream = MagicMock()
        load_stream.synchronize.side_effect = lambda: call_order.append("load_sync")
        load_stream.wait_event = Mock()
        connector.load_stream = load_stream
        connector._record_failed_load = Mock()
        connector.to_gpu = Mock()

        channel = MagicMock()
        channel.transport_stream = transport_stream
        channel.submit_batched_read = Mock(return_value=MagicMock())

        ctx = MagicMock()
        ctx.max_pipeline_depth = 1
        pool_a = [MagicMock()]
        pool_b = [MagicMock()]
        ctx.allocate_buffers = Mock(side_effect=[pool_a, pool_b])

        def release(buffers):
            call_order.append(f"release:{id(buffers)}")

        ctx.release_buffers = Mock(side_effect=release)
        ctx.send_done_now = Mock(side_effect=lambda: call_order.append("done"))

        real_proxies = [_make_real_proxy(ctx, channel, i) for i in range(2)]

        submit_calls = {"n": 0}

        def fake_submit(proxies_arg):
            submit_calls["n"] += 1
            call_order.append(f"submit:{submit_calls['n']}")
            if submit_calls["n"] == 1:
                return MagicMock()
            raise LeaseExpiredError("lease expired mid-pipeline")

        with (
            patch.object(
                ProxyMemoryObj, "submit_resolve_batch", side_effect=fake_submit
            ),
            patch("torch.npu.Event", return_value=MagicMock()),
        ):
            connector._remote_batched_to_gpu(
                real_proxies,
                [0, 16],
                [16, 32],
                req_id="req-uaf",
            )

        assert submit_calls["n"] == 2
        assert "transport_sync" in call_order
        assert "load_sync" in call_order
        assert "done" in call_order
        transport_idx = call_order.index("transport_sync")
        release_idxs = [
            i for i, name in enumerate(call_order) if name.startswith("release:")
        ]
        done_idx = call_order.index("done")
        assert release_idxs, "ping-pong pools must be released"
        assert transport_idx < min(release_idxs), (
            "transport DMA must finish before release_buffers"
        )
        assert transport_idx < done_idx, (
            "transport DMA must finish before send_done_now"
        )
        ctx.release_buffers.assert_any_call(pool_a)
        ctx.release_buffers.assert_any_call(pool_b)
        ctx.send_done_now.assert_called_once()
        connector._record_failed_load.assert_called_once_with("req-uaf")
