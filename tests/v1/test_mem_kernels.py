# SPDX-License-Identifier: Apache-2.0
# Standard
import random

# Third Party
from lmcache.v1.memory_management import PinMemoryAllocator
from lmcache_tests.v1.utils import check_mem_obj_equal
import pytest
import torch

# First Party
import lmcache_ascend.c_ops as lmc_ops

# Local
from .utils import (
    check_paged_kv_cache_equal,
    generate_dsa_kv_cache,
    generate_kv_cache_paged_list_tensors,
    generate_kv_cache_paged_list_tuple_tensors,
    generate_mla_kv_cache,
)


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("head_size", [256])
def test_multi_layer_kernel_kvcache_merged_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    device = "npu"

    num_blocks = 256
    dtype = torch.bfloat16
    hidden_dim_size = num_heads * head_size
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # layer by layer extract
    memory_obj_old_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_old = mem_allocator.allocate(mem_obj_shape, dtype)
        for layer_id in range(num_layers):
            lmc_ops.load_and_reshape_flash(
                memory_obj_old.tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        memory_obj_old_list.append(memory_obj_old)
    end_event.record()
    # wait for all the operations to finish
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Old extract time: ", elapsed_time_ms / 1000)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()

    # on ascend kv_cache_pointers need to be on device
    kv_cache_pointers = kv_cache_pointers.npu()

    memory_obj_new_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_new_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("New extract time: ", elapsed_time_ms / 1000)

    check_mem_obj_equal(
        memory_obj_old_list,
        memory_obj_new_list,
    )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()

    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_new_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            False,
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("head_size", [128])
def test_multi_layer_kernel_kvcache_separate_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    device = "npu"

    num_blocks = 1000
    dtype = torch.bfloat16
    hidden_dim_size = num_heads * head_size
    kv_cache = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # New extract with multi layer kernel
    kv_cache_pointers = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    # for i in range(num_layers):
    #     kv_cache_pointers[i] = kv_cache[i].data_ptr()

    for i in range(num_layers):
        kv_cache_pointers[i * 2 + 0] = kv_cache[i][0].data_ptr()  # Key pointer
        kv_cache_pointers[i * 2 + 1] = kv_cache[i][1].data_ptr()  # Value pointer

    # on ascend kv_cache_pointers need to be on device
    kv_cache_pointers = kv_cache_pointers.npu()

    memory_obj_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = torch.Size(
            [2, num_layers, len(slot_mapping_temp), num_heads * head_size]
        )

        memory_obj_new = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers,
            slot_mapping_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,
            False,
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_list.append(memory_obj_new)

    end_event.record()
    # wait for all the operations to finish
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"New extract time: {elapsed_time_ms / 1000:.4f}s")

    # check_mem_obj_equal(
    #     memory_obj_old_list,
    #     memory_obj_new_list,
    # )

    # Generate new paged kv_cache
    kv_cache_new = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    kv_cache_pointers_new = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    for i in range(num_layers):
        kv_cache_pointers_new[i * 2 + 0] = kv_cache_new[i][0].data_ptr()
        kv_cache_pointers_new[i * 2 + 1] = kv_cache_new[i][1].data_ptr()

    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        memory_obj_new = memory_obj_list[chunk_id]
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_new.tensor,
            kv_cache_pointers_new,
            slot_mapping_temp,
            kv_cache_new[0][0].device,
            page_buffer_size,
            False,  # to gpu
            False,
            2,
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache, kv_cache_new, slot_mapping, num_heads=num_heads, head_size=head_size
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("head_size", [256])
def test_fused_multi_layer_kvcache_merged_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    """
    - from_gpu: Verify that the output of the fused method is
      consistent with the baseline method.
    - to_gpu: Verify that the KV cache content is correct
      after data is written back.
    """
    device = "npu"
    num_blocks = 256
    dtype = torch.bfloat16
    kvs = 2
    hidden_dim_size = num_heads * head_size

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # kv_cache_pointers
    kv_cache_pointers = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers[i] = kv_cache[i].data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    # Staging buffer for fused method
    staging_buffer = torch.empty(
        [kvs, num_layers, chunk_size, hidden_dim_size], dtype=dtype, device=device
    )

    # from_gpu: Baseline method for data extraction
    memory_obj_baseline_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])

        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_baseline_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nBaseline from_gpu time: {elapsed_time_ms:.6f} ms")

    # from_gpu: Fused method for data extraction
    memory_obj_fused_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        staging_cache = staging_buffer[:, :, : len(slot_temp), :]
        lmc_ops.fused_multi_layer_kv_transfer(
            memory_obj.tensor,
            staging_cache,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_fused_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Fused from_gpu time: {elapsed_time_ms:.6f} ms")

    check_mem_obj_equal(memory_obj_baseline_list, memory_obj_fused_list)

    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )

    kv_cache_pointers_new = torch.empty(
        num_layers, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        kv_cache_pointers_new[i] = kv_cache_new[i].data_ptr()
    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_fused_list[chunk_id].tensor,
            kv_cache_pointers_new,
            slot_temp,
            kv_cache_new[0].device,
            page_buffer_size,
            False,  # to_gpu
            False,
            1,  # MERGED_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("chunk_size", [256, 512])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("head_size", [128])
def test_fused_multi_layer_kvcache_separate_fmt(
    num_tokens, num_heads, chunk_size, num_layers, block_size, head_size
):
    """
    - from_gpu: Verify that the output of the fused method is
      consistent with the baseline method.
    - to_gpu: Verify that the KV cache content is correct
      after data is written back.
    """
    device = "npu"
    num_blocks = 1000
    dtype = torch.bfloat16
    kvs = 2
    hidden_dim_size = num_heads * head_size

    kv_cache = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024  # 4GB
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    kv_cache_pointers = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    for i in range(num_layers):
        kv_cache_pointers[i * 2 + 0] = kv_cache[i][0].data_ptr()  # Key pointer
        kv_cache_pointers[i * 2 + 1] = kv_cache[i][1].data_ptr()  # Value pointer
    kv_cache_pointers = kv_cache_pointers.npu()

    # Staging buffer for fused method
    staging_buffer = torch.empty(
        [kvs, num_layers, chunk_size, hidden_dim_size], dtype=dtype, device=device
    )

    # from_gpu: Baseline method for data extraction
    memory_obj_baseline_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,  # use_mla
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_baseline_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nBaseline from_gpu time: {elapsed_time_ms:.6f} ms")

    # from_gpu: Fused method for data extraction
    memory_obj_fused_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([kvs, num_layers, len(slot_temp), hidden_dim_size])

        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        staging_cache = staging_buffer[:, :, : len(slot_temp), :]
        lmc_ops.fused_multi_layer_kv_transfer(
            memory_obj.tensor,
            staging_cache,
            kv_cache_pointers,
            slot_temp,
            kv_cache[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,  # use_mla
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_fused_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Fused from_gpu time: {elapsed_time_ms:.6f} ms")

    check_mem_obj_equal(memory_obj_baseline_list, memory_obj_fused_list)

    kv_cache_new = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    kv_cache_pointers_new = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )

    for i in range(num_layers):
        kv_cache_pointers_new[i * 2 + 0] = kv_cache_new[i][0].data_ptr()
        kv_cache_pointers_new[i * 2 + 1] = kv_cache_new[i][1].data_ptr()
    kv_cache_pointers_new = kv_cache_pointers_new.npu()

    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_fused_list[chunk_id].tensor,
            kv_cache_pointers_new,
            slot_temp,
            kv_cache_new[0][0].device,
            page_buffer_size,
            False,  # to_gpu
            False,  # use_mla
            2,  # SEPARATE_KV
            hidden_dim_size,  # k_hidden_dims
            hidden_dim_size,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
    )

    mem_allocator.close()


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8, 1])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize("vllm_two_major", [True, False])
def test_single_layer_kernel(
    num_tokens,
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
    vllm_two_major,
):
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16
    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )
    kv_cache_new = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, kvs, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (kvs, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    for layer_id in range(num_layers):
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache[layer_id],
            slot_mapping,
            True,
            1,  # MERGED KV
            token_major,
            vllm_two_major,
        )
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_new[layer_id],
            slot_mapping,
            False,
            1,  # MERGED KV
            token_major,
            vllm_two_major,
        )
    torch.npu.synchronize()
    check_paged_kv_cache_equal(
        kv_cache,
        kv_cache_new,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 2048])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8, 1])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
def test_single_layer_kernel_separate_kv(
    num_tokens,
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
):
    """
    Test single_layer_kv_transfer with SEPARATE_KV format
    - from_gpu: GPU KV cache -> staging buffer
    - to_gpu: staging buffer -> GPU KV cache
    - Verify: src == dst
    """
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    # Generate SEPARATE_KV format caches: List[Tuple[Tensor, Tensor]]
    kv_cache_src = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
    )

    kv_cache_dst = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks,
        device,
        num_layers,
        num_heads,
        head_size,
        block_size,
        dtype,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)

    # Staging buffer
    if token_major:
        tmp_gpu_buffer = torch.empty(
            (num_tokens, kvs, hidden_dim_size), dtype=dtype, device=device
        )
    else:
        tmp_gpu_buffer = torch.empty(
            (kvs, num_tokens, hidden_dim_size), dtype=dtype, device=device
        )

    # Transfer test: GPU -> staging -> GPU
    for layer_id in range(num_layers):
        # Convert tuple to list for C++ interface

        # from_gpu: GPU cache -> staging buffer
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_src[layer_id],
            slot_mapping,
            True,  # from_gpu
            2,  # SEPARATE KV
            token_major,
            False,  # vllm_two_major (not used for SEPARATE KV)
        )

        # to_gpu: staging buffer -> GPU cache
        lmc_ops.single_layer_kv_transfer(
            tmp_gpu_buffer,
            kv_cache_dst[layer_id],
            slot_mapping,
            False,  # to_gpu
            2,  # SEPARATE KV
            token_major,
            False,
        )

    torch.npu.synchronize()

    # Verify correctness
    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=False,  # Not applicable for SEPARATE KV
        kv_format=2,  # SEPARATE KV
    )


# TODO: MLA is not supported for layerwise yet
@pytest.mark.parametrize("num_tokens", [256, 500, 1024, 8000])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_chunks", [1, 3, 5])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
@pytest.mark.parametrize("vllm_two_major", [True, False])
def test_batched_fused_single_layer_kernel(
    num_tokens,
    num_layers,
    num_chunks,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
    vllm_two_major,
):
    """
    - Baseline: 1 kernel call + N torch.copy_ calls
    - Batched fusion: batched_fused_single_layer_kv_transfer
    """
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    # Compute chunk partitioning
    base_chunk_size = num_tokens // num_chunks
    remainder = num_tokens % num_chunks

    chunk_sizes = []
    chunk_offsets = []
    current_offset = 0

    for i in range(num_chunks):
        size = base_chunk_size + (1 if i < remainder else 0)
        chunk_sizes.append(size)
        chunk_offsets.append(current_offset)
        current_offset += size

    kv_cache_src = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )

    kv_cache_dst_baseline = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )

    kv_cache_dst_fused = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        vllm_two_major=vllm_two_major,
    )

    slot_mapping_list = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping_full = torch.tensor(slot_mapping_list, device=device)

    # Buffer shapes
    def get_buffer_shape(n_tokens):
        if token_major:
            return (n_tokens, kvs, hidden_dim_size)
        else:
            return (kvs, n_tokens, hidden_dim_size)

    full_buffer_shape = get_buffer_shape(num_tokens)
    chunk_buffer_shapes = [get_buffer_shape(chunk_sizes[i]) for i in range(num_chunks)]

    # Staging buffers
    staging_cache_baseline = torch.empty(full_buffer_shape, dtype=dtype, device=device)
    staging_cache_fused = torch.empty(full_buffer_shape, dtype=dtype, device=device)

    total_cpu_bytes = 0
    for chunk_size in chunk_sizes:
        chunk_bytes = chunk_size * kvs * hidden_dim_size * torch.finfo(dtype).bits // 8
        total_cpu_bytes += chunk_bytes * num_layers
    total_cpu_bytes *= 2
    total_cpu_bytes = int(total_cpu_bytes * 1.2)

    mem_allocator = PinMemoryAllocator(total_cpu_bytes)

    try:

        def create_chunked_cpu_buffers():
            cpu_memory_objs = []
            cpu_tensors = []
            for layer_id in range(num_layers):
                layer_memory_objs = []
                layer_tensors = []
                for chunk_id in range(num_chunks):
                    chunk_shape = torch.Size(chunk_buffer_shapes[chunk_id])
                    memory_obj = mem_allocator.allocate(chunk_shape, dtype)
                    layer_memory_objs.append(memory_obj)
                    layer_tensors.append(memory_obj.tensor)
                cpu_memory_objs.append(layer_memory_objs)
                cpu_tensors.append(layer_tensors)
            return cpu_memory_objs, cpu_tensors

        cpu_mem_objs_baseline, cpu_baseline = create_chunked_cpu_buffers()
        cpu_mem_objs_fused, cpu_fused = create_chunked_cpu_buffers()

        # from_gpu: Baseline
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.single_layer_kv_transfer(
                staging_cache_baseline,
                kv_cache_src[layer_id],
                slot_mapping_full,
                True,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

            for chunk_id in range(num_chunks):
                offset = chunk_offsets[chunk_id]
                chunk_size = chunk_sizes[chunk_id]

                if token_major:
                    staging_chunk = staging_cache_baseline[offset : offset + chunk_size]
                else:
                    staging_chunk = staging_cache_baseline[
                        :, offset : offset + chunk_size
                    ]

                cpu_baseline[layer_id][chunk_id].copy_(staging_chunk, non_blocking=True)

        end_event.record()
        torch.npu.synchronize()
        baseline_from_time = start_event.elapsed_time(end_event)

        # from_gpu: Batched Fused
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_fused[layer_id],
                staging_cache_fused,
                kv_cache_src[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                True,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        fused_from_time = start_event.elapsed_time(end_event)

        print(
            f"\n[from_gpu] Baseline: {baseline_from_time:.3f} ms, "
            f"Fused: {fused_from_time:.3f} ms"
        )

        # Verify from_gpu correctness
        for layer_id in range(num_layers):
            for chunk_id in range(num_chunks):
                assert torch.equal(
                    cpu_baseline[layer_id][chunk_id], cpu_fused[layer_id][chunk_id]
                ), f"from_gpu mismatch at layer={layer_id}, chunk={chunk_id}"

        # to_gpu: Baseline
        start_event.record()

        for layer_id in range(num_layers):
            for chunk_id in range(num_chunks):
                offset = chunk_offsets[chunk_id]
                chunk_size = chunk_sizes[chunk_id]

                if token_major:
                    staging_chunk = staging_cache_baseline[offset : offset + chunk_size]
                else:
                    staging_chunk = staging_cache_baseline[
                        :, offset : offset + chunk_size
                    ]

                staging_chunk.copy_(cpu_baseline[layer_id][chunk_id], non_blocking=True)

            lmc_ops.single_layer_kv_transfer(
                staging_cache_baseline,
                kv_cache_dst_baseline[layer_id],
                slot_mapping_full,
                False,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        baseline_to_time = start_event.elapsed_time(end_event)

        # to_gpu: Batched Fused
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_baseline[layer_id],
                staging_cache_fused,
                kv_cache_dst_fused[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                False,
                1,  # MERGED KV
                token_major,
                vllm_two_major,
            )

        end_event.record()
        torch.npu.synchronize()
        fused_to_time = start_event.elapsed_time(end_event)

        print(
            f"[to_gpu] Baseline: {baseline_to_time:.3f} ms, "
            f"Fused: {fused_to_time:.3f} ms"
        )

        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst_baseline,
            slot_mapping_full,
            num_heads=num_heads,
            head_size=head_size,
            vllm_two_major=vllm_two_major,
        )

        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst_fused,
            slot_mapping_full,
            num_heads=num_heads,
            head_size=head_size,
            vllm_two_major=vllm_two_major,
        )

    finally:
        mem_allocator.close()
        del kv_cache_src
        del kv_cache_dst_baseline
        del kv_cache_dst_fused
        del staging_cache_baseline
        del staging_cache_fused
        del cpu_baseline
        del cpu_fused
        torch.npu.empty_cache()


@pytest.mark.parametrize("num_tokens", [256, 1024, 2048])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("num_chunks", [1, 3, 5])
@pytest.mark.parametrize("num_blocks", [1000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("token_major", [True, False])
def test_batched_fused_single_layer_kernel_separate_kv(
    num_tokens,
    num_layers,
    num_chunks,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    token_major,
):
    """
    Test for SEPARATE_KV format (K and V stored separately)
    """
    device = "npu"
    kvs = 2
    hidden_dim_size = num_heads * head_size
    dtype = torch.bfloat16

    # Compute chunk partitioning
    base_chunk_size = num_tokens // num_chunks
    remainder = num_tokens % num_chunks

    chunk_sizes = []
    chunk_offsets = []
    current_offset = 0

    for i in range(num_chunks):
        size = base_chunk_size + (1 if i < remainder else 0)
        chunk_sizes.append(size)
        chunk_offsets.append(current_offset)
        current_offset += size

    # Generate KV caches with SEPARATE_KV format
    kv_cache_src = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    kv_cache_dst = generate_kv_cache_paged_list_tuple_tensors(
        num_blocks, device, num_layers, num_heads, head_size, block_size, dtype
    )

    slot_mapping_list = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping_full = torch.tensor(slot_mapping_list, device=device)

    def get_buffer_shape(n_tokens):
        if token_major:
            return (n_tokens, kvs, hidden_dim_size)
        else:
            return (kvs, n_tokens, hidden_dim_size)

    full_buffer_shape = get_buffer_shape(num_tokens)

    staging_cache = torch.empty(full_buffer_shape, dtype=dtype, device=device)

    total_cpu_bytes = 0
    for chunk_size in chunk_sizes:
        chunk_bytes = chunk_size * kvs * hidden_dim_size * torch.finfo(dtype).bits // 8
        total_cpu_bytes += chunk_bytes * num_layers
    total_cpu_bytes = int(total_cpu_bytes * 1.2)

    mem_allocator = PinMemoryAllocator(total_cpu_bytes)

    try:
        cpu_memory_objs = []
        cpu_tensors = []
        for layer_id in range(num_layers):
            layer_memory_objs = []
            layer_tensors = []
            for chunk_id in range(num_chunks):
                chunk_shape = torch.Size(get_buffer_shape(chunk_sizes[chunk_id]))
                memory_obj = mem_allocator.allocate(chunk_shape, dtype)
                layer_memory_objs.append(memory_obj)
                layer_tensors.append(memory_obj.tensor)
            cpu_memory_objs.append(layer_memory_objs)
            cpu_tensors.append(layer_tensors)

        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_tensors[layer_id],
                staging_cache,
                kv_cache_src[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                True,  # from_gpu
                2,  # SEPARATE_KV
                token_major,
                False,
            )

        end_event.record()
        torch.npu.synchronize()
        from_gpu_time = start_event.elapsed_time(end_event)

        start_event.record()

        for layer_id in range(num_layers):
            lmc_ops.batched_fused_single_layer_kv_transfer(
                cpu_tensors[layer_id],
                staging_cache,
                kv_cache_dst[layer_id],
                slot_mapping_full,
                chunk_offsets,
                chunk_sizes,
                False,  # to_gpu
                2,  # SEPARATE_KV
                token_major,
                False,
            )

        end_event.record()
        torch.npu.synchronize()
        to_gpu_time = start_event.elapsed_time(end_event)

        print(
            f"\n[SEPARATE_KV] from_gpu: {from_gpu_time:.3f} ms, "
            f"to_gpu: {to_gpu_time:.3f} ms"
        )

        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_full,
            num_heads=num_heads,
            head_size=head_size,
            vllm_two_major=False,
            kv_format=2,  # SEPARATE_KV
        )

    finally:
        mem_allocator.close()
        del kv_cache_src
        del kv_cache_dst
        del staging_cache
        del cpu_memory_objs
        del cpu_tensors
        torch.npu.empty_cache()


def _make_acl_batch_merged_cache_pointers(kv_cache):
    kv_cache_pointers = torch.empty(
        len(kv_cache), dtype=torch.int64, device="cpu", pin_memory=True
    )
    for layer_id, cache in enumerate(kv_cache):
        kv_cache_pointers[layer_id] = cache.data_ptr()
    return kv_cache_pointers


def _assert_acl_batch_merged_cpu_buffer_matches_cache(
    cpu_tensor, kv_cache, slot_mapping_cpu, hidden_dim
):
    slot_mapping_device = slot_mapping_cpu.to(
        device=kv_cache[0].device, dtype=torch.long
    )
    for layer_id, layer_cache in enumerate(kv_cache):
        for cache_id in range(2):
            expected = (
                layer_cache[cache_id].reshape(-1, hidden_dim)[slot_mapping_device].cpu()
            )
            actual = cpu_tensor[
                cache_id, layer_id, : slot_mapping_cpu.numel(), :hidden_dim
            ]
            assert torch.equal(actual, expected), (
                f"ACL batch merged mismatch at layer={layer_id}, cache={cache_id}"
            )


def _run_acl_batch_merged_format_roundtrip(
    kv_cache_src,
    kv_cache_dst,
    slot_mapping_cpu,
    page_buffer_size,
    hidden_dim,
    num_heads,
    head_size,
    mem_allocator,
    transfer_func_name,
):
    num_layers = len(kv_cache_src)
    dtype = kv_cache_src[0].dtype
    mem_obj_shape = torch.Size([2, num_layers, slot_mapping_cpu.numel(), hidden_dim])
    memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
    transfer_func = getattr(lmc_ops, transfer_func_name)

    kv_cache_pointers_src = _make_acl_batch_merged_cache_pointers(kv_cache_src)
    transfer_func(
        memory_obj.tensor,
        kv_cache_pointers_src,
        slot_mapping_cpu,
        kv_cache_src[0].device,
        page_buffer_size,
        True,  # from_gpu
        False,  # use_mla
        1,  # MERGED_KV
        hidden_dim,
        hidden_dim,
        0,
    )
    torch.npu.synchronize()

    _assert_acl_batch_merged_cpu_buffer_matches_cache(
        memory_obj.tensor, kv_cache_src, slot_mapping_cpu, hidden_dim
    )

    kv_cache_pointers_dst = _make_acl_batch_merged_cache_pointers(kv_cache_dst)
    transfer_func(
        memory_obj.tensor,
        kv_cache_pointers_dst,
        slot_mapping_cpu,
        kv_cache_dst[0].device,
        page_buffer_size,
        False,  # to_gpu
        False,  # use_mla
        1,  # MERGED_KV
        hidden_dim,
        hidden_dim,
        0,
    )
    torch.npu.synchronize()

    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping_cpu.to(device=kv_cache_dst[0].device, dtype=torch.long),
        num_heads=num_heads,
        head_size=head_size,
    )


@pytest.mark.parametrize("num_tokens", [17, 128])
@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    "transfer_func_name",
    ["multi_layer_kv_transfer_acl_batch", "multi_layer_kv_transfer_acl_batch_sync"],
)
def test_multi_layer_kv_transfer_acl_batch_merged_format(
    num_tokens,
    num_layers,
    slot_dtype,
    transfer_func_name,
):
    device = "npu"
    num_blocks = 64
    block_size = 16
    num_heads = 1
    head_size = 128
    dtype = torch.bfloat16
    hidden_dim = num_heads * head_size
    page_buffer_size = num_blocks * block_size

    kv_cache_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )
    kv_cache_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device=device,
        block_size=block_size,
        dtype=dtype,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
    )
    slot_mapping_cpu = torch.tensor(
        random.sample(range(0, page_buffer_size), num_tokens),
        dtype=slot_dtype,
        device="cpu",
    )

    total_elements = 2 * num_layers * num_tokens * hidden_dim
    mem_allocator = PinMemoryAllocator(total_elements * dtype.itemsize * 2)
    try:
        _run_acl_batch_merged_format_roundtrip(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu,
            page_buffer_size,
            hidden_dim,
            num_heads,
            head_size,
            mem_allocator,
            transfer_func_name,
        )
    finally:
        mem_allocator.close()
        torch.npu.empty_cache()


def _make_acl_batch_tuple_cache_pointers(kv_cache):
    kv_size = len(kv_cache[0])
    kv_cache_pointers = torch.empty(
        len(kv_cache) * kv_size, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for layer_id, layer_cache in enumerate(kv_cache):
        for cache_id, cache in enumerate(layer_cache):
            kv_cache_pointers[layer_id * kv_size + cache_id] = cache.data_ptr()
    return kv_cache_pointers


def _assert_acl_batch_cpu_buffer_matches_cache(
    cpu_tensor, kv_cache, slot_mapping_cpu, hidden_dims
):
    slot_mapping_device = slot_mapping_cpu.to(
        device=kv_cache[0][0].device, dtype=torch.long
    )
    for layer_id, layer_cache in enumerate(kv_cache):
        expected_parts = []
        for cache, hidden_dim in zip(layer_cache, hidden_dims, strict=False):
            expected_parts.append(cache.reshape(-1, hidden_dim)[slot_mapping_device])
        expected = torch.cat(expected_parts, dim=-1).cpu()
        actual = cpu_tensor[
            0, layer_id, : slot_mapping_cpu.numel(), : expected.size(-1)
        ]
        assert torch.equal(actual, expected), f"ACL batch mismatch at layer={layer_id}"


def _assert_component_major_cpu_buffer_matches_cache(
    cpu_tensor, kv_cache, slot_mapping_cpu, hidden_dims
):
    slot_mapping_device = slot_mapping_cpu.to(
        device=kv_cache[0][0].device, dtype=torch.long
    )
    num_layers = len(kv_cache)
    num_tokens = slot_mapping_cpu.numel()
    flat_cpu_tensor = cpu_tensor.reshape(-1)

    component_plane_offset = 0
    for cache_id, hidden_dim in enumerate(hidden_dims):
        for layer_id, layer_cache in enumerate(kv_cache):
            expected = (
                layer_cache[cache_id].reshape(-1, hidden_dim)[slot_mapping_device].cpu()
            )
            layer_offset = component_plane_offset + layer_id * num_tokens * hidden_dim
            actual = flat_cpu_tensor[
                layer_offset : layer_offset + num_tokens * hidden_dim
            ].reshape(num_tokens, hidden_dim)
            assert torch.equal(actual, expected), (
                "Component-major ACL batch mismatch at "
                f"layer={layer_id}, cache={cache_id}"
            )
        component_plane_offset += num_layers * num_tokens * hidden_dim


def _run_acl_batch_tuple_format_roundtrip(
    kv_cache_src,
    kv_cache_dst,
    slot_mapping_cpu,
    page_buffer_size,
    hidden_dims,
    kvcache_format,
    mem_allocator,
    timing_label=None,
    timing_bytes=None,
):
    num_layers = len(kv_cache_src)
    total_hidden_dims = sum(hidden_dims)
    dtype = kv_cache_src[0][0].dtype
    mem_obj_shape = torch.Size(
        [1, num_layers, slot_mapping_cpu.numel(), total_hidden_dims]
    )
    memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)

    kv_cache_pointers_src = _make_acl_batch_tuple_cache_pointers(kv_cache_src)
    if timing_label is not None:
        from_start_event = torch.npu.Event(enable_timing=True)
        from_end_event = torch.npu.Event(enable_timing=True)
        from_start_event.record()
    lmc_ops.multi_layer_kv_transfer_acl_batch(
        memory_obj.tensor,
        kv_cache_pointers_src,
        slot_mapping_cpu,
        kv_cache_src[0][0].device,
        page_buffer_size,
        True,  # from_gpu
        True,  # use_mla
        kvcache_format,
        hidden_dims[0],
        hidden_dims[1],
        hidden_dims[2] if len(hidden_dims) == 3 else 0,
    )
    if timing_label is not None:
        from_end_event.record()
    torch.npu.synchronize()
    if timing_label is not None:
        from_gpu_time_ms = from_start_event.elapsed_time(from_end_event)
        if timing_bytes is None:
            print(f"\n[{timing_label}] ACL batch from_gpu: {from_gpu_time_ms:.6f} ms")
        else:
            from_gpu_gib_s = timing_bytes / (from_gpu_time_ms / 1000) / (1024**3)
            print(
                f"\n[{timing_label}] ACL batch from_gpu: "
                f"{from_gpu_time_ms:.6f} ms, {from_gpu_gib_s:.3f} GiB/s"
            )

    _assert_acl_batch_cpu_buffer_matches_cache(
        memory_obj.tensor, kv_cache_src, slot_mapping_cpu, hidden_dims
    )

    kv_cache_pointers_dst = _make_acl_batch_tuple_cache_pointers(kv_cache_dst)
    if timing_label is not None:
        to_start_event = torch.npu.Event(enable_timing=True)
        to_end_event = torch.npu.Event(enable_timing=True)
        to_start_event.record()
    lmc_ops.multi_layer_kv_transfer_acl_batch(
        memory_obj.tensor,
        kv_cache_pointers_dst,
        slot_mapping_cpu,
        kv_cache_dst[0][0].device,
        page_buffer_size,
        False,  # to_gpu
        True,  # use_mla
        kvcache_format,
        hidden_dims[0],
        hidden_dims[1],
        hidden_dims[2] if len(hidden_dims) == 3 else 0,
    )
    if timing_label is not None:
        to_end_event.record()
    torch.npu.synchronize()
    if timing_label is not None:
        to_gpu_time_ms = to_start_event.elapsed_time(to_end_event)
        if timing_bytes is None:
            print(f"[{timing_label}] ACL batch to_gpu: {to_gpu_time_ms:.6f} ms")
        else:
            to_gpu_gib_s = timing_bytes / (to_gpu_time_ms / 1000) / (1024**3)
            print(
                f"[{timing_label}] ACL batch to_gpu: "
                f"{to_gpu_time_ms:.6f} ms, {to_gpu_gib_s:.3f} GiB/s"
            )

    return memory_obj


def _run_acl_batch_component_major_roundtrip(
    kv_cache_src,
    kv_cache_dst,
    slot_mapping_cpu,
    page_buffer_size,
    hidden_dims,
    mem_allocator,
    timing_label,
    timing_bytes,
):
    num_layers = len(kv_cache_src)
    total_hidden_dims = sum(hidden_dims)
    dtype = kv_cache_src[0][0].dtype
    mem_obj_shape = torch.Size(
        [1, num_layers, slot_mapping_cpu.numel(), total_hidden_dims]
    )
    memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)

    kv_cache_pointers_src = _make_acl_batch_tuple_cache_pointers(kv_cache_src)
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    lmc_ops.multi_layer_kv_transfer_acl_batch_component_major(
        memory_obj.tensor,
        kv_cache_pointers_src,
        slot_mapping_cpu,
        kv_cache_src[0][0].device,
        page_buffer_size,
        True,  # from_gpu
        True,  # use_mla
        4,  # DSA_KV
        hidden_dims[0],
        hidden_dims[1],
        hidden_dims[2],
    )
    end_event.record()
    torch.npu.synchronize()
    from_gpu_time_ms = start_event.elapsed_time(end_event)
    _print_transfer_timing(
        timing_label,
        "ACL component-major",
        "from_gpu",
        from_gpu_time_ms,
        timing_bytes,
    )

    _assert_component_major_cpu_buffer_matches_cache(
        memory_obj.tensor, kv_cache_src, slot_mapping_cpu, hidden_dims
    )

    kv_cache_pointers_dst = _make_acl_batch_tuple_cache_pointers(kv_cache_dst)
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    lmc_ops.multi_layer_kv_transfer_acl_batch_component_major(
        memory_obj.tensor,
        kv_cache_pointers_dst,
        slot_mapping_cpu,
        kv_cache_dst[0][0].device,
        page_buffer_size,
        False,  # to_gpu
        True,  # use_mla
        4,  # DSA_KV
        hidden_dims[0],
        hidden_dims[1],
        hidden_dims[2],
    )
    end_event.record()
    torch.npu.synchronize()
    to_gpu_time_ms = start_event.elapsed_time(end_event)
    _print_transfer_timing(
        timing_label,
        "ACL component-major",
        "to_gpu",
        to_gpu_time_ms,
        timing_bytes,
    )

    return memory_obj


def _print_transfer_timing(
    timing_label, transfer_name, phase, elapsed_ms, timing_bytes
):
    gib_s = timing_bytes / (elapsed_ms / 1000) / (1024**3)
    print(
        f"[{timing_label}] {transfer_name} {phase}: "
        f"{elapsed_ms:.6f} ms, {gib_s:.3f} GiB/s"
    )


def _run_multi_layer_dsa_kernel_roundtrip(
    kv_cache_src,
    kv_cache_dst,
    slot_mapping_cpu,
    page_buffer_size,
    hidden_dims,
    mem_allocator,
    timing_label,
    timing_bytes,
):
    num_layers = len(kv_cache_src)
    total_hidden_dims = sum(hidden_dims)
    dtype = kv_cache_src[0][0].dtype
    mem_obj_shape = torch.Size(
        [1, num_layers, slot_mapping_cpu.numel(), total_hidden_dims]
    )
    memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
    slot_mapping_npu = slot_mapping_cpu.to(device=kv_cache_src[0][0].device)

    kv_cache_pointers_src = _make_acl_batch_tuple_cache_pointers(kv_cache_src).npu()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    lmc_ops.multi_layer_kv_transfer(
        memory_obj.tensor,
        kv_cache_pointers_src,
        slot_mapping_npu,
        kv_cache_src[0][0].device,
        page_buffer_size,
        True,  # from_gpu
        False,
        4,  # DSA_KV
        hidden_dims[0],
        hidden_dims[1],
        hidden_dims[2],
    )
    end_event.record()
    torch.npu.synchronize()
    from_gpu_time_ms = start_event.elapsed_time(end_event)
    _print_transfer_timing(
        timing_label,
        "multi_layer_kernel",
        "from_gpu",
        from_gpu_time_ms,
        timing_bytes,
    )

    kv_cache_pointers_dst = _make_acl_batch_tuple_cache_pointers(kv_cache_dst).npu()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    lmc_ops.multi_layer_kv_transfer(
        memory_obj.tensor,
        kv_cache_pointers_dst,
        slot_mapping_npu,
        kv_cache_dst[0][0].device,
        page_buffer_size,
        False,  # to_gpu
        False,
        4,  # DSA_KV
        hidden_dims[0],
        hidden_dims[1],
        hidden_dims[2],
    )
    end_event.record()
    torch.npu.synchronize()
    to_gpu_time_ms = start_event.elapsed_time(end_event)
    _print_transfer_timing(
        timing_label,
        "multi_layer_kernel",
        "to_gpu",
        to_gpu_time_ms,
        timing_bytes,
    )

    return memory_obj


@pytest.mark.parametrize("num_tokens", [17, 128])
@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
def test_multi_layer_kv_transfer_acl_batch_mla_format(
    num_tokens,
    num_layers,
    slot_dtype,
):
    device = "npu"
    num_blocks = 64
    block_size = 16
    num_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 128
    dtype = torch.bfloat16
    page_buffer_size = num_blocks * block_size

    kv_cache_src = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    kv_cache_dst = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    slot_mapping_cpu = torch.tensor(
        random.sample(range(0, page_buffer_size), num_tokens),
        dtype=slot_dtype,
        device="cpu",
    )

    total_elements = num_layers * num_tokens * (kv_lora_rank + qk_rope_head_dim)
    mem_allocator = PinMemoryAllocator(total_elements * dtype.itemsize * 2)
    try:
        _run_acl_batch_tuple_format_roundtrip(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu,
            page_buffer_size,
            [kv_lora_rank, qk_rope_head_dim],
            3,  # MLA_KV
            mem_allocator,
        )
        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu.to(device=device, dtype=torch.long),
            num_heads=num_kv_heads,
            head_size=None,
            kv_format=3,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    finally:
        mem_allocator.close()
        torch.npu.empty_cache()


@pytest.mark.parametrize("num_tokens", [17, 128])
@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("slot_pattern", ["random", "consecutive"])
def test_multi_layer_kv_transfer_acl_batch_dsa_format(
    num_tokens,
    num_layers,
    slot_dtype,
    slot_pattern,
):
    device = "npu"
    num_blocks = 64
    block_size = 16
    num_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 128
    dsa_head_dim = 128
    dtype = torch.bfloat16
    page_buffer_size = num_blocks * block_size

    kv_cache_src = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    kv_cache_dst = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    if slot_pattern == "consecutive":
        slot_mapping_cpu = torch.arange(
            3,
            3 + num_tokens,
            dtype=slot_dtype,
            device="cpu",
        )
    else:
        slot_mapping_cpu = torch.tensor(
            random.sample(range(0, page_buffer_size), num_tokens),
            dtype=slot_dtype,
            device="cpu",
        )

    total_hidden_dims = kv_lora_rank + qk_rope_head_dim + dsa_head_dim
    total_elements = num_layers * num_tokens * total_hidden_dims
    mem_allocator = PinMemoryAllocator(total_elements * dtype.itemsize * 2)
    try:
        _run_acl_batch_tuple_format_roundtrip(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu,
            page_buffer_size,
            [kv_lora_rank, qk_rope_head_dim, dsa_head_dim],
            4,  # DSA_KV
            mem_allocator,
            timing_label=(
                f"DSA {slot_pattern}, layers={num_layers}, tokens={num_tokens}, "
                f"slot_dtype={slot_dtype}"
            ),
            timing_bytes=total_elements * dtype.itemsize,
        )
        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu.to(device=device, dtype=torch.long),
            num_heads=num_kv_heads,
            head_size=None,
            kv_format=4,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dsa_head_dim=dsa_head_dim,
        )
    finally:
        mem_allocator.close()
        torch.npu.empty_cache()


@pytest.mark.parametrize("num_tokens", [128, 512, 1024, 2048, 4192])
@pytest.mark.parametrize("slot_pattern", ["random", "consecutive"])
def test_multi_layer_kv_transfer_acl_batch_dsa_scaling(
    num_tokens,
    slot_pattern,
):
    device = "npu"
    num_layers = 79
    block_size = 128
    num_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 128
    dsa_head_dim = 128
    dtype = torch.bfloat16
    slot_offset = 3
    num_blocks = (slot_offset + num_tokens + block_size - 1) // block_size
    page_buffer_size = num_blocks * block_size

    kv_cache_src = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    kv_cache_dst = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    kv_cache_component_dst = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    kv_cache_kernel_dst = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    if slot_pattern == "consecutive":
        slot_mapping_cpu = torch.arange(
            slot_offset,
            slot_offset + num_tokens,
            dtype=torch.int64,
            device="cpu",
        )
    else:
        slot_mapping_cpu = torch.tensor(
            random.sample(range(0, page_buffer_size), num_tokens),
            dtype=torch.int64,
            device="cpu",
        )

    total_hidden_dims = kv_lora_rank + qk_rope_head_dim + dsa_head_dim
    total_elements = num_layers * num_tokens * total_hidden_dims
    timing_bytes = total_elements * dtype.itemsize
    mem_allocator = PinMemoryAllocator(timing_bytes * 6)
    hidden_dims = [kv_lora_rank, qk_rope_head_dim, dsa_head_dim]
    timing_label = (
        f"DSA scaling {slot_pattern}, layers={num_layers}, "
        f"tokens={num_tokens}, payload={timing_bytes / (1024**2):.1f} MiB"
    )
    acl_memory_obj = None
    component_memory_obj = None
    kernel_memory_obj = None
    try:
        acl_memory_obj = _run_acl_batch_tuple_format_roundtrip(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu,
            page_buffer_size,
            hidden_dims,
            4,  # DSA_KV
            mem_allocator,
            timing_label=timing_label,
            timing_bytes=timing_bytes,
        )
        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_dst,
            slot_mapping_cpu.to(device=device, dtype=torch.long),
            num_heads=num_kv_heads,
            head_size=None,
            kv_format=4,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dsa_head_dim=dsa_head_dim,
        )
        component_memory_obj = _run_acl_batch_component_major_roundtrip(
            kv_cache_src,
            kv_cache_component_dst,
            slot_mapping_cpu,
            page_buffer_size,
            hidden_dims,
            mem_allocator,
            timing_label,
            timing_bytes,
        )
        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_component_dst,
            slot_mapping_cpu.to(device=device, dtype=torch.long),
            num_heads=num_kv_heads,
            head_size=None,
            kv_format=4,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dsa_head_dim=dsa_head_dim,
        )
        kernel_memory_obj = _run_multi_layer_dsa_kernel_roundtrip(
            kv_cache_src,
            kv_cache_kernel_dst,
            slot_mapping_cpu,
            page_buffer_size,
            hidden_dims,
            mem_allocator,
            timing_label,
            timing_bytes,
        )
        check_paged_kv_cache_equal(
            kv_cache_src,
            kv_cache_kernel_dst,
            slot_mapping_cpu.to(device=device, dtype=torch.long),
            num_heads=num_kv_heads,
            head_size=None,
            kv_format=4,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dsa_head_dim=dsa_head_dim,
        )
    finally:
        if acl_memory_obj is not None:
            acl_memory_obj.ref_count_down()
        if component_memory_obj is not None:
            component_memory_obj.ref_count_down()
        if kernel_memory_obj is not None:
            kernel_memory_obj.ref_count_down()
        mem_allocator.close()
        torch.npu.empty_cache()


@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [128])
def test_multi_layer_kv_transfer_mla_format(
    num_tokens,
    num_kv_heads,
    chunk_size,
    num_layers,
    block_size,
    kv_lora_rank,
    qk_rope_head_dim,
):
    """
    Test MLA (Multilayer Attention) format KV cache transfer.
    - from_gpu: Extract KV cache from paged memory to CPU buffer
    - to_gpu: Write KV cache from CPU buffer back to paged memory
    - Verify correctness
    """
    device = "npu"
    num_blocks = 1000
    dtype = torch.bfloat16

    # Generate MLA format KV cache
    kv_cache_src = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, page_buffer_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Prepare kv cache pointers
    kv_cache_pointers = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache = kv_cache_src[i]
        kv_cache_pointers[i * 2 + 0] = k_cache.data_ptr()
        kv_cache_pointers[i * 2 + 1] = v_cache.data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    # from_gpu: Extract KV cache
    memory_obj_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    total_hidden_dims = kv_lora_rank + qk_rope_head_dim
    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([1, num_layers, len(slot_temp), total_hidden_dims])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache_src[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            3,  # MLA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )
        memory_obj_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nMLA from_gpu time: {elapsed_time_ms:.6f} ms")

    # Generate destination KV cache
    kv_cache_dst = generate_mla_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_size=block_size,
        dtype=dtype,
    )

    # Prepare destination pointers
    kv_cache_pointers_dst = torch.empty(
        num_layers * 2, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache = kv_cache_dst[i]
        kv_cache_pointers_dst[i * 2 + 0] = k_cache.data_ptr()
        kv_cache_pointers_dst[i * 2 + 1] = v_cache.data_ptr()
    kv_cache_pointers_dst = kv_cache_pointers_dst.npu()

    # to_gpu: Write back
    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_list[chunk_id].tensor,
            kv_cache_pointers_dst,
            slot_temp,
            kv_cache_dst[0][0].device,
            page_buffer_size,
            False,  # to_gpu
            False,
            3,  # MLA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            0,  # dsa_hidden_dims
        )

    # Verify correctness
    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=3,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    mem_allocator.close()


@pytest.mark.parametrize("num_tokens", [256, 1024])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("chunk_size", [256])
@pytest.mark.parametrize("num_layers", [1, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [128])
@pytest.mark.parametrize("dsa_head_dim", [128])
def test_multi_layer_kv_transfer_dsa_format(
    num_tokens,
    num_kv_heads,
    chunk_size,
    num_layers,
    block_size,
    kv_lora_rank,
    qk_rope_head_dim,
    dsa_head_dim,
):
    """
    Test DSA (Deep Sparse Attention) format KV cache transfer.
    - from_gpu: Extract KV cache from paged memory to CPU buffer
    - to_gpu: Write KV cache from CPU buffer back to paged memory
    - Verify correctness
    """
    device = "npu"
    num_blocks = 1000
    dtype = torch.bfloat16

    # Generate DSA format KV cache
    kv_cache_src = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )
    page_buffer_size = num_blocks * block_size

    slot_mapping = random.sample(range(0, page_buffer_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    pinned_cpu_size = 4 * 1024 * 1024 * 1024
    mem_allocator = PinMemoryAllocator(pinned_cpu_size)

    # Prepare kv cache pointers
    kv_cache_pointers = torch.empty(
        num_layers * 3, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache = kv_cache_src[i]
        kv_cache_pointers[i * 3 + 0] = k_cache.data_ptr()
        kv_cache_pointers[i * 3 + 1] = v_cache.data_ptr()
        kv_cache_pointers[i * 3 + 2] = dsa_k_cache.data_ptr()
    kv_cache_pointers = kv_cache_pointers.npu()

    # from_gpu: Extract KV cache
    memory_obj_list = []
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()

    total_hidden_dims = kv_lora_rank + qk_rope_head_dim + dsa_head_dim
    for slot_temp in slot_mapping_chunked:
        mem_obj_shape = torch.Size([1, num_layers, len(slot_temp), total_hidden_dims])
        memory_obj = mem_allocator.allocate(mem_obj_shape, dtype)
        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_temp,
            kv_cache_src[0][0].device,
            page_buffer_size,
            True,  # from_gpu
            False,
            4,  # DSA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            dsa_head_dim,  # dsa_hidden_dims
        )
        memory_obj_list.append(memory_obj)

    end_event.record()
    torch.npu.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"\nDSA from_gpu time: {elapsed_time_ms:.6f} ms")

    # Generate destination KV cache
    kv_cache_dst = generate_dsa_kv_cache(
        num_blocks=num_blocks,
        device=device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
        block_size=block_size,
        dtype=dtype,
    )

    # Prepare destination pointers
    kv_cache_pointers_dst = torch.empty(
        num_layers * 3, dtype=torch.int64, device="cpu", pin_memory=True
    )
    for i in range(num_layers):
        k_cache, v_cache, dsa_k_cache = kv_cache_dst[i]
        kv_cache_pointers_dst[i * 3 + 0] = k_cache.data_ptr()
        kv_cache_pointers_dst[i * 3 + 1] = v_cache.data_ptr()
        kv_cache_pointers_dst[i * 3 + 2] = dsa_k_cache.data_ptr()
    kv_cache_pointers_dst = kv_cache_pointers_dst.npu()

    # to_gpu: Write back
    for chunk_id, slot_temp in enumerate(slot_mapping_chunked):
        lmc_ops.multi_layer_kv_transfer(
            memory_obj_list[chunk_id].tensor,
            kv_cache_pointers_dst,
            slot_temp,
            kv_cache_dst[0][0].device,
            page_buffer_size,
            False,  # to_gpu
            False,
            4,  # DSA_KV
            kv_lora_rank,  # k_hidden_dims
            qk_rope_head_dim,  # v_hidden_dims
            dsa_head_dim,  # dsa_hidden_dims
        )

    # Verify correctness
    check_paged_kv_cache_equal(
        kv_cache_src,
        kv_cache_dst,
        slot_mapping,
        num_heads=num_kv_heads,
        head_size=128,
        kv_format=4,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dsa_head_dim=dsa_head_dim,
    )

    mem_allocator.close()
