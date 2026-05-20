/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * BatchChannel: a thin, RoCE-backed point-to-point batch transfer primitive built on
 * top of the internal hccl::Transport (TRANS_TYPE_IBV_EXP) layer.
 *
 * This is a refactor of the standalone test driver in test/hcomm_test/batch_channel.h
 * suited for the LMCache process-aware bring-up: the in-process Rendezvous handshake
 * is gone. The owning PingPongAgent establishes the per-peer ctrl socket via the
 * device-shared listen socket and hands it to BatchChannel::Init together with the
 * already-opened NIC ctx, dispatcher, and shared staging regions.
 *
 * Per-peer state still owned by BatchChannel: Transport, NotifyPool (per-channel for
 * v1 to keep the working notify-protocol identical to the test driver), the InputMem
 * / OutputMem wrappers that point into the agent's shared regions.
 *
 * Both ends MUST call Send/Recv with matching (ptrs, sizes, count). No wire-level
 * manifest is exchanged.
 */

#ifndef HCCL_PINGPONG_BATCH_CHANNEL_H
#define HCCL_PINGPONG_BATCH_CHANNEL_H

#include <cstdint>
#include <memory>
#include <string>

#include "acl/acl.h"
#include "hccl/hccl_types.h"

#include "hccl_ip_address.h"
#include "hccl_socket.h"
#include "dispatcher.h"
#include "hccl_network_pub.h"

// Scatter entry descriptor shared between sender and receiver. Sender populates
// ddrBuf (host-resident contiguous source, layout = concatenation of the per-
// destination byte slices in order). Receiver populates dstBufs (numEl device
// pointers). v1 keeps ScatterSend H2D-only to avoid the cost of having ACL infer
// memcpy direction per call; a follow-up extension will add a D2D variant.
struct P2pScatterEntry {
    void *ddrBuf;           // sender-side: host DDR pointer. Receiver-side: may be NULL.
    void **dstBufs;         // receiver-side: numEl device pointers. Sender-side: may be NULL.
    uint64_t *counts;       // element counts per destination slice.
    HcclDataType dataType;  // element type (bytes/elem resolved via SalGetDataTypeSize).
    uint32_t numEl;         // length of dstBufs[] / counts[].
};

// Externally owned shared staging regions. The agent allocates these once per
// device and hands the same pair to every per-peer BatchChannel.
struct BatchChannelSharedMemory {
    void *inputBase{nullptr};    // Local receiver-landing region (registered as INPUT_MEM).
    uint64_t inputBytes{0};
    void *outputBase{nullptr};   // Local sender-staging region (registered as OUTPUT_MEM).
    uint64_t outputBytes{0};
};

// Resources owned by the agent (or otherwise externally) that BatchChannel uses to
// run its data plane. Lifetime: must outlive the BatchChannel that consumes them.
struct BatchChannelSharedResources {
    HcclNetDevCtx nicNetDevCtx{nullptr};                  // shared per-device NIC ctx
    HcclDispatcher dispatcher{nullptr};                   // shared per-device dispatcher
    std::shared_ptr<hccl::HcclSocket> ctrlSocket;         // pre-established per-peer ctrl socket
    hccl::HcclIpAddress localIp;                          // local NIC IPv4
    hccl::HcclIpAddress remoteIp;                         // peer NIC IPv4
    uint32_t phyId{0};                                    // local physical device id
    std::string tag{"hcomm_batch_channel"};               // unique tag (notifyPool op + MachinePara::tag); lifetime owned by the channel
};

struct BatchChannelConfig {
    // Size of a single staging chunk, in bytes. Each packed/split transfer slot is
    // page-aligned inside a chunk and the chunk itself is the RDMA write granularity.
    uint64_t chunkSizeBytes = 2ULL * 1024 * 1024;

    // Number of chunks per ping-pong buffer. The sender fires one Post per chunk so
    // it can pipeline multiple RDMA writes inside the same buffer epoch.
    uint32_t nChunksPerBuff = 4;

    // Number of ping-pong buffers. The shared-region design requires exactly 2.
    uint32_t nBuffs = 2;

    // If true, Send-family calls enqueue a Wait(recvDone) after all chunks. Then
    // synchronizing the sender stream means the remote receiver has drained the
    // batch.
    bool waitRecvDone = true;

    // RoCE traffic class / service level (passed straight to MachinePara).
    uint32_t tc = 132;
    uint32_t sl = 4;
};

class BatchChannel {
public:
    BatchChannel();
    ~BatchChannel();

    BatchChannel(const BatchChannel &) = delete;
    BatchChannel &operator=(const BatchChannel &) = delete;

    /**
     * Initialize the channel using a pre-established control socket and shared
     * device-level resources (NIC ctx + dispatcher) plus the shared staging regions.
     *
     * @param localDevId   Logical device id this end will use.
     * @param peerDevId    Peer's logical device id (only used to populate MachinePara).
     * @param isListener   true if this end is the listener (server) of the per-peer
     *                     ctrl socket, false if it is the client. Independent of the
     *                     eventual sender/receiver direction.
     * @param res          Shared resources (NIC ctx, dispatcher, ctrl socket, IPs).
     *                     Must outlive this BatchChannel.
     * @param sharedMem    Externally owned shared input/output regions.
     * @param cfg          Channel tuning. Defaults give a 2 MiB chunk x 4-per-buf x
     *                     2 buffers = 16 MiB shared region per direction.
     */
    HcclResult Init(uint32_t localDevId, uint32_t peerDevId,
                    bool isListener,
                    const BatchChannelSharedResources &res,
                    const BatchChannelSharedMemory &sharedMem,
                    const BatchChannelConfig &cfg = {});

    /**
     * Send a scatter/gather list of device buffers to the peer. Peer MUST call Recv
     * with the matching (sizes, count) on its own local device buffers.
     */
    HcclResult Send(void **srcPtrs, const uint64_t *sizes, uint32_t count,
                    aclrtStream stream);

    /**
     * Receive a scatter/gather list of device buffers from the peer.
     *
     * Init() primes recvReady[b] for every b, which tells the sender all buffers
     * are initially empty and safe to fill. Recv() returns buffer credits after
     * each full buffer epoch and once more for a trailing partial epoch.
     */
    HcclResult Recv(void **dstPtrs, const uint64_t *sizes, uint32_t count,
                    aclrtStream stream);

    /**
     * Scatter-send a batch of host DDR entries to the peer. Sender H2D-copies the
     * entry's ddrBuf into the ping-pong staging region, sliced by counts[k] *
     * sizeof(dataType), RDMA-writes the slice across to the receiver, and Posts a
     * chunk-done notify. Greedy packing only within a single entry.
     *
     * NOTE (v1): hardcoded ACL_MEMCPY_HOST_TO_DEVICE; passing a device pointer in
     * ddrBuf will silently fault. Tracked TODO: scatter_send_d2d_followup.
     */
    HcclResult ScatterSend(P2pScatterEntry *entries, uint32_t batchSize,
                           aclrtStream stream);

    /**
     * Scatter-recv: drains the ping-pong ring into per-destination device buffers.
     * Mirror of ScatterSend.
     */
    HcclResult ScatterRecv(P2pScatterEntry *entries, uint32_t batchSize,
                           aclrtStream stream);

    /**
     * Tear the channel down. Idempotent. The caller owns device lifetime, the
     * shared NIC ctx / dispatcher / ctrl socket, and the shared memory passed to
     * Init().
     */
    HcclResult Finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // HCCL_PINGPONG_BATCH_CHANNEL_H
