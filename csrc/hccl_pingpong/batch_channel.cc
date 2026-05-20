/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Production-side BatchChannel.
 *
 * Differences from test/hcomm_test/batch_channel.cc:
 *  - Init takes a BatchChannelSharedResources struct populated by PingPongAgent
 *    (NIC ctx, dispatcher, ctrlSocket already established) instead of an in-process
 *    Rendezvous, and does NOT call aclrtSetDevice / HcclNetInit / HcclNetOpenDev /
 *    HcclDispatcherInit.
 *  - Finalize() only releases BatchChannel-owned state (notifyPool, transport,
 *    DeviceMem wrappers). The agent owns the NIC ctx, dispatcher, listen socket,
 *    and ctrlSocket lifetimes, so they outlive any single channel.
 *  - NotifyPool stays per-channel (matches the proven test driver protocol).
 *
 * Behavioural parity with the test driver: same chunk/packing layout, same notify
 * slot indices, same priming protocol.
 */

#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
// HCCL / libhccl_plf use the legacy libstdc++ std::string ABI; match
// csrc/hccl/hccl_agent.cpp so NotifyPool::RegisterOp resolves at load time.
#define _GLIBCXX_USE_CXX11_ABI 0

#include "batch_channel.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hccl_common.h"
#include "notify_pool.h"
#include "stream_pub.h"
#include "mem_device_pub.h"
#include "transport_pub.h"
#include "task_logic_info_pub.h"   // UserMemType
#include "sal_pub.h"               // SalGetDataTypeSize

#define ACLCHECK(expr)                                                             \
    do {                                                                           \
        auto _r = (expr);                                                          \
        if (_r != ACL_SUCCESS) {                                                   \
            std::cerr << "ACL error " << _r << " at " << __FILE__ << ":"           \
                      << __LINE__ << " (" #expr ")" << std::endl;                  \
            return HCCL_E_INTERNAL;                                                \
        }                                                                          \
    } while (0)

#define HCCLCHECK(expr)                                                            \
    do {                                                                           \
        HcclResult _r = (expr);                                                    \
        if (_r != HCCL_SUCCESS) {                                                  \
            std::cerr << "HCCL error " << _r << " at " << __FILE__ << ":"          \
                      << __LINE__ << " (" #expr ")" << std::endl;                  \
            return _r;                                                             \
        }                                                                          \
    } while (0)

namespace {

// Chunk size must be page-aligned so the registered MR (page-granular) lines up
// with chunk boundaries. The intra-chunk packing offsets are tight (no padding)
// because both ends compute the same cumulative offsets from the same (sizes,
// count) input.
constexpr uint64_t kPageSizeBytes = 4ULL * 1024;

}  // namespace

struct BatchChannel::Impl {
    BatchChannelConfig cfg;
    BatchChannelSharedResources res;
    BatchChannelSharedMemory sharedMem{};
    uint32_t localDevId = 0;
    uint32_t peerDevId = 0;
    bool isListener = false;

    std::unique_ptr<hccl::NotifyPool> notifyPool;
    hccl::DeviceMem inputMem;
    hccl::DeviceMem outputMem;
    std::shared_ptr<hccl::Transport> transport;

    uint64_t buffSize = 0;
    uint64_t totalStaging = 0;
    uint8_t *localInputBase = nullptr;
    uint8_t *localOutputBase = nullptr;
    uintptr_t remoteInputBase = 0;
    uint32_t notifyNum = 0;

    bool initialized = false;

    // sendDone[b][c]   slots [0 .. nBuffs*nChunksPerBuff - 1]
    // recvReady[b]     slots [nBuffs*nChunksPerBuff .. +nBuffs - 1]
    // recvDone (opt.)  slot  [nBuffs*nChunksPerBuff + nBuffs]
    uint32_t sendDoneSlot(uint32_t b, uint32_t c) const
    {
        return b * cfg.nChunksPerBuff + c;
    }
    uint32_t recvReadySlot(uint32_t b) const
    {
        return cfg.nBuffs * cfg.nChunksPerBuff + b;
    }
    uint32_t recvDoneSlot() const
    {
        return cfg.nBuffs * cfg.nChunksPerBuff + cfg.nBuffs;
    }

    uint64_t chunkOffsetInMem(uint32_t b, uint32_t c) const
    {
        return static_cast<uint64_t>(b) * buffSize +
               static_cast<uint64_t>(c) * cfg.chunkSizeBytes;
    }

    HcclResult InitDataPlane();
    HcclResult PrimeRecvReady();
    HcclResult Teardown();

    using SegmentList = std::vector<std::pair<void *, uint64_t>>;

    HcclResult PushSegments(const SegmentList &segs, aclrtMemcpyKind memcpyKind,
                            hccl::Stream &hcclStream, aclrtStream aclStream);
    HcclResult PullSegments(const SegmentList &segs,
                            hccl::Stream &hcclStream, aclrtStream aclStream);
    HcclResult WaitRecvDoneIfEnabled(hccl::Stream &hcclStream);
    HcclResult PostRecvDone(hccl::Stream &hcclStream);

    HcclResult SendChunk(uint64_t chunkOffset, uint32_t bufIdx, uint32_t chunkIdx,
                         const SegmentList &packedSrcs, aclrtMemcpyKind memcpyKind,
                         hccl::Stream &hcclStream, aclrtStream aclStream,
                         bool isFirstChunkInBuffer);
    HcclResult RecvChunk(uint64_t chunkOffset, uint32_t bufIdx, uint32_t chunkIdx,
                         const SegmentList &packedDsts,
                         hccl::Stream &hcclStream, aclrtStream aclStream,
                         bool isLastChunkInBuffer);
};

BatchChannel::BatchChannel() : impl_(new Impl()) {}
BatchChannel::~BatchChannel()
{
    if (impl_ && impl_->initialized) {
        (void)Finalize();
    }
}

HcclResult BatchChannel::Impl::InitDataPlane()
{
    notifyPool = std::make_unique<hccl::NotifyPool>();
    HCCLCHECK(notifyPool->Init(static_cast<s32>(res.phyId)));
    HCCLCHECK(notifyPool->RegisterOp(res.tag.c_str()));

    buffSize = cfg.chunkSizeBytes * cfg.nChunksPerBuff;
    totalStaging = buffSize * cfg.nBuffs;

    if (sharedMem.inputBase == nullptr || sharedMem.outputBase == nullptr ||
        sharedMem.inputBytes < totalStaging || sharedMem.outputBytes < totalStaging) {
        std::cerr << "BatchChannel: shared input/output regions must be non-null and at least "
                  << totalStaging << " bytes each (input=" << sharedMem.inputBytes
                  << ", output=" << sharedMem.outputBytes << ")" << std::endl;
        return HCCL_E_MEMORY;
    }

    inputMem = hccl::DeviceMem::create(sharedMem.inputBase, sharedMem.inputBytes);
    outputMem = hccl::DeviceMem::create(sharedMem.outputBase, sharedMem.outputBytes);
    if (!inputMem || !outputMem) {
        std::cerr << "BatchChannel: DeviceMem::create failed for shared regions" << std::endl;
        return HCCL_E_MEMORY;
    }
    localInputBase = static_cast<uint8_t *>(inputMem.ptr());
    localOutputBase = static_cast<uint8_t *>(outputMem.ptr());

    notifyNum = cfg.nBuffs * cfg.nChunksPerBuff + cfg.nBuffs +
                (cfg.waitRecvDone ? 1 : 0);

    hccl::MachinePara machinePara;
    machinePara.machineType = isListener ? hccl::MachineType::MACHINE_SERVER_TYPE
                                         : hccl::MachineType::MACHINE_CLIENT_TYPE;
    machinePara.linkMode = hccl::LinkMode::LINK_DUPLEX_MODE;
    machinePara.collectiveId = res.tag + "_comm";
    machinePara.tag = res.tag;
    machinePara.serverId = res.tag + "_server";
    machinePara.localIpAddr = res.localIp;
    machinePara.remoteIpAddr = res.remoteIp;
    machinePara.localSocketPort = 0;
    machinePara.remoteSocketPort = 0;
    machinePara.localDeviceId = static_cast<s32>(res.phyId);
    machinePara.remoteDeviceId = static_cast<s32>(peerDevId);
    // TransportIbverbs::GetNicHandle uses deviceLogicId as the NetworkManager key;
    // it must match NetDevContext::GetLogicId(). PingPongAgent calls HcclNetOpenDev
    // with (phyId, phyId) for logic and physical, not the torch logical index.
    machinePara.deviceLogicId = static_cast<s32>(res.phyId);
    machinePara.localUserrank = isListener ? 0 : 1;
    machinePara.remoteUserrank = isListener ? 1 : 0;
    machinePara.localWorldRank = machinePara.localUserrank;
    machinePara.remoteWorldRank = machinePara.remoteUserrank;
    machinePara.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    machinePara.deviceType = DevType::DEV_TYPE_910B;
    {
        DevType detected{};
        if (hrtGetDeviceType(detected) == HCCL_SUCCESS &&
            detected != DevType::DEV_TYPE_COUNT) {
            machinePara.deviceType = detected;
        }
    }
    machinePara.sockets.push_back(res.ctrlSocket);
    machinePara.inputMem = inputMem;
    machinePara.outputMem = outputMem;
    machinePara.linkAttribute = 0x03;   // READ | WRITE
    machinePara.supportDataReceivedAck = false;
    machinePara.isAicpuModeEn = false;
    machinePara.srcPorts.push_back(0);
    machinePara.notifyNum = notifyNum;
    machinePara.tc = cfg.tc;
    machinePara.sl = cfg.sl;
    machinePara.specifyLink = LinkTypeInServer::RESERVED_LINK_TYPE;

    hccl::TransportPara para{};
    // Per-notify timeout for transport->Wait. A single chunk-level notify
    // wait in healthy operation completes in microseconds; we still need
    // SOME headroom for slow links / kernel scheduling, but the value also
    // bounds how long a poisoned stream takes to drain after a peer-side
    // rejection.
    //
    // Scenario: a sender's _handle_read_request raises (e.g. unknown
    // receiver_id) and sends back a PingPongReadAck(ok=False) instead of
    // posting sendDoneSlot[]. The receiver's batched_read picks up the
    // error ack via _wait_for_ack_or_stream and raises RuntimeError, but
    // the recv_batch ops queued on transport_stream still sit on
    // device-side notify waits. At process exit, torch.npu.Stream.__del__
    // -> aclrtDestroyStream blocks until those waits drain. We want that
    // drain to happen inside the test's per-process timeout
    // (test_pingpong_read_failure_propagates uses 60 s), so we cap each
    // wait at 10 s — comfortably more than any sane chunk-level latency
    // and comfortably less than the test budget.
    para.timeout = std::chrono::seconds(10);
    para.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    para.transportResourceInfoAddr = nullptr;
    para.transportResourceInfoSize = 0;
    para.virtualFlag = false;

    transport = std::make_shared<hccl::Transport>(
        hccl::TransportType::TRANS_TYPE_IBV_EXP, para, res.dispatcher,
        notifyPool, machinePara);
    HCCLCHECK(transport->Init());

    void *remoteInputPtr = nullptr;
    HCCLCHECK(transport->GetRemoteMem(hccl::UserMemType::INPUT_MEM, &remoteInputPtr));
    remoteInputBase = reinterpret_cast<uintptr_t>(remoteInputPtr);

    HCCLCHECK(PrimeRecvReady());

    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Impl::PrimeRecvReady()
{
    aclrtStream aclStream = nullptr;
    ACLCHECK(aclrtCreateStream(&aclStream));

    hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(aclStream), true);
    for (uint32_t b = 0; b < cfg.nBuffs; ++b) {
        HcclResult ret = transport->Post(recvReadySlot(b), hcclStream);
        if (ret != HCCL_SUCCESS) {
            (void)aclrtDestroyStream(aclStream);
            std::cerr << "HCCL error " << ret << " at " << __FILE__ << ":" << __LINE__
                      << " (transport->Post(recvReadySlot(b), hcclStream))" << std::endl;
            return ret;
        }
    }

    aclError syncRet = aclrtSynchronizeStream(aclStream);
    aclError destroyRet = aclrtDestroyStream(aclStream);
    if (syncRet != ACL_SUCCESS) {
        std::cerr << "ACL error " << syncRet << " at " << __FILE__ << ":" << __LINE__
                  << " (aclrtSynchronizeStream(aclStream))" << std::endl;
        return HCCL_E_INTERNAL;
    }
    if (destroyRet != ACL_SUCCESS) {
        std::cerr << "ACL error " << destroyRet << " at " << __FILE__ << ":" << __LINE__
                  << " (aclrtDestroyStream(aclStream))" << std::endl;
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Init(uint32_t localDevId, uint32_t peerDevId,
                              bool isListener,
                              const BatchChannelSharedResources &res,
                              const BatchChannelSharedMemory &sharedMem,
                              const BatchChannelConfig &cfg)
{
    if (!impl_) return HCCL_E_INTERNAL;
    if (impl_->initialized) {
        std::cerr << "BatchChannel::Init called twice" << std::endl;
        return HCCL_E_INTERNAL;
    }
    if (cfg.chunkSizeBytes == 0 || (cfg.chunkSizeBytes % kPageSizeBytes) != 0) {
        std::cerr << "BatchChannel: chunkSizeBytes must be a non-zero multiple of "
                  << kPageSizeBytes << std::endl;
        return HCCL_E_PARA;
    }
    if (cfg.nBuffs != 2 || cfg.nChunksPerBuff == 0) {
        std::cerr << "BatchChannel: shared-region mode requires nBuffs == 2 and "
                  << "nChunksPerBuff > 0" << std::endl;
        return HCCL_E_PARA;
    }
    if (res.nicNetDevCtx == nullptr || res.dispatcher == nullptr ||
        res.ctrlSocket == nullptr || res.tag.empty()) {
        std::cerr << "BatchChannel: shared resources are incomplete" << std::endl;
        return HCCL_E_PARA;
    }

    impl_->cfg = cfg;
    impl_->res = res;
    impl_->sharedMem = sharedMem;
    impl_->localDevId = localDevId;
    impl_->peerDevId = peerDevId;
    impl_->isListener = isListener;

    HCCLCHECK(impl_->InitDataPlane());

    impl_->initialized = true;
    return HCCL_SUCCESS;
}

// --- Chunk primitives ----------------------------------------------------------------

HcclResult BatchChannel::Impl::SendChunk(uint64_t chunkOffset, uint32_t bufIdx,
    uint32_t chunkIdx, const SegmentList &packedSrcs, aclrtMemcpyKind memcpyKind,
    hccl::Stream &hcclStream, aclrtStream aclStream, bool isFirstChunkInBuffer)
{
    if (isFirstChunkInBuffer) {
        HCCLCHECK(transport->Wait(recvReadySlot(bufIdx), hcclStream));
    }

    uint64_t entryOffset = 0;
    uint8_t *chunkBase = localOutputBase + chunkOffset;
    const uint64_t chunkSize = cfg.chunkSizeBytes;
    for (const auto &entry : packedSrcs) {
        void *src = entry.first;
        uint64_t size = entry.second;
        ACLCHECK(aclrtMemcpyAsync(chunkBase + entryOffset, chunkSize - entryOffset,
                                  src, size, memcpyKind, aclStream));
        entryOffset += size;
    }

    hccl::Transport::Buffer remoteBuf(
        reinterpret_cast<void *>(remoteInputBase + chunkOffset), entryOffset);
    hccl::Transport::Buffer localBuf(chunkBase, entryOffset);
    HCCLCHECK(transport->WriteAsync(remoteBuf, localBuf, hcclStream));

    HCCLCHECK(transport->Post(sendDoneSlot(bufIdx, chunkIdx), hcclStream));
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Impl::RecvChunk(uint64_t chunkOffset, uint32_t bufIdx,
    uint32_t chunkIdx, const SegmentList &packedDsts,
    hccl::Stream &hcclStream, aclrtStream aclStream, bool isLastChunkInBuffer)
{
    HCCLCHECK(transport->Wait(sendDoneSlot(bufIdx, chunkIdx), hcclStream));

    uint64_t entryOffset = 0;
    const uint8_t *chunkBase = localInputBase + chunkOffset;
    for (const auto &entry : packedDsts) {
        void *dst = entry.first;
        uint64_t size = entry.second;
        ACLCHECK(aclrtMemcpyAsync(dst, size, chunkBase + entryOffset, size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, aclStream));
        entryOffset += size;
    }

    if (isLastChunkInBuffer) {
        HCCLCHECK(transport->Post(recvReadySlot(bufIdx), hcclStream));
    }
    return HCCL_SUCCESS;
}

// --- Ring drivers --------------------------------------------------------------------

HcclResult BatchChannel::Impl::PushSegments(const SegmentList &segs,
    aclrtMemcpyKind memcpyKind, hccl::Stream &hcclStream, aclrtStream aclStream)
{
    const uint64_t chunkSize = cfg.chunkSizeBytes;
    const uint32_t count = static_cast<uint32_t>(segs.size());
    uint32_t curSendBuffer = 0;
    uint32_t curSendChunk = 0;
    uint32_t i = 0;
    while (i < count) {
        uint64_t sizeI = segs[i].second;
        if (sizeI > chunkSize) {
            uint64_t remaining = sizeI;
            uint64_t srcOffset = 0;
            while (remaining > 0) {
                uint64_t thisChunk = std::min<uint64_t>(remaining, chunkSize);
                uint64_t chunkOffset = chunkOffsetInMem(curSendBuffer, curSendChunk);
                SegmentList packed;
                packed.emplace_back(
                    static_cast<uint8_t *>(segs[i].first) + srcOffset, thisChunk);
                HCCLCHECK(SendChunk(chunkOffset, curSendBuffer, curSendChunk, packed,
                                    memcpyKind, hcclStream, aclStream,
                                    /*isFirstChunkInBuffer=*/curSendChunk == 0));
                remaining -= thisChunk;
                srcOffset += thisChunk;
                curSendChunk++;
                if (curSendChunk == cfg.nChunksPerBuff) {
                    curSendChunk = 0;
                    curSendBuffer = (curSendBuffer + 1) % cfg.nBuffs;
                }
            }
            ++i;
        } else {
            SegmentList packed;
            uint64_t acc = 0;
            uint32_t j = i;
            while (j < count) {
                uint64_t sz = segs[j].second;
                if (sz > chunkSize) break;
                if (acc + sz > chunkSize) break;
                packed.emplace_back(segs[j].first, sz);
                acc += sz;
                ++j;
            }
            uint64_t chunkOffset = chunkOffsetInMem(curSendBuffer, curSendChunk);
            HCCLCHECK(SendChunk(chunkOffset, curSendBuffer, curSendChunk, packed,
                                memcpyKind, hcclStream, aclStream,
                                /*isFirstChunkInBuffer=*/curSendChunk == 0));
            curSendChunk++;
            if (curSendChunk == cfg.nChunksPerBuff) {
                curSendChunk = 0;
                curSendBuffer = (curSendBuffer + 1) % cfg.nBuffs;
            }
            i = j;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Impl::PullSegments(const SegmentList &segs,
    hccl::Stream &hcclStream, aclrtStream aclStream)
{
    const uint64_t chunkSize = cfg.chunkSizeBytes;
    const uint32_t count = static_cast<uint32_t>(segs.size());
    uint32_t curRecvBuffer = 0;
    uint32_t curRecvChunk = 0;
    uint32_t i = 0;
    while (i < count) {
        uint64_t sizeI = segs[i].second;
        if (sizeI > chunkSize) {
            uint64_t remaining = sizeI;
            uint64_t dstOffset = 0;
            while (remaining > 0) {
                uint64_t thisChunk = std::min<uint64_t>(remaining, chunkSize);
                uint64_t chunkOffset = chunkOffsetInMem(curRecvBuffer, curRecvChunk);
                SegmentList packed;
                packed.emplace_back(
                    static_cast<uint8_t *>(segs[i].first) + dstOffset, thisChunk);
                bool isLast = (curRecvChunk + 1 == cfg.nChunksPerBuff);
                HCCLCHECK(RecvChunk(chunkOffset, curRecvBuffer, curRecvChunk, packed,
                                    hcclStream, aclStream, isLast));
                remaining -= thisChunk;
                dstOffset += thisChunk;
                curRecvChunk++;
                if (curRecvChunk == cfg.nChunksPerBuff) {
                    curRecvChunk = 0;
                    curRecvBuffer = (curRecvBuffer + 1) % cfg.nBuffs;
                }
            }
            ++i;
        } else {
            SegmentList packed;
            uint64_t acc = 0;
            uint32_t j = i;
            while (j < count) {
                uint64_t sz = segs[j].second;
                if (sz > chunkSize) break;
                if (acc + sz > chunkSize) break;
                packed.emplace_back(segs[j].first, sz);
                acc += sz;
                ++j;
            }
            uint64_t chunkOffset = chunkOffsetInMem(curRecvBuffer, curRecvChunk);
            bool isLast = (curRecvChunk + 1 == cfg.nChunksPerBuff);
            HCCLCHECK(RecvChunk(chunkOffset, curRecvBuffer, curRecvChunk, packed,
                                hcclStream, aclStream, isLast));
            curRecvChunk++;
            if (curRecvChunk == cfg.nChunksPerBuff) {
                curRecvChunk = 0;
                curRecvBuffer = (curRecvBuffer + 1) % cfg.nBuffs;
            }
            i = j;
        }
    }
    if (curRecvChunk > 0) {
        HCCLCHECK(transport->Post(recvReadySlot(curRecvBuffer), hcclStream));
    }
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Impl::WaitRecvDoneIfEnabled(hccl::Stream &hcclStream)
{
    if (cfg.waitRecvDone) {
        HCCLCHECK(transport->Wait(recvDoneSlot(), hcclStream));
    }
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Impl::PostRecvDone(hccl::Stream &hcclStream)
{
    if (cfg.waitRecvDone) {
        HCCLCHECK(transport->Post(recvDoneSlot(), hcclStream));
    }
    return HCCL_SUCCESS;
}

// --- Public Send / Recv --------------------------------------------------------------

HcclResult BatchChannel::Send(void **srcPtrs, const uint64_t *sizes, uint32_t count,
                              aclrtStream stream)
{
    if (!impl_ || !impl_->initialized) return HCCL_E_INTERNAL;
    if (count == 0) return HCCL_SUCCESS;
    if (srcPtrs == nullptr || sizes == nullptr) return HCCL_E_PARA;

    hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);
    Impl::SegmentList segs;
    segs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        segs.emplace_back(srcPtrs[i], sizes[i]);
    }
    HCCLCHECK(impl_->PushSegments(segs, ACL_MEMCPY_DEVICE_TO_DEVICE, hcclStream, stream));
    return impl_->WaitRecvDoneIfEnabled(hcclStream);
}

HcclResult BatchChannel::Recv(void **dstPtrs, const uint64_t *sizes, uint32_t count,
                              aclrtStream stream)
{
    if (!impl_ || !impl_->initialized) return HCCL_E_INTERNAL;
    if (count == 0) return HCCL_SUCCESS;
    if (dstPtrs == nullptr || sizes == nullptr) return HCCL_E_PARA;

    hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);

    Impl::SegmentList segs;
    segs.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        segs.emplace_back(dstPtrs[i], sizes[i]);
    }
    HCCLCHECK(impl_->PullSegments(segs, hcclStream, stream));
    return impl_->PostRecvDone(hcclStream);
}

// --- Public ScatterSend / ScatterRecv ------------------------------------------------

HcclResult BatchChannel::ScatterSend(P2pScatterEntry *entries, uint32_t batchSize,
                                     aclrtStream stream)
{
    if (!impl_ || !impl_->initialized) return HCCL_E_INTERNAL;
    if (batchSize == 0) return HCCL_SUCCESS;
    if (entries == nullptr) return HCCL_E_PARA;

    hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);
    bool pushedAny = false;

    for (uint32_t b = 0; b < batchSize; ++b) {
        P2pScatterEntry &entry = entries[b];
        if (entry.numEl == 0) continue;
        if (entry.ddrBuf == nullptr || entry.counts == nullptr) return HCCL_E_PARA;

        u32 elemSize = 0;
        HCCLCHECK(SalGetDataTypeSize(entry.dataType, elemSize));
        if (elemSize == 0) return HCCL_E_PARA;

        Impl::SegmentList segs;
        segs.reserve(entry.numEl);
        uint64_t srcOffset = 0;
        uint8_t *base = static_cast<uint8_t *>(entry.ddrBuf);
        for (uint32_t k = 0; k < entry.numEl; ++k) {
            uint64_t bytes = entry.counts[k] * static_cast<uint64_t>(elemSize);
            if (bytes == 0) continue;
            segs.emplace_back(base + srcOffset, bytes);
            srcOffset += bytes;
        }
        if (segs.empty()) continue;

        // TODO(scatter_send_d2d_followup): support a D2D variant so NPU-backed source
        // pages can be RDMA-written directly without staging through host memory.
        HCCLCHECK(impl_->PushSegments(segs, ACL_MEMCPY_HOST_TO_DEVICE, hcclStream, stream));
        pushedAny = true;
    }
    if (pushedAny) {
        HCCLCHECK(impl_->WaitRecvDoneIfEnabled(hcclStream));
    }
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::ScatterRecv(P2pScatterEntry *entries, uint32_t batchSize,
                                     aclrtStream stream)
{
    if (!impl_ || !impl_->initialized) return HCCL_E_INTERNAL;
    if (batchSize == 0) return HCCL_SUCCESS;
    if (entries == nullptr) return HCCL_E_PARA;

    hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);
    bool pulledAny = false;

    for (uint32_t b = 0; b < batchSize; ++b) {
        P2pScatterEntry &entry = entries[b];
        if (entry.numEl == 0) continue;
        if (entry.dstBufs == nullptr || entry.counts == nullptr) return HCCL_E_PARA;

        u32 elemSize = 0;
        HCCLCHECK(SalGetDataTypeSize(entry.dataType, elemSize));
        if (elemSize == 0) return HCCL_E_PARA;

        Impl::SegmentList segs;
        segs.reserve(entry.numEl);
        for (uint32_t k = 0; k < entry.numEl; ++k) {
            uint64_t bytes = entry.counts[k] * static_cast<uint64_t>(elemSize);
            if (bytes == 0) continue;
            if (entry.dstBufs[k] == nullptr) return HCCL_E_PARA;
            segs.emplace_back(entry.dstBufs[k], bytes);
        }
        if (segs.empty()) continue;

        HCCLCHECK(impl_->PullSegments(segs, hcclStream, stream));
        pulledAny = true;
    }
    if (pulledAny) {
        HCCLCHECK(impl_->PostRecvDone(hcclStream));
    }
    return HCCL_SUCCESS;
}

// --- Teardown ------------------------------------------------------------------------

HcclResult BatchChannel::Impl::Teardown()
{
    transport.reset();
    if (notifyPool) {
        (void)notifyPool->UnregisterOp(res.tag.c_str());
        notifyPool->Destroy();
        notifyPool.reset();
    }
    inputMem = hccl::DeviceMem();
    outputMem = hccl::DeviceMem();
    localInputBase = nullptr;
    localOutputBase = nullptr;
    remoteInputBase = 0;
    initialized = false;
    return HCCL_SUCCESS;
}

HcclResult BatchChannel::Finalize()
{
    if (!impl_) return HCCL_SUCCESS;
    if (!impl_->initialized) return HCCL_SUCCESS;
    return impl_->Teardown();
}
