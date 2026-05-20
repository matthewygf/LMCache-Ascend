/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include "pingpong_agent.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "pingpong_utils.h"

#include "hccl_common.h"
#include "adapter_hccp_common.h"   // hrtGetDevicePhyIdByIndex

std::shared_ptr<PingPongAgent>
    PingPongAgent::instances[PP_MAX_LOCAL_DEVICES];
std::mutex PingPongAgent::instanceMutex;

namespace {

HcclResult WaitSocketConnected(std::shared_ptr<hccl::HcclSocket> &socket)
{
    const int timeout_ms = 120000;
    const int sleep_ms = 10;
    int elapsed_ms = 0;

    do {
        auto status = socket->GetStatus();
        if (status == hccl::HcclSocketStatus::SOCKET_OK) {
            return HCCL_SUCCESS;
        }
        if (status == hccl::HcclSocketStatus::SOCKET_ERROR) {
            return HCCL_E_INTERNAL;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        elapsed_ms += sleep_ms;
    } while (elapsed_ms < timeout_ms);

    return HCCL_E_TIMEOUT;
}

}  // namespace

HcclResult
PingPongAgent::GetInstance(uint32_t deviceId,
                           std::shared_ptr<PingPongAgent> &outAgent)
{
    if (deviceId >= PP_MAX_LOCAL_DEVICES) {
        return HCCL_E_PARA;
    }
    std::lock_guard<std::mutex> lock(instanceMutex);
    if (instances[deviceId] == nullptr) {
        instances[deviceId] =
            std::shared_ptr<PingPongAgent>(new PingPongAgent(deviceId));
    }
    outAgent = instances[deviceId];
    return HCCL_SUCCESS;
}

PingPongAgent::~PingPongAgent()
{
    std::lock_guard<std::mutex> lock(agentMutex_);
    // Tear down BatchChannels first; they hold references to ctrl sockets which
    // outlive them up the dependency chain (NIC ctx -> dispatcher).
    conns_.clear();
    nicServerSocket_.reset();
    if (dispatcher_) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    if (nicNetDevCtx_) {
        HcclNetCloseDev(nicNetDevCtx_);
        nicNetDevCtx_ = nullptr;
    }
    FreeSharedRegions();
}

uint64_t PingPongAgent::SharedRegionBytes() const
{
    return sharedRegionBytes_;
}

HcclResult PingPongAgent::AllocSharedRegions()
{
    sharedRegionBytes_ = cfg_.chunkSizeBytes * cfg_.nChunksPerBuff * cfg_.nBuffs;
    ACL_CHECK(aclrtMalloc(&inputBase_, sharedRegionBytes_,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&outputBase_, sharedRegionBytes_,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(inputBase_, sharedRegionBytes_, 0, sharedRegionBytes_));
    ACL_CHECK(aclrtMemset(outputBase_, sharedRegionBytes_, 0, sharedRegionBytes_));
    return HCCL_SUCCESS;
}

void PingPongAgent::FreeSharedRegions()
{
    if (inputBase_) {
        (void)aclrtFree(inputBase_);
        inputBase_ = nullptr;
    }
    if (outputBase_) {
        (void)aclrtFree(outputBase_);
        outputBase_ = nullptr;
    }
    sharedRegionBytes_ = 0;
}

HcclResult PingPongAgent::Init(const PingPongConfig &cfg)
{
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ == State::INITIALIZED) {
        return HCCL_SUCCESS;
    }
    if (state_ == State::ERROR) {
        return HCCL_E_INTERNAL;
    }

    if (cfg.nBuffs != 2 || cfg.nChunksPerBuff == 0 || cfg.chunkSizeBytes == 0) {
        std::cerr << "PingPongAgent: invalid config (nBuffs must be 2, others non-zero)"
                  << std::endl;
        state_ = State::ERROR;
        return HCCL_E_PARA;
    }
    cfg_ = cfg;

    u32 phy = 0;
    HCCL_CHECK(hrtGetDevicePhyIdByIndex(devId_, phy));
    phyId_ = phy;

    // See HcclAgent::Init() comment: passing phyId for both args until upstream fix.
    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE,
                static_cast<s32>(phyId_), static_cast<s32>(phyId_), false);
    HCCL_CHECK(GetLocalIpv4(phyId_, localIp_));
    HCCL_CHECK(HcclNetOpenDev(&nicNetDevCtx_, NicType::DEVICE_NIC_TYPE,
                              static_cast<s32>(phyId_),
                              static_cast<s32>(phyId_), localIp_));

    nicServerSocket_ = std::make_shared<hccl::HcclSocket>(nicNetDevCtx_, 0);
    HCCL_CHECK(nicServerSocket_->Init());
    HCCL_CHECK(nicServerSocket_->Listen());

    HCCL_CHECK(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL,
                                  static_cast<s32>(phyId_), &dispatcher_));

    HcclResult allocRet = AllocSharedRegions();
    if (allocRet != HCCL_SUCCESS) {
        FreeSharedRegions();
        state_ = State::ERROR;
        return allocRet;
    }

    state_ = State::INITIALIZED;
    return HCCL_SUCCESS;
}

HcclResult PingPongAgent::GetClientMeta(PingPongClientMeta &meta)
{
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;
    meta.devId = static_cast<int32_t>(devId_);
    meta.phyDevId = static_cast<int32_t>(phyId_);
    meta.ipv4Addr = localIp_.GetBinaryAddress().addr.s_addr;
    return HCCL_SUCCESS;
}

HcclResult PingPongAgent::GetServerMeta(PingPongServerMeta &meta)
{
    std::lock_guard<std::mutex> lock(agentMutex_);
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;
    meta.devId = static_cast<int32_t>(devId_);
    meta.phyDevId = static_cast<int32_t>(phyId_);
    meta.ipv4Addr = localIp_.GetBinaryAddress().addr.s_addr;
    meta.listenPort = nicServerSocket_->GetLocalPort();
    FillRandom(meta.tagCtrl, PP_TAG_SIZE);
    return HCCL_SUCCESS;
}

HcclResult PingPongAgent::Accept(const PingPongClientMeta &client,
                                 const PingPongServerMeta &server,
                                 PingPongConn &conn)
{
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    // Per-peer whitelist + accept on the SHARED listen socket. Each peer is told
    // by upper layers to use this same listenPort and a freshly-randomized tagCtrl,
    // so the listener disambiguates concurrent connections via the tag.
    std::vector<SocketWlistInfo> wlist(1);
    wlist[0].connLimit = 1;
    std::memcpy(&wlist[0].tag[0], server.tagCtrl, PP_TAG_SIZE);
    wlist[0].remoteIp.addr.s_addr = client.ipv4Addr;
    HCCL_CHECK(nicServerSocket_->AddWhiteList(wlist));

    std::string tag(server.tagCtrl, PP_TAG_SIZE);
    std::shared_ptr<hccl::HcclSocket> ctrlSocket;
    HCCL_CHECK(nicServerSocket_->Accept(tag, ctrlSocket));

    BatchChannelSharedResources res;
    res.nicNetDevCtx = nicNetDevCtx_;
    res.dispatcher = dispatcher_;
    res.ctrlSocket = ctrlSocket;
    res.localIp = localIp_;
    res.remoteIp = hccl::HcclIpAddress(client.ipv4Addr);
    res.phyId = phyId_;
    res.tag = tag;

    BatchChannelSharedMemory mem;
    mem.inputBase = inputBase_;
    mem.inputBytes = sharedRegionBytes_;
    mem.outputBase = outputBase_;
    mem.outputBytes = sharedRegionBytes_;

    BatchChannelConfig bcCfg;
    bcCfg.chunkSizeBytes = cfg_.chunkSizeBytes;
    bcCfg.nChunksPerBuff = cfg_.nChunksPerBuff;
    bcCfg.nBuffs = cfg_.nBuffs;
    bcCfg.waitRecvDone = cfg_.waitRecvDone;
    bcCfg.tc = cfg_.tc;
    bcCfg.sl = cfg_.sl;

    auto channel = std::unique_ptr<BatchChannel>(new BatchChannel());
    HCCL_CHECK(channel->Init(devId_, static_cast<uint32_t>(client.phyDevId),
                             /*isListener=*/true, res, mem, bcCfg));

    BatchChannel *raw = channel.get();
    {
        std::lock_guard<std::mutex> lock(agentMutex_);
        auto inserted = conns_.emplace(static_cast<PingPongConn>(raw),
                                       std::move(channel));
        if (!inserted.second) return HCCL_E_INTERNAL;
    }
    conn = static_cast<PingPongConn>(raw);
    return HCCL_SUCCESS;
}

HcclResult PingPongAgent::Connect(const PingPongServerMeta &server,
                                  PingPongConn &conn)
{
    if (state_ != State::INITIALIZED) return HCCL_E_INTERNAL;

    hccl::HcclIpAddress remoteIp(server.ipv4Addr);
    std::string tag(server.tagCtrl, PP_TAG_SIZE);

    auto ctrlSocket = std::make_shared<hccl::HcclSocket>(
        tag, nicNetDevCtx_, remoteIp, server.listenPort,
        hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
    HCCL_CHECK(ctrlSocket->Init());
    HCCL_CHECK(ctrlSocket->Connect());
    HCCL_CHECK(WaitSocketConnected(ctrlSocket));

    BatchChannelSharedResources res;
    res.nicNetDevCtx = nicNetDevCtx_;
    res.dispatcher = dispatcher_;
    res.ctrlSocket = ctrlSocket;
    res.localIp = localIp_;
    res.remoteIp = remoteIp;
    res.phyId = phyId_;
    res.tag = tag;

    BatchChannelSharedMemory mem;
    mem.inputBase = inputBase_;
    mem.inputBytes = sharedRegionBytes_;
    mem.outputBase = outputBase_;
    mem.outputBytes = sharedRegionBytes_;

    BatchChannelConfig bcCfg;
    bcCfg.chunkSizeBytes = cfg_.chunkSizeBytes;
    bcCfg.nChunksPerBuff = cfg_.nChunksPerBuff;
    bcCfg.nBuffs = cfg_.nBuffs;
    bcCfg.waitRecvDone = cfg_.waitRecvDone;
    bcCfg.tc = cfg_.tc;
    bcCfg.sl = cfg_.sl;

    auto channel = std::unique_ptr<BatchChannel>(new BatchChannel());
    HCCL_CHECK(channel->Init(devId_, static_cast<uint32_t>(server.phyDevId),
                             /*isListener=*/false, res, mem, bcCfg));

    BatchChannel *raw = channel.get();
    {
        std::lock_guard<std::mutex> lock(agentMutex_);
        auto inserted = conns_.emplace(static_cast<PingPongConn>(raw),
                                       std::move(channel));
        if (!inserted.second) return HCCL_E_INTERNAL;
    }
    conn = static_cast<PingPongConn>(raw);
    return HCCL_SUCCESS;
}

BatchChannel *PingPongAgent::LookupConn(PingPongConn conn)
{
    std::lock_guard<std::mutex> lock(agentMutex_);
    auto it = conns_.find(conn);
    if (it == conns_.end()) return nullptr;
    return it->second.get();
}

HcclResult PingPongAgent::SendBatch(PingPongConn conn,
                                    const std::vector<PingPongOp> &ops,
                                    aclrtStream stream)
{
    BatchChannel *channel = LookupConn(conn);
    if (channel == nullptr) return HCCL_E_INTERNAL;
    if (ops.empty()) return HCCL_SUCCESS;

    std::vector<void *> ptrs;
    std::vector<uint64_t> sizes;
    ptrs.reserve(ops.size());
    sizes.reserve(ops.size());
    for (const auto &op : ops) {
        ptrs.push_back(op.localAddr);
        sizes.push_back(op.size);
    }
    return channel->Send(ptrs.data(), sizes.data(),
                         static_cast<uint32_t>(ops.size()), stream);
}

HcclResult PingPongAgent::RecvBatch(PingPongConn conn,
                                    const std::vector<PingPongOp> &ops,
                                    aclrtStream stream)
{
    BatchChannel *channel = LookupConn(conn);
    if (channel == nullptr) return HCCL_E_INTERNAL;
    if (ops.empty()) return HCCL_SUCCESS;

    std::vector<void *> ptrs;
    std::vector<uint64_t> sizes;
    ptrs.reserve(ops.size());
    sizes.reserve(ops.size());
    for (const auto &op : ops) {
        ptrs.push_back(op.localAddr);
        sizes.push_back(op.size);
    }
    return channel->Recv(ptrs.data(), sizes.data(),
                         static_cast<uint32_t>(ops.size()), stream);
}

HcclResult PingPongAgent::ScatterSend(PingPongConn conn,
                                      std::vector<PingPongScatterEntry> &entries,
                                      aclrtStream stream)
{
    BatchChannel *channel = LookupConn(conn);
    if (channel == nullptr) return HCCL_E_INTERNAL;
    if (entries.empty()) return HCCL_SUCCESS;

    std::vector<P2pScatterEntry> raw(entries.size());
    for (size_t b = 0; b < entries.size(); ++b) {
        auto &dst = raw[b];
        auto &src = entries[b];
        if (src.counts.size() != 0 && src.counts.size() < UINT32_MAX) {
            dst.numEl = static_cast<uint32_t>(src.counts.size());
        } else {
            return HCCL_E_PARA;
        }
        dst.ddrBuf = src.ddrBuf;
        dst.dstBufs = nullptr;
        dst.counts = src.counts.data();
        dst.dataType = src.dataType;
    }
    return channel->ScatterSend(raw.data(),
                                static_cast<uint32_t>(raw.size()), stream);
}

HcclResult PingPongAgent::ScatterRecv(PingPongConn conn,
                                      std::vector<PingPongScatterEntry> &entries,
                                      aclrtStream stream)
{
    BatchChannel *channel = LookupConn(conn);
    if (channel == nullptr) return HCCL_E_INTERNAL;
    if (entries.empty()) return HCCL_SUCCESS;

    std::vector<P2pScatterEntry> raw(entries.size());
    for (size_t b = 0; b < entries.size(); ++b) {
        auto &dst = raw[b];
        auto &src = entries[b];
        if (src.counts.size() != src.dstBufs.size()) return HCCL_E_PARA;
        if (src.counts.size() == 0 || src.counts.size() >= UINT32_MAX) {
            return HCCL_E_PARA;
        }
        dst.ddrBuf = nullptr;
        dst.dstBufs = src.dstBufs.data();
        dst.counts = src.counts.data();
        dst.dataType = src.dataType;
        dst.numEl = static_cast<uint32_t>(src.counts.size());
    }
    return channel->ScatterRecv(raw.data(),
                                static_cast<uint32_t>(raw.size()), stream);
}
