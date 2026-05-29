/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
// HCCL / libhccl_plf use the legacy libstdc++ std::string ABI; match
// csrc/hccl_pingpong/batch_channel.cc so symbols resolve at load time.
#define _GLIBCXX_USE_CXX11_ABI 0

#include "onesided_agent.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "adapter_hccp_common.h" // hrtGetDevicePhyIdByIndex, hrtGetDeviceType
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "pingpong_utils.h"
#include "stream_pub.h"
#include "task_logic_info_pub.h" // UserMemType

std::shared_ptr<OneSidedAgent> OneSidedAgent::instances[OS_MAX_LOCAL_DEVICES];
std::mutex OneSidedAgent::instanceMutex;

namespace {

HcclResult WaitSocketConnected(std::shared_ptr<hccl::HcclSocket> &socket) {
  const int timeoutMs = 120000;
  const int sleepMs = 10;
  int elapsedMs = 0;

  do {
    auto status = socket->GetStatus();
    if (status == hccl::HcclSocketStatus::SOCKET_OK) {
      return HCCL_SUCCESS;
    }
    if (status == hccl::HcclSocketStatus::SOCKET_ERROR) {
      return HCCL_E_INTERNAL;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    elapsedMs += sleepMs;
  } while (elapsedMs < timeoutMs);

  return HCCL_E_TIMEOUT;
}

HcclResult CheckAcl(aclError ret) {
  return ret == ACL_SUCCESS ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}

HcclResult CheckShape(const std::vector<MemDetails> &localMems,
                      const std::vector<MemDetails> &remoteMems) {
  if (localMems.size() != remoteMems.size()) {
    return HCCL_E_PARA;
  }
  for (size_t i = 0; i < localMems.size(); ++i) {
    if (localMems[i].size == 0 || remoteMems[i].size == 0 ||
        localMems[i].size != remoteMems[i].size) {
      return HCCL_E_PARA;
    }
  }
  return HCCL_SUCCESS;
}

} // namespace

OneSidedAgent::ConnState::~ConnState() {
  // Transport references the notifyPool; tear it down first.
  transport.reset();
  if (notifyPool) {
    (void)notifyPool->UnregisterOp(tag.c_str());
    notifyPool->Destroy();
    notifyPool.reset();
  }
  // inputMem/outputMem are non-owning DeviceMem::create() views over the
  // agent-owned staging buffer (freed once in FreeStaging via aclrtFree).
  // They hold no resource of their own, so there is nothing to release here;
  // they are simply dropped with this ConnState.
}

HcclResult
OneSidedAgent::GetInstance(uint32_t deviceId,
                           std::shared_ptr<OneSidedAgent> &outAgent) {
  if (deviceId >= OS_MAX_LOCAL_DEVICES) {
    return HCCL_E_PARA;
  }
  std::lock_guard<std::mutex> lock(instanceMutex);
  if (instances[deviceId] == nullptr) {
    instances[deviceId] =
        std::shared_ptr<OneSidedAgent>(new OneSidedAgent(deviceId));
  }
  outAgent = instances[deviceId];
  return HCCL_SUCCESS;
}

OneSidedAgent::~OneSidedAgent() {
  std::lock_guard<std::mutex> lock(agentMutex_);
  // Tear down connections (transports/notifyPools) before the resources they
  // depend on: ctrl/listen sockets, dispatcher, NIC ctx.
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
  FreeStaging();
}

HcclResult OneSidedAgent::AllocStaging() {
  stagingBytes_ = cfg_.stagingBytes;
  if (stagingBytes_ == 0) {
    return HCCL_E_PARA;
  }
  HcclResult ret = CheckAcl(
      aclrtMalloc(&stagingBase_, stagingBytes_, ACL_MEM_MALLOC_HUGE_FIRST));
  if (ret != HCCL_SUCCESS) {
    return ret;
  }
  ret = CheckAcl(aclrtMemset(stagingBase_, stagingBytes_, 0, stagingBytes_));
  if (ret != HCCL_SUCCESS) {
    FreeStaging();
  }
  return ret;
}

void OneSidedAgent::FreeStaging() {
  if (stagingBase_) {
    (void)aclrtFree(stagingBase_);
    stagingBase_ = nullptr;
  }
  stagingBytes_ = 0;
}

HcclResult OneSidedAgent::Init(const OneSidedConfig &cfg) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  if (state_ == State::INITIALIZED) {
    return HCCL_SUCCESS;
  }
  if (state_ == State::ERROR) {
    return HCCL_E_INTERNAL;
  }
  if (cfg.stagingBytes == 0 || cfg.numSlots == 0) {
    state_ = State::ERROR;
    return HCCL_E_PARA;
  }
  cfg_ = cfg;
  notifyNum_ = OS_NOTIFIES_PER_SLOT * cfg_.numSlots;

  u32 phy = 0;
  HCCL_CHECK(hrtGetDevicePhyIdByIndex(devId_, phy));
  phyId_ = phy;

  HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, static_cast<s32>(phyId_),
              static_cast<s32>(phyId_), false);
  HCCL_CHECK(GetLocalIpv4(phyId_, localIp_));
  HCCL_CHECK(HcclNetOpenDev(&nicNetDevCtx_, NicType::DEVICE_NIC_TYPE,
                            static_cast<s32>(phyId_), static_cast<s32>(phyId_),
                            localIp_));

  nicServerSocket_ = std::make_shared<hccl::HcclSocket>(nicNetDevCtx_, 0);
  HCCL_CHECK(nicServerSocket_->Init());
  HCCL_CHECK(nicServerSocket_->Listen());

  HCCL_CHECK(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL,
                                static_cast<s32>(phyId_), &dispatcher_));

  HcclResult allocRet = AllocStaging();
  if (allocRet != HCCL_SUCCESS) {
    state_ = State::ERROR;
    return allocRet;
  }

  state_ = State::INITIALIZED;
  return HCCL_SUCCESS;
}

void OneSidedAgent::FillCommonMeta(HcclQpMeta &meta) const {
  meta.devId = static_cast<int32_t>(devId_);
  meta.phyDevId = static_cast<int32_t>(phyId_);
  meta.ipv4Addr = localIp_.GetBinaryAddress().addr.s_addr;
}

HcclResult OneSidedAgent::GetClientMeta(HcclQpMeta &meta) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  if (state_ != State::INITIALIZED)
    return HCCL_E_INTERNAL;
  FillCommonMeta(meta);
  // Client connects out, so it advertises no listen port and an empty tag.
  meta.listenPort = 0;
  std::memset(meta.tagCtrl, 0, OS_TAG_SIZE);
  return HCCL_SUCCESS;
}

HcclResult OneSidedAgent::GetServerMeta(HcclQpMeta &meta) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  if (state_ != State::INITIALIZED)
    return HCCL_E_INTERNAL;
  FillCommonMeta(meta);
  // Server listens, so it advertises its port and a fresh whitelist tag.
  meta.listenPort = nicServerSocket_->GetLocalPort();
  FillRandom(meta.tagCtrl, OS_TAG_SIZE);
  return HCCL_SUCCESS;
}

HcclResult OneSidedAgent::Accept(const HcclQpMeta &remoteMeta,
                                 OneSidedConn &conn) {
  if (state_ != State::INITIALIZED)
    return HCCL_E_INTERNAL;

  std::vector<SocketWlistInfo> wlist(1);
  wlist[0].connLimit = 1;
  std::memcpy(&wlist[0].tag[0], remoteMeta.tagCtrl, OS_TAG_SIZE);
  wlist[0].remoteIp.addr.s_addr = remoteMeta.ipv4Addr;
  HCCL_CHECK(nicServerSocket_->AddWhiteList(wlist));

  std::string tag(remoteMeta.tagCtrl, OS_TAG_SIZE);
  std::shared_ptr<hccl::HcclSocket> ctrlSocket;
  HCCL_CHECK(nicServerSocket_->Accept(tag, ctrlSocket));
  return BuildConnection(remoteMeta, /*isListener=*/true, ctrlSocket, conn);
}

HcclResult OneSidedAgent::Connect(const HcclQpMeta &remoteMeta,
                                  OneSidedConn &conn) {
  if (state_ != State::INITIALIZED)
    return HCCL_E_INTERNAL;

  hccl::HcclIpAddress remoteIp(remoteMeta.ipv4Addr);
  std::string tag(remoteMeta.tagCtrl, OS_TAG_SIZE);

  auto ctrlSocket = std::make_shared<hccl::HcclSocket>(
      tag, nicNetDevCtx_, remoteIp, remoteMeta.listenPort,
      hccl::HcclSocketRole::SOCKET_ROLE_CLIENT);
  HCCL_CHECK(ctrlSocket->Init());
  HCCL_CHECK(ctrlSocket->Connect());
  HCCL_CHECK(WaitSocketConnected(ctrlSocket));

  return BuildConnection(remoteMeta, /*isListener=*/false, ctrlSocket, conn);
}

HcclResult
OneSidedAgent::BuildConnection(const HcclQpMeta &remoteMeta, bool isListener,
                               std::shared_ptr<hccl::HcclSocket> ctrlSocket,
                               OneSidedConn &conn) {
  if (!ctrlSocket)
    return HCCL_E_PARA;

  auto state = std::unique_ptr<ConnState>(new ConnState());
  state->tag.assign(remoteMeta.tagCtrl, OS_TAG_SIZE);
  state->ctrlSocket = ctrlSocket;
  state->notifyPool = std::make_unique<hccl::NotifyPool>();
  HCCL_CHECK(state->notifyPool->Init(static_cast<s32>(phyId_)));
  HCCL_CHECK(state->notifyPool->RegisterOp(state->tag.c_str()));

  // Single bounded staging region, registered as both input and output mem.
  state->inputMem = hccl::DeviceMem::create(stagingBase_, stagingBytes_);
  state->outputMem = hccl::DeviceMem::create(stagingBase_, stagingBytes_);
  if (!state->inputMem || !state->outputMem) {
    return HCCL_E_MEMORY;
  }

  hccl::MachinePara machinePara;
  machinePara.machineType = isListener ? hccl::MachineType::MACHINE_SERVER_TYPE
                                       : hccl::MachineType::MACHINE_CLIENT_TYPE;
  machinePara.linkMode = hccl::LinkMode::LINK_DUPLEX_MODE;
  machinePara.collectiveId = state->tag + "_comm";
  machinePara.tag = state->tag;
  machinePara.serverId = state->tag + "_server";
  machinePara.localIpAddr = localIp_;
  machinePara.remoteIpAddr = hccl::HcclIpAddress(remoteMeta.ipv4Addr);
  machinePara.localSocketPort = 0;
  machinePara.remoteSocketPort = 0;
  machinePara.localDeviceId = static_cast<s32>(phyId_);
  machinePara.remoteDeviceId = static_cast<s32>(remoteMeta.phyDevId);
  machinePara.deviceLogicId = static_cast<s32>(phyId_);
  machinePara.localUserrank = isListener ? 0 : 1;
  machinePara.remoteUserrank = isListener ? 1 : 0;
  machinePara.localWorldRank = machinePara.localUserrank;
  machinePara.remoteWorldRank = machinePara.remoteUserrank;
  machinePara.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
  // DEV_TYPE_910B is only the fallback for the currently supported SoC.
  // hrtGetDeviceType detects the actual device at runtime and overrides it.
  // TODO(onesided): validate/extend this path as we broaden support beyond
  // 910B (e.g. other Ascend generations).
  machinePara.deviceType = DevType::DEV_TYPE_910B;
  {
    DevType detected{};
    if (hrtGetDeviceType(detected) == HCCL_SUCCESS &&
        detected != DevType::DEV_TYPE_COUNT) {
      machinePara.deviceType = detected;
    }
  }
  machinePara.sockets.push_back(ctrlSocket);
  machinePara.inputMem = state->inputMem;
  machinePara.outputMem = state->outputMem;
  // bit0 WRITE | bit1 READ. The receiver issues one-sided reads, so READ must
  // be enabled; keep WRITE on too for parity with the proven pingpong link.
  machinePara.linkAttribute = 0x03;
  machinePara.supportDataReceivedAck = false;
  // Host-driven path: we enqueue WriteAsync/ReadAsync/Post/Wait from the host
  // onto an ACL stream. AICPU unfold mode must stay OFF for this. Turning it
  // on switches the transport to the AICPU-driven doorbell model (an AICPU
  // kernel drives the QP / batched reads+writes), which is a different
  // execution model that our host-side stream ops do not satisfy.
  // TODO(onesided): revisit isAicpuModeEn=true if/when we adopt the native
  // AICPU batched read fast path.
  machinePara.isAicpuModeEn = false;
  machinePara.srcPorts.push_back(0);
  // One (data_ready, consumed) pair per staging slot — see OneSidedConfig.
  machinePara.notifyNum = notifyNum_;
  machinePara.tc = cfg_.tc;
  machinePara.sl = cfg_.sl;
  machinePara.specifyLink = LinkTypeInServer::RESERVED_LINK_TYPE;

  hccl::TransportPara para{};
  para.timeout = std::chrono::seconds(std::max<int32_t>(cfg_.timeoutSec, 1));
  para.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
  para.transportResourceInfoAddr = nullptr;
  para.transportResourceInfoSize = 0;
  para.virtualFlag = false;

  state->transport = std::make_shared<hccl::Transport>(
      hccl::TransportType::TRANS_TYPE_IBV_EXP, para, dispatcher_,
      state->notifyPool, machinePara);
  HCCL_CHECK(state->transport->Init());

  // Capture the peer's staging base for one-sided reads. Both ends register
  // their staging as INPUT_MEM, so the receiver reads remote INPUT_MEM.
  void *remotePtr = nullptr;
  HCCL_CHECK(
      state->transport->GetRemoteMem(hccl::UserMemType::INPUT_MEM, &remotePtr));
  state->remoteStagingBase = reinterpret_cast<uintptr_t>(remotePtr);

  u64 remoteSize = 0;
  if (state->transport->GetRemoteMemSize(hccl::UserMemType::INPUT_MEM,
                                         remoteSize) == HCCL_SUCCESS &&
      remoteSize > 0) {
    state->remoteStagingBytes = remoteSize;
  } else {
    state->remoteStagingBytes = stagingBytes_;
  }

  ConnState *raw = state.get();
  {
    std::lock_guard<std::mutex> lock(agentMutex_);
    auto inserted =
        conns_.emplace(static_cast<OneSidedConn>(raw), std::move(state));
    if (!inserted.second)
      return HCCL_E_INTERNAL;
  }
  conn = static_cast<OneSidedConn>(raw);
  return HCCL_SUCCESS;
}

OneSidedAgent::ConnState *OneSidedAgent::LookupConn(OneSidedConn conn) {
  std::lock_guard<std::mutex> lock(agentMutex_);
  auto it = conns_.find(conn);
  if (it == conns_.end())
    return nullptr;
  return it->second.get();
}

HcclResult OneSidedAgent::CloseConnection(OneSidedConn conn) {
  std::unique_ptr<ConnState> doomed;
  {
    std::lock_guard<std::mutex> lock(agentMutex_);
    auto it = conns_.find(conn);
    if (it == conns_.end()) {
      // Idempotent: already torn down (e.g. both error paths racing).
      return HCCL_SUCCESS;
    }
    // Move the ConnState out under the lock, then let it destruct AFTER
    // releasing agentMutex_. The destructor tears down the transport and
    // NotifyPool, which must not run while holding the agent lock (avoids
    // re-entrancy / lock-order issues against other agent calls).
    doomed = std::move(it->second);
    conns_.erase(it);
  }
  // ConnState destructor (transport.reset + notifyPool Unregister/Destroy)
  // runs here as `doomed` goes out of scope.
  return HCCL_SUCCESS;
}

HcclResult OneSidedAgent::BatchRead(OneSidedConn conn,
                                    const std::vector<MemDetails> &localMems,
                                    const std::vector<MemDetails> &remoteMems,
                                    aclrtStream stream) {
  ConnState *state = LookupConn(conn);
  if (state == nullptr || !state->transport || stream == nullptr) {
    return HCCL_E_PARA;
  }
  HCCL_CHECK(CheckShape(localMems, remoteMems));
  if (localMems.empty()) {
    return HCCL_SUCCESS;
  }

  const uintptr_t localBase = reinterpret_cast<uintptr_t>(stagingBase_);
  hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);

  for (size_t i = 0; i < localMems.size(); ++i) {
    const uint64_t localOff = localMems[i].addr;
    const uint64_t remoteOff = remoteMems[i].addr;
    const uint64_t size = localMems[i].size;

    if (localOff + size > stagingBytes_ ||
        remoteOff + size > state->remoteStagingBytes) {
      return HCCL_E_PARA;
    }

    hccl::Transport::Buffer localBuf(
        reinterpret_cast<void *>(localBase + localOff), size);
    hccl::Transport::Buffer remoteBuf(
        reinterpret_cast<void *>(state->remoteStagingBase + remoteOff), size);
    HCCL_CHECK(state->transport->ReadAsync(localBuf, remoteBuf, hcclStream));
  }
  return HCCL_SUCCESS;
}

HcclResult OneSidedAgent::BatchWrite(OneSidedConn conn,
                                     const std::vector<MemDetails> &localMems,
                                     const std::vector<MemDetails> &remoteMems,
                                     aclrtStream stream) {
  ConnState *state = LookupConn(conn);
  if (state == nullptr || !state->transport || stream == nullptr) {
    return HCCL_E_PARA;
  }
  HCCL_CHECK(CheckShape(localMems, remoteMems));
  if (localMems.empty()) {
    return HCCL_SUCCESS;
  }

  const uintptr_t localBase = reinterpret_cast<uintptr_t>(stagingBase_);
  hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);

  for (size_t i = 0; i < localMems.size(); ++i) {
    const uint64_t localOff = localMems[i].addr;
    const uint64_t remoteOff = remoteMems[i].addr;
    const uint64_t size = localMems[i].size;

    if (localOff + size > stagingBytes_ ||
        remoteOff + size > state->remoteStagingBytes) {
      return HCCL_E_PARA;
    }

    hccl::Transport::Buffer remoteBuf(
        reinterpret_cast<void *>(state->remoteStagingBase + remoteOff), size);
    hccl::Transport::Buffer localBuf(
        reinterpret_cast<void *>(localBase + localOff), size);
    HCCL_CHECK(state->transport->WriteAsync(remoteBuf, localBuf, hcclStream));
  }
  return HCCL_SUCCESS;
}

HcclResult OneSidedAgent::Post(OneSidedConn conn, uint32_t notifyIdx,
                               aclrtStream stream) {
  ConnState *state = LookupConn(conn);
  if (state == nullptr || !state->transport || notifyIdx >= notifyNum_ ||
      stream == nullptr) {
    return HCCL_E_PARA;
  }
  hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);
  return state->transport->Post(notifyIdx, hcclStream);
}

HcclResult OneSidedAgent::Wait(OneSidedConn conn, uint32_t notifyIdx,
                               aclrtStream stream) {
  ConnState *state = LookupConn(conn);
  if (state == nullptr || !state->transport || notifyIdx >= notifyNum_ ||
      stream == nullptr) {
    return HCCL_E_PARA;
  }
  hccl::Stream hcclStream(reinterpret_cast<rtStream_t>(stream), true);
  return state->transport->Wait(notifyIdx, hcclStream);
}

void *OneSidedAgent::GetStagingBase() const { return stagingBase_; }

uint64_t OneSidedAgent::GetStagingBytes() const { return stagingBytes_; }

uint64_t OneSidedAgent::GetRemoteStagingBytes(OneSidedConn conn) {
  ConnState *state = LookupConn(conn);
  if (state == nullptr)
    return 0;
  return state->remoteStagingBytes;
}

uint32_t OneSidedAgent::GetNumNotifies() const { return notifyNum_; }
