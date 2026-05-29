/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * OneSidedAgent: singleton-per-device facade for a one-sided RoCE staging
 * data plane.
 *
 * Design (phase 1, staging-only — see plan hccl_onesided_channel):
 *  - A hardware limitation prevents registering the whole KV pool with HCCL, so
 *    each agent registers exactly ONE bounded staging region (used as both the
 *    transport input and output memory). User KV buffers are never registered.
 *  - The handshake mirrors PingPongAgent (shared listen socket, per-peer tag).
 *  - Each connection owns a SINGLE hccl::Transport (TRANS_TYPE_IBV_EXP) over
 * one control socket — the same proven object BatchChannel uses. That transport
 *    provides the one-sided operations (WriteAsync/ReadAsync) and the notifies
 *    (Post/Wait).
 *    Notifies are allocated per staging slot: slot s -> (data_ready = 2*s,
 *    consumed = 2*s + 1), because HCCL notifies are binary and a shared
 *    data_ready would coalesce back-to-back posts and hang a pipeline. There is
 *    intentionally no second transport and no TransportMem descriptor exchange.
 *  - Preferred write-driven protocol: sender D2D-copies requested pages into
 *    its staging region, WriteAsync(local staging -> remote staging), then
 *    Post(data_ready). Receiver Wait(data_ready), copies out of its local
 *    staging region, then Post(consumed).
 *  - BatchRead remains available for probes, but should not drive the slot
 *    protocol because ReadAsync completion is not dependency-guarded for later
 *    async device-to-host/device copies.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "dispatcher.h"
#include "hccl_ip_address.h"
#include "hccl_network_pub.h"
#include "hccl_socket.h"
#include "mem_device_pub.h"
#include "notify_pool.h"
#include "transport_pub.h"

constexpr uint32_t OS_MAX_LOCAL_DEVICES = 16;
constexpr uint32_t OS_TAG_SIZE = 32;

// Per-staging-slot notify layout. HCCL notifies are binary (a second Post
// before the matching Wait coalesces into one set), so every staging slot owns
// its OWN (data_ready, consumed) pair. A single shared data_ready would hang a
// pipelined ping-pong: the sender posts data_ready for two different slots
// back-to-back before the receiver waits, and the second signal would be lost.
// Notify index for slot `s`: data_ready = 2*s, consumed = 2*s + 1.
constexpr uint32_t OS_NOTIFIES_PER_SLOT = 2;
constexpr uint32_t OS_NOTIFY_DATA_READY = 0; // offset within a slot's pair
constexpr uint32_t OS_NOTIFY_CONSUMED = 1;   // offset within a slot's pair

// Notify index helper: maps (slot, which) -> absolute transport notify index.
constexpr uint32_t OsNotifyIndex(uint32_t slot, uint32_t which) {
  return slot * OS_NOTIFIES_PER_SLOT + which;
}

struct OneSidedConfig {
  uint64_t stagingBytes = 64ULL * 1024 * 1024;
  // Number of staging slots the data plane pipelines over. The transport
  // allocates OS_NOTIFIES_PER_SLOT * numSlots notifies at Init time, so this
  // must cover the largest slot index any Post/Wait will reference.
  uint32_t numSlots = 2;
  uint32_t tc = 132;
  uint32_t sl = 4;
  // Per-notify-wait transport timeout. Kept low (mirrors pingpong's
  // BatchChannel para.timeout) so a stale/desynced Wait left behind by a
  // failed transfer drains in ~10s instead of stalling the receiver stream
  // for minutes before error-path teardown can reclaim the connection.
  int32_t timeoutSec = 10;
};

struct HcclQpMeta {
  /// Torch logical device index (same as OneSidedAgent devId_).
  int32_t devId{0};
  /// Ascend physical device id — required for MachinePara local/remoteDeviceId.
  int32_t phyDevId{0};
  uint32_t ipv4Addr{0};
  uint32_t listenPort{0};
  char
      tagCtrl[OS_TAG_SIZE]; // random per-peer; also reused as the transport tag
};

using OneSidedConn = void *;

class OneSidedAgent {
public:
  static HcclResult GetInstance(uint32_t deviceId,
                                std::shared_ptr<OneSidedAgent> &outAgent);

  OneSidedAgent(const OneSidedAgent &) = delete;
  OneSidedAgent &operator=(const OneSidedAgent &) = delete;
  OneSidedAgent(OneSidedAgent &&) = delete;
  OneSidedAgent &operator=(OneSidedAgent &&) = delete;

  ~OneSidedAgent();

  HcclResult Init(const OneSidedConfig &cfg);
  HcclResult GetClientMeta(HcclQpMeta &meta);
  HcclResult GetServerMeta(HcclQpMeta &meta);
  HcclResult Accept(const HcclQpMeta &remoteMeta, OneSidedConn &conn);
  HcclResult Connect(const HcclQpMeta &remoteMeta, OneSidedConn &conn);

  // Tear down a single connection: drop its ConnState (which releases the
  // transport + per-connection NotifyPool) so a subsequent handshake builds
  // a fresh transport with notify values reset to 0. Idempotent: closing an
  // unknown/already-closed conn returns HCCL_SUCCESS. The caller MUST have
  // quiesced any stream that issued ops on this connection (e.g. via
  // aclrtSynchronizeStream / torch stream synchronize) before calling, so no
  // in-flight stream op references the transport being destroyed.
  HcclResult CloseConnection(OneSidedConn conn);

  // One-sided read: pull a batch of slices from the remote staging region into
  // the local staging region. localMems[i].addr and remoteMems[i].addr are
  // byte OFFSETS into the respective staging regions (not absolute pointers);
  // localMems[i].size == remoteMems[i].size is the slice length. key is unused.
  HcclResult BatchRead(OneSidedConn conn,
                       const std::vector<MemDetails> &localMems,
                       const std::vector<MemDetails> &remoteMems,
                       aclrtStream stream);

  // One-sided write: push a batch of slices from the local staging region into
  // the remote staging region. localMems[i].addr and remoteMems[i].addr are
  // byte OFFSETS into the respective staging regions (not absolute pointers);
  // localMems[i].size == remoteMems[i].size is the slice length. key is unused.
  HcclResult BatchWrite(OneSidedConn conn,
                        const std::vector<MemDetails> &localMems,
                        const std::vector<MemDetails> &remoteMems,
                        aclrtStream stream);

  // Notify Post/Wait. notifyIdx is an absolute index in [0, numNotifies);
  // use OsNotifyIndex(slot, OS_NOTIFY_DATA_READY|OS_NOTIFY_CONSUMED) to map a
  // staging slot to its pair.
  HcclResult Post(OneSidedConn conn, uint32_t notifyIdx, aclrtStream stream);
  HcclResult Wait(OneSidedConn conn, uint32_t notifyIdx, aclrtStream stream);

  void *GetStagingBase() const;
  uint64_t GetStagingBytes() const;
  uint64_t GetRemoteStagingBytes(OneSidedConn conn);
  uint32_t GetNumNotifies() const;

private:
  explicit OneSidedAgent(uint32_t devId)
      : devId_(devId), state_(State::CREATED) {}

  struct ConnState {
    ~ConnState();

    std::string tag;
    std::shared_ptr<hccl::HcclSocket> ctrlSocket;
    std::unique_ptr<hccl::NotifyPool> notifyPool;
    std::shared_ptr<hccl::Transport> transport;
    hccl::DeviceMem inputMem;
    hccl::DeviceMem outputMem;
    uintptr_t remoteStagingBase{0};
    uint64_t remoteStagingBytes{0};
  };

  HcclResult AllocStaging();
  void FreeStaging();
  // Fills the identity fields (devId / phyDevId / ipv4Addr) shared by the
  // client and server metas. Caller must hold agentMutex_.
  void FillCommonMeta(HcclQpMeta &meta) const;
  HcclResult BuildConnection(const HcclQpMeta &remoteMeta, bool isListener,
                             std::shared_ptr<hccl::HcclSocket> ctrlSocket,
                             OneSidedConn &conn);
  ConnState *LookupConn(OneSidedConn conn);

  enum class State { CREATED, INITIALIZED, ERROR };

  static std::shared_ptr<OneSidedAgent> instances[OS_MAX_LOCAL_DEVICES];
  static std::mutex instanceMutex;

  uint32_t devId_;
  State state_;
  OneSidedConfig cfg_{};
  uint32_t phyId_{0};
  hccl::HcclIpAddress localIp_;

  HcclNetDevCtx nicNetDevCtx_{nullptr};
  std::shared_ptr<hccl::HcclSocket> nicServerSocket_;
  HcclDispatcher dispatcher_{nullptr};

  void *stagingBase_{nullptr};
  uint64_t stagingBytes_{0};
  uint32_t notifyNum_{0};

  std::mutex agentMutex_;
  std::unordered_map<OneSidedConn, std::unique_ptr<ConnState>> conns_;
};
