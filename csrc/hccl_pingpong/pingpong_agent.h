/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PingPongAgent: singleton-per-device facade for the BatchChannel data plane.
 *
 * Architecture (mirrors HcclAgent):
 *  - One agent per logical device id, returned by GetInstance().
 *  - Init() opens the device NIC, creates the listen socket, starts the dispatcher,
 *    and allocates the device-shared input/output ping-pong regions.
 *  - GetServerMeta() returns (ipv4, listenPort, random tagCtrl). The tagCtrl doubles
 *    as the unique tag for the underlying BatchChannel notifyPool / MachinePara.
 *  - Accept(client, server) and Connect(server) use the SAME shared listen socket
 *    (per-peer whitelisted by tagCtrl) to establish the BatchChannel ctrl socket and
 *    construct an owning BatchChannel. The returned conn handle is the BatchChannel
 *    raw pointer; the agent keeps it alive in conns_.
 *  - SendBatch / RecvBatch / ScatterSend / ScatterRecv look up the BatchChannel by
 *    conn handle and forward the call.
 *
 * Concurrency model: external callers (the Python channel) serialize read/scatter
 * dispatch using a single channel-wide lock so multiple peers cannot race over the
 * shared input/output regions. The agent does NOT serialize transfers internally.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "batch_channel.h"
#include "dispatcher.h"
#include "hccl_ip_address.h"
#include "hccl_network_pub.h"
#include "hccl_socket.h"
#include "notify_pool.h"

constexpr uint32_t PP_MAX_LOCAL_DEVICES = 16;
constexpr uint32_t PP_TAG_SIZE = 32;

struct PingPongConfig {
    uint64_t chunkSizeBytes = 2ULL * 1024 * 1024;
    uint32_t nChunksPerBuff = 4;
    uint32_t nBuffs = 2;
    bool waitRecvDone = true;
    uint32_t tc = 132;
    uint32_t sl = 4;
};

struct PingPongClientMeta {
    /// Torch logical device index (same as PingPongAgent devId_).
    int32_t devId{0};
    /// Ascend physical device id — required for MachinePara::remoteDeviceId / localDeviceId.
    int32_t phyDevId{0};
    uint32_t ipv4Addr{0};
};

struct PingPongServerMeta {
    int32_t devId{0};
    int32_t phyDevId{0};
    uint32_t ipv4Addr{0};
    uint32_t listenPort{0};
    char tagCtrl[PP_TAG_SIZE];   // random per-peer; also reused as BatchChannel tag
};

// One direction-agnostic descriptor: localAddr is the local endpoint pointer
// (sender uses it as src; receiver uses it as dst).
struct PingPongOp {
    void *localAddr{nullptr};
    uint64_t size{0};
};

// Mirrors P2pScatterEntry but uses std::vector for pybind ergonomics.
struct PingPongScatterEntry {
    void *ddrBuf{nullptr};                   // sender: host DDR base (numEl slices laid out tight). receiver: nullptr.
    std::vector<void *> dstBufs;             // sender: empty. receiver: numEl device pointers.
    std::vector<uint64_t> counts;            // numEl element counts (both sides).
    HcclDataType dataType{HCCL_DATA_TYPE_FP32};
};

typedef void *PingPongConn;

class PingPongAgent {
public:
    static HcclResult GetInstance(uint32_t deviceId,
                                  std::shared_ptr<PingPongAgent> &outAgent);

    PingPongAgent(const PingPongAgent &) = delete;
    PingPongAgent &operator=(const PingPongAgent &) = delete;
    PingPongAgent(PingPongAgent &&) = delete;
    PingPongAgent &operator=(PingPongAgent &&) = delete;

    ~PingPongAgent();

    HcclResult Init(const PingPongConfig &cfg);

    HcclResult GetClientMeta(PingPongClientMeta &meta);
    HcclResult GetServerMeta(PingPongServerMeta &meta);

    HcclResult Accept(const PingPongClientMeta &client,
                      const PingPongServerMeta &server, PingPongConn &conn);
    HcclResult Connect(const PingPongServerMeta &server, PingPongConn &conn);

    HcclResult SendBatch(PingPongConn conn, const std::vector<PingPongOp> &ops,
                         aclrtStream stream);
    HcclResult RecvBatch(PingPongConn conn, const std::vector<PingPongOp> &ops,
                         aclrtStream stream);

    HcclResult ScatterSend(PingPongConn conn,
                           std::vector<PingPongScatterEntry> &entries,
                           aclrtStream stream);
    HcclResult ScatterRecv(PingPongConn conn,
                           std::vector<PingPongScatterEntry> &entries,
                           aclrtStream stream);

    // Per-direction shared region byte size. Useful for the Python side so it can
    // tell BatchChannel-bound staging regions apart from user buffers when debug-
    // dumping memory layouts.
    uint64_t SharedRegionBytes() const;

private:
    explicit PingPongAgent(uint32_t devId)
        : devId_(devId), state_(State::CREATED) {}

    HcclResult AllocSharedRegions();
    void FreeSharedRegions();
    BatchChannel *LookupConn(PingPongConn conn);

    enum class State { CREATED, INITIALIZED, ERROR };

    static std::shared_ptr<PingPongAgent> instances[PP_MAX_LOCAL_DEVICES];
    static std::mutex instanceMutex;

    uint32_t devId_;
    State state_;
    PingPongConfig cfg_{};
    uint32_t phyId_{0};
    hccl::HcclIpAddress localIp_;

    HcclNetDevCtx nicNetDevCtx_{nullptr};
    std::shared_ptr<hccl::HcclSocket> nicServerSocket_;
    HcclDispatcher dispatcher_{nullptr};

    // Device-shared ping-pong regions. Allocated once on Init, reused by every
    // BatchChannel created by this agent.
    void *inputBase_{nullptr};
    void *outputBase_{nullptr};
    uint64_t sharedRegionBytes_{0};

    std::mutex agentMutex_;

    std::unordered_map<PingPongConn, std::unique_ptr<BatchChannel>> conns_;
};
