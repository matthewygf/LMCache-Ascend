/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "onesided_agent.h"

namespace py = pybind11;

namespace {

void check_hccl_result(HcclResult result, const std::string &msg) {
  if (result != HCCL_SUCCESS) {
    throw std::runtime_error(msg +
                             " | HCCL Error Code: " + std::to_string(result));
  }
}

} // namespace

PYBIND11_MODULE(hccl_onesided_npu_comms, m) {
  m.doc() = "pybind11 wrapper for HCCL one-sided staging channel";

  py::class_<OneSidedConfig>(m, "OneSidedConfig")
      .def(py::init<>())
      .def_readwrite("staging_bytes", &OneSidedConfig::stagingBytes)
      .def_readwrite("num_slots", &OneSidedConfig::numSlots)
      .def_readwrite("tc", &OneSidedConfig::tc)
      .def_readwrite("sl", &OneSidedConfig::sl)
      .def_readwrite("timeout_sec", &OneSidedConfig::timeoutSec)
      .def(py::pickle(
          [](const OneSidedConfig &c) {
            return py::make_tuple(c.stagingBytes, c.numSlots, c.tc, c.sl,
                                  c.timeoutSec);
          },
          [](py::tuple t) {
            if (t.size() != 5)
              throw std::runtime_error("Invalid OneSidedConfig pickle");
            OneSidedConfig c;
            c.stagingBytes = t[0].cast<uint64_t>();
            c.numSlots = t[1].cast<uint32_t>();
            c.tc = t[2].cast<uint32_t>();
            c.sl = t[3].cast<uint32_t>();
            c.timeoutSec = t[4].cast<int32_t>();
            return c;
          }));

  py::class_<HcclQpMeta>(m, "HcclQpMeta")
      .def(py::init<>())
      .def_readwrite("dev_id", &HcclQpMeta::devId)
      .def_readwrite("phy_dev_id", &HcclQpMeta::phyDevId)
      .def_readwrite("ipv4_addr", &HcclQpMeta::ipv4Addr)
      .def_readwrite("listen_port", &HcclQpMeta::listenPort)
      .def_property(
          "tag_ctrl",
          [](const HcclQpMeta &s) { return py::bytes(s.tagCtrl, OS_TAG_SIZE); },
          [](HcclQpMeta &s, const py::bytes &b) {
            py::buffer_info info(py::buffer(b).request());
            if (static_cast<size_t>(info.size) != OS_TAG_SIZE)
              throw std::runtime_error("tag_ctrl must be 32 bytes");
            std::memcpy(s.tagCtrl, info.ptr, OS_TAG_SIZE);
          })
      .def(py::pickle(
          [](const HcclQpMeta &s) {
            return py::make_tuple(s.devId, s.phyDevId, s.ipv4Addr, s.listenPort,
                                  py::bytes(s.tagCtrl, OS_TAG_SIZE));
          },
          [](py::tuple t) {
            if (t.size() != 5)
              throw std::runtime_error("Invalid HcclQpMeta pickle");
            HcclQpMeta s;
            s.devId = t[0].cast<int32_t>();
            s.phyDevId = t[1].cast<int32_t>();
            s.ipv4Addr = t[2].cast<uint32_t>();
            s.listenPort = t[3].cast<uint32_t>();
            py::buffer_info info(py::buffer(t[4].cast<py::bytes>()).request());
            if (static_cast<size_t>(info.size) != OS_TAG_SIZE)
              throw std::runtime_error("Invalid tag_ctrl size in pickle state");
            std::memcpy(s.tagCtrl, info.ptr, OS_TAG_SIZE);
            return s;
          }));

  // For OneSidedAgent.batch_read, `addr` is a byte OFFSET into the staging
  // region (local for local_mems, remote for remote_mems), NOT an absolute
  // device pointer. `key` is unused by the one-sided read path.
  py::class_<MemDetails>(m, "MemDetails")
      .def(py::init([](uint64_t addr, uint64_t size, uint32_t key) {
             MemDetails md;
             md.addr = addr;
             md.size = size;
             md.key = key;
             return md;
           }),
           py::arg("addr"), py::arg("size"), py::arg("key") = 0)
      .def_readwrite("addr", &MemDetails::addr)
      .def_readwrite("size", &MemDetails::size)
      .def_readwrite("key", &MemDetails::key);

  py::class_<OneSidedAgent, std::shared_ptr<OneSidedAgent>>(m, "OneSidedAgent")
      .def_static(
          "get_instance",
          [](uint32_t deviceId) {
            std::shared_ptr<OneSidedAgent> agent;
            check_hccl_result(OneSidedAgent::GetInstance(deviceId, agent),
                              "Failed to get OneSidedAgent instance");
            return agent;
          },
          py::arg("device_id"))
      .def(
          "init",
          [](OneSidedAgent &self, const OneSidedConfig &cfg) {
            check_hccl_result(self.Init(cfg), "OneSidedAgent Init failed");
          },
          py::arg("config") = OneSidedConfig{})
      .def("get_client_meta",
           [](OneSidedAgent &self) {
             HcclQpMeta meta{};
             check_hccl_result(self.GetClientMeta(meta),
                               "Failed to get client meta");
             return meta;
           })
      .def("get_server_meta",
           [](OneSidedAgent &self) {
             HcclQpMeta meta{};
             check_hccl_result(self.GetServerMeta(meta),
                               "Failed to get server meta");
             return meta;
           })
      .def(
          "accept",
          [](OneSidedAgent &self, const HcclQpMeta &remoteMeta) {
            py::gil_scoped_release release;
            OneSidedConn conn = nullptr;
            check_hccl_result(self.Accept(remoteMeta, conn), "Accept failed");
            return reinterpret_cast<uintptr_t>(conn);
          },
          py::arg("remote_meta"))
      .def(
          "connect",
          [](OneSidedAgent &self, const HcclQpMeta &remoteMeta) {
            py::gil_scoped_release release;
            OneSidedConn conn = nullptr;
            check_hccl_result(self.Connect(remoteMeta, conn), "Connect failed");
            return reinterpret_cast<uintptr_t>(conn);
          },
          py::arg("remote_meta"))
      .def(
          "close_connection",
          [](OneSidedAgent &self, uintptr_t conn) {
            py::gil_scoped_release release;
            check_hccl_result(
                self.CloseConnection(reinterpret_cast<OneSidedConn>(conn)),
                "close_connection failed");
          },
          py::arg("conn"))
      .def(
          "batch_read",
          [](OneSidedAgent &self, uintptr_t conn,
             const std::vector<MemDetails> &localMems,
             const std::vector<MemDetails> &remoteMems, uintptr_t stream) {
            py::gil_scoped_release release;
            check_hccl_result(
                self.BatchRead(reinterpret_cast<OneSidedConn>(conn), localMems,
                               remoteMems,
                               reinterpret_cast<aclrtStream>(stream)),
                "batch_read failed");
          },
          py::arg("conn"), py::arg("local_mems"), py::arg("remote_mems"),
          py::arg("stream"))
      .def(
          "batch_write",
          [](OneSidedAgent &self, uintptr_t conn,
             const std::vector<MemDetails> &localMems,
             const std::vector<MemDetails> &remoteMems, uintptr_t stream) {
            py::gil_scoped_release release;
            check_hccl_result(
                self.BatchWrite(reinterpret_cast<OneSidedConn>(conn), localMems,
                                remoteMems,
                                reinterpret_cast<aclrtStream>(stream)),
                "batch_write failed");
          },
          py::arg("conn"), py::arg("local_mems"), py::arg("remote_mems"),
          py::arg("stream"))
      .def(
          "post",
          [](OneSidedAgent &self, uintptr_t conn, uint32_t notifyIdx,
             uintptr_t stream) {
            py::gil_scoped_release release;
            check_hccl_result(self.Post(reinterpret_cast<OneSidedConn>(conn),
                                        notifyIdx,
                                        reinterpret_cast<aclrtStream>(stream)),
                              "post failed");
          },
          py::arg("conn"), py::arg("notify_idx"), py::arg("stream"))
      .def(
          "wait",
          [](OneSidedAgent &self, uintptr_t conn, uint32_t notifyIdx,
             uintptr_t stream) {
            py::gil_scoped_release release;
            check_hccl_result(self.Wait(reinterpret_cast<OneSidedConn>(conn),
                                        notifyIdx,
                                        reinterpret_cast<aclrtStream>(stream)),
                              "wait failed");
          },
          py::arg("conn"), py::arg("notify_idx"), py::arg("stream"))
      .def("get_staging_base",
           [](OneSidedAgent &self) {
             return reinterpret_cast<uintptr_t>(self.GetStagingBase());
           })
      .def("get_staging_bytes", &OneSidedAgent::GetStagingBytes)
      .def("get_num_notifies", &OneSidedAgent::GetNumNotifies)
      .def(
          "get_remote_staging_bytes",
          [](OneSidedAgent &self, uintptr_t conn) {
            return self.GetRemoteStagingBytes(
                reinterpret_cast<OneSidedConn>(conn));
          },
          py::arg("conn"));

  // Per-slot notify layout helpers (mirror OsNotifyIndex in the header):
  // slot s owns data_ready = 2*s and consumed = 2*s + 1.
  m.attr("NOTIFIES_PER_SLOT") = OS_NOTIFIES_PER_SLOT;
  m.attr("NOTIFY_DATA_READY") = OS_NOTIFY_DATA_READY;
  m.attr("NOTIFY_CONSUMED") = OS_NOTIFY_CONSUMED;
  m.def(
      "data_ready_notify",
      [](uint32_t slot) { return OsNotifyIndex(slot, OS_NOTIFY_DATA_READY); },
      py::arg("slot"),
      "Absolute notify index of slot's data_ready notify (2*slot).");
  m.def(
      "consumed_notify",
      [](uint32_t slot) { return OsNotifyIndex(slot, OS_NOTIFY_CONSUMED); },
      py::arg("slot"),
      "Absolute notify index of slot's consumed notify (2*slot + 1).");

  m.def(
      "acl_memcpy_async_h2d",
      [](uintptr_t dst, uintptr_t src, uint64_t size, uintptr_t stream) {
        auto ret = aclrtMemcpyAsync(reinterpret_cast<void *>(dst), size,
                                    reinterpret_cast<const void *>(src), size,
                                    ACL_MEMCPY_HOST_TO_DEVICE,
                                    reinterpret_cast<aclrtStream>(stream));
        if (ret != ACL_SUCCESS) {
          throw std::runtime_error("aclrtMemcpyAsync H2D failed | ACL "
                                   "Error Code: " +
                                   std::to_string(ret));
        }
      },
      py::arg("dst"), py::arg("src"), py::arg("size"), py::arg("stream"));

  m.def(
      "acl_memcpy_async_d2h",
      [](uintptr_t dst, uintptr_t src, uint64_t size, uintptr_t stream) {
        auto ret = aclrtMemcpyAsync(reinterpret_cast<void *>(dst), size,
                                    reinterpret_cast<const void *>(src), size,
                                    ACL_MEMCPY_DEVICE_TO_HOST,
                                    reinterpret_cast<aclrtStream>(stream));
        if (ret != ACL_SUCCESS) {
          throw std::runtime_error("aclrtMemcpyAsync D2H failed | ACL "
                                   "Error Code: " +
                                   std::to_string(ret));
        }
      },
      py::arg("dst"), py::arg("src"), py::arg("size"), py::arg("stream"));

  m.def(
      "acl_memcpy_async_d2d",
      [](uintptr_t dst, uintptr_t src, uint64_t size, uintptr_t stream) {
        auto ret = aclrtMemcpyAsync(reinterpret_cast<void *>(dst), size,
                                    reinterpret_cast<const void *>(src), size,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                                    reinterpret_cast<aclrtStream>(stream));
        if (ret != ACL_SUCCESS) {
          throw std::runtime_error("aclrtMemcpyAsync D2D failed | ACL "
                                   "Error Code: " +
                                   std::to_string(ret));
        }
      },
      py::arg("dst"), py::arg("src"), py::arg("size"), py::arg("stream"));
}
