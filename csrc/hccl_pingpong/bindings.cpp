/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
// HCCL is built against the legacy libstdc++ ABI, see csrc/hccl/bindings.cpp.
#define _GLIBCXX_USE_CXX11_ABI 0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "pingpong_agent.h"

namespace py = pybind11;

namespace {

void check_hccl_result(HcclResult result, const std::string &msg)
{
    if (result != HCCL_SUCCESS) {
        throw std::runtime_error(msg + " | HCCL Error Code: " +
                                 std::to_string(result));
    }
}

}  // namespace

PYBIND11_MODULE(hccl_pingpong_npu_comms, m)
{
    m.doc() = "pybind11 wrapper for the HCCL ping-pong BatchChannel agent";

    // ------------------------------------------------------------------
    // PingPongConfig
    // ------------------------------------------------------------------
    py::class_<PingPongConfig>(m, "PingPongConfig")
        .def(py::init<>())
        .def_readwrite("chunk_size_bytes", &PingPongConfig::chunkSizeBytes)
        .def_readwrite("n_chunks_per_buff", &PingPongConfig::nChunksPerBuff)
        .def_readwrite("n_buffs", &PingPongConfig::nBuffs)
        .def_readwrite("wait_recv_done", &PingPongConfig::waitRecvDone)
        .def_readwrite("tc", &PingPongConfig::tc)
        .def_readwrite("sl", &PingPongConfig::sl)
        .def(py::pickle(
            [](const PingPongConfig &c) {
                return py::make_tuple(c.chunkSizeBytes, c.nChunksPerBuff,
                                      c.nBuffs, c.waitRecvDone, c.tc, c.sl);
            },
            [](py::tuple t) {
                if (t.size() != 6)
                    throw std::runtime_error("Invalid PingPongConfig pickle");
                PingPongConfig c;
                c.chunkSizeBytes = t[0].cast<uint64_t>();
                c.nChunksPerBuff = t[1].cast<uint32_t>();
                c.nBuffs = t[2].cast<uint32_t>();
                c.waitRecvDone = t[3].cast<bool>();
                c.tc = t[4].cast<uint32_t>();
                c.sl = t[5].cast<uint32_t>();
                return c;
            }));

    // ------------------------------------------------------------------
    // PingPongClientMeta / PingPongServerMeta
    // ------------------------------------------------------------------
    py::class_<PingPongClientMeta>(m, "PingPongClientMeta")
        .def(py::init<>())
        .def_readwrite("dev_id", &PingPongClientMeta::devId)
        .def_readwrite("phy_dev_id", &PingPongClientMeta::phyDevId)
        .def_readwrite("ipv4_addr", &PingPongClientMeta::ipv4Addr)
        .def(py::pickle(
            [](const PingPongClientMeta &c) {
                return py::make_tuple(c.devId, c.ipv4Addr, c.phyDevId);
            },
            [](py::tuple t) {
                PingPongClientMeta c;
                if (t.size() == 3) {
                    c.devId = t[0].cast<int32_t>();
                    c.ipv4Addr = t[1].cast<uint32_t>();
                    c.phyDevId = t[2].cast<int32_t>();
                } else if (t.size() == 2) {
                    // Legacy: dev_id carried physical id only.
                    c.devId = t[0].cast<int32_t>();
                    c.ipv4Addr = t[1].cast<uint32_t>();
                    c.phyDevId = c.devId;
                } else {
                    throw std::runtime_error(
                        "Invalid PingPongClientMeta pickle");
                }
                return c;
            }));

    py::class_<PingPongServerMeta>(m, "PingPongServerMeta")
        .def(py::init<>())
        .def_readwrite("dev_id", &PingPongServerMeta::devId)
        .def_readwrite("phy_dev_id", &PingPongServerMeta::phyDevId)
        .def_readwrite("ipv4_addr", &PingPongServerMeta::ipv4Addr)
        .def_readwrite("listen_port", &PingPongServerMeta::listenPort)
        .def_property(
            "tag_ctrl",
            [](const PingPongServerMeta &s) {
                return py::bytes(s.tagCtrl, PP_TAG_SIZE);
            },
            [](PingPongServerMeta &s, const py::bytes &b) {
                py::buffer_info info(py::buffer(b).request());
                if (static_cast<size_t>(info.size) != PP_TAG_SIZE)
                    throw std::runtime_error("tag_ctrl must be 32 bytes");
                memcpy(s.tagCtrl, info.ptr, PP_TAG_SIZE);
            })
        .def(py::pickle(
            [](const PingPongServerMeta &s) {
                return py::make_tuple(s.devId, s.ipv4Addr, s.listenPort,
                                      py::bytes(s.tagCtrl, PP_TAG_SIZE),
                                      s.phyDevId);
            },
            [](py::tuple t) {
                PingPongServerMeta s;
                if (t.size() == 5) {
                    s.devId = t[0].cast<int32_t>();
                    s.ipv4Addr = t[1].cast<uint32_t>();
                    s.listenPort = t[2].cast<uint32_t>();
                    py::buffer_info info(
                        py::buffer(t[3].cast<py::bytes>()).request());
                    if (static_cast<size_t>(info.size) != PP_TAG_SIZE)
                        throw std::runtime_error(
                            "Invalid tag_ctrl size in pickle state");
                    memcpy(s.tagCtrl, info.ptr, PP_TAG_SIZE);
                    s.phyDevId = t[4].cast<int32_t>();
                } else if (t.size() == 4) {
                    s.devId = t[0].cast<int32_t>();
                    s.ipv4Addr = t[1].cast<uint32_t>();
                    s.listenPort = t[2].cast<uint32_t>();
                    py::buffer_info info(
                        py::buffer(t[3].cast<py::bytes>()).request());
                    if (static_cast<size_t>(info.size) != PP_TAG_SIZE)
                        throw std::runtime_error(
                            "Invalid tag_ctrl size in pickle state");
                    memcpy(s.tagCtrl, info.ptr, PP_TAG_SIZE);
                    s.phyDevId = s.devId;
                } else {
                    throw std::runtime_error(
                        "Invalid PingPongServerMeta pickle");
                }
                return s;
            }));

    // ------------------------------------------------------------------
    // PingPongOp / PingPongScatterEntry
    // ------------------------------------------------------------------
    py::class_<PingPongOp>(m, "PingPongOp")
        .def(py::init([](uintptr_t local_addr, uint64_t size) {
                 return PingPongOp{reinterpret_cast<void *>(local_addr), size};
             }),
             py::arg("local_addr"), py::arg("size"))
        .def_property(
            "local_addr",
            [](const PingPongOp &op) {
                return reinterpret_cast<uintptr_t>(op.localAddr);
            },
            [](PingPongOp &op, uintptr_t a) {
                op.localAddr = reinterpret_cast<void *>(a);
            })
        .def_readwrite("size", &PingPongOp::size);

    py::class_<PingPongScatterEntry>(m, "PingPongScatterEntry")
        .def(py::init<>())
        .def_property(
            "ddr_buf",
            [](const PingPongScatterEntry &e) {
                return reinterpret_cast<uintptr_t>(e.ddrBuf);
            },
            [](PingPongScatterEntry &e, uintptr_t a) {
                e.ddrBuf = reinterpret_cast<void *>(a);
            })
        .def_property(
            "dst_bufs",
            [](const PingPongScatterEntry &e) {
                std::vector<uintptr_t> out;
                out.reserve(e.dstBufs.size());
                for (void *p : e.dstBufs) {
                    out.push_back(reinterpret_cast<uintptr_t>(p));
                }
                return out;
            },
            [](PingPongScatterEntry &e, const std::vector<uintptr_t> &v) {
                e.dstBufs.clear();
                e.dstBufs.reserve(v.size());
                for (uintptr_t a : v) {
                    e.dstBufs.push_back(reinterpret_cast<void *>(a));
                }
            })
        .def_readwrite("counts", &PingPongScatterEntry::counts)
        .def_readwrite("data_type", &PingPongScatterEntry::dataType);

    // ------------------------------------------------------------------
    // PingPongAgent
    // ------------------------------------------------------------------
    py::class_<PingPongAgent, std::shared_ptr<PingPongAgent>>(m, "PingPongAgent")
        .def_static(
            "get_instance",
            [](uint32_t deviceId) {
                std::shared_ptr<PingPongAgent> agent;
                check_hccl_result(
                    PingPongAgent::GetInstance(deviceId, agent),
                    "Failed to get PingPongAgent instance");
                return agent;
            },
            py::arg("device_id"))
        .def(
            "init",
            [](PingPongAgent &self, const PingPongConfig &cfg) {
                check_hccl_result(self.Init(cfg), "PingPongAgent Init failed");
            },
            py::arg("config") = PingPongConfig{})
        .def("get_client_meta",
             [](PingPongAgent &self) {
                 PingPongClientMeta meta{};
                 check_hccl_result(self.GetClientMeta(meta),
                                   "Failed to get client meta");
                 return meta;
             })
        .def("get_server_meta",
             [](PingPongAgent &self) {
                 PingPongServerMeta meta{};
                 check_hccl_result(self.GetServerMeta(meta),
                                   "Failed to get server meta");
                 return meta;
             })
        .def(
            "accept",
            [](PingPongAgent &self, const PingPongClientMeta &client,
               const PingPongServerMeta &server) {
                py::gil_scoped_release release;
                PingPongConn conn = nullptr;
                check_hccl_result(self.Accept(client, server, conn),
                                  "Accept failed");
                return reinterpret_cast<uintptr_t>(conn);
            },
            py::arg("client_meta"), py::arg("server_meta"))
        .def(
            "connect",
            [](PingPongAgent &self, const PingPongServerMeta &server) {
                py::gil_scoped_release release;
                PingPongConn conn = nullptr;
                check_hccl_result(self.Connect(server, conn),
                                  "Connect failed");
                return reinterpret_cast<uintptr_t>(conn);
            },
            py::arg("server_meta"))
        .def(
            "send_batch",
            [](PingPongAgent &self, uintptr_t conn,
               const std::vector<PingPongOp> &ops, uintptr_t stream) {
                py::gil_scoped_release release;
                check_hccl_result(
                    self.SendBatch(reinterpret_cast<PingPongConn>(conn), ops,
                                   reinterpret_cast<aclrtStream>(stream)),
                    "send_batch failed");
            },
            py::arg("conn"), py::arg("ops"), py::arg("stream"))
        .def(
            "recv_batch",
            [](PingPongAgent &self, uintptr_t conn,
               const std::vector<PingPongOp> &ops, uintptr_t stream) {
                py::gil_scoped_release release;
                check_hccl_result(
                    self.RecvBatch(reinterpret_cast<PingPongConn>(conn), ops,
                                   reinterpret_cast<aclrtStream>(stream)),
                    "recv_batch failed");
            },
            py::arg("conn"), py::arg("ops"), py::arg("stream"))
        .def(
            "scatter_send",
            [](PingPongAgent &self, uintptr_t conn,
               std::vector<PingPongScatterEntry> entries, uintptr_t stream) {
                py::gil_scoped_release release;
                check_hccl_result(
                    self.ScatterSend(reinterpret_cast<PingPongConn>(conn),
                                     entries,
                                     reinterpret_cast<aclrtStream>(stream)),
                    "scatter_send failed");
            },
            py::arg("conn"), py::arg("entries"), py::arg("stream"))
        .def(
            "scatter_recv",
            [](PingPongAgent &self, uintptr_t conn,
               std::vector<PingPongScatterEntry> entries, uintptr_t stream) {
                py::gil_scoped_release release;
                check_hccl_result(
                    self.ScatterRecv(reinterpret_cast<PingPongConn>(conn),
                                     entries,
                                     reinterpret_cast<aclrtStream>(stream)),
                    "scatter_recv failed");
            },
            py::arg("conn"), py::arg("entries"), py::arg("stream"))
        .def("shared_region_bytes", &PingPongAgent::SharedRegionBytes);

    // Expose HcclDataType so Python can specify scatter element types without
    // needing torch / numpy.
    py::enum_<HcclDataType>(m, "HcclDataType")
        .value("HCCL_DATA_TYPE_INT8", HCCL_DATA_TYPE_INT8)
        .value("HCCL_DATA_TYPE_INT16", HCCL_DATA_TYPE_INT16)
        .value("HCCL_DATA_TYPE_INT32", HCCL_DATA_TYPE_INT32)
        .value("HCCL_DATA_TYPE_FP16", HCCL_DATA_TYPE_FP16)
        .value("HCCL_DATA_TYPE_FP32", HCCL_DATA_TYPE_FP32)
        .value("HCCL_DATA_TYPE_INT64", HCCL_DATA_TYPE_INT64)
        .value("HCCL_DATA_TYPE_UINT64", HCCL_DATA_TYPE_UINT64)
        .value("HCCL_DATA_TYPE_UINT8", HCCL_DATA_TYPE_UINT8)
        .value("HCCL_DATA_TYPE_UINT16", HCCL_DATA_TYPE_UINT16)
        .value("HCCL_DATA_TYPE_UINT32", HCCL_DATA_TYPE_UINT32)
        .value("HCCL_DATA_TYPE_FP64", HCCL_DATA_TYPE_FP64)
        .value("HCCL_DATA_TYPE_BFP16", HCCL_DATA_TYPE_BFP16)
        .export_values();
}
