/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif
#define _GLIBCXX_USE_CXX11_ABI 0

#include "pingpong_utils.h"

#include <random>
#include <vector>

#include "adapter_hccp_common.h"   // hrtRaGetDeviceIP

void FillRandom(char *buffer, size_t length)
{
    if (length == 0 || buffer == nullptr) {
        return;
    }

    const char *characters =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t characters_size = 62;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, characters_size - 1);

    for (size_t i = 0; i < length; ++i) {
        buffer[i] = characters[dis(gen)];
    }
    // length-terminator is the caller's responsibility; we don't write past length.
}

HcclResult GetLocalIpv4(uint32_t phyId, hccl::HcclIpAddress &localIp)
{
    std::vector<hccl::HcclIpAddress> ips;
    HCCL_CHECK(hrtRaGetDeviceIP(phyId, ips));
    for (const auto &ip : ips) {
        if (ip.GetFamily() == AF_INET) {
            localIp = ip;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}
