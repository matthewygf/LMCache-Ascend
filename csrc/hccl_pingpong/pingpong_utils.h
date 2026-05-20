/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl_ip_address.h"

#define ACL_CHECK(answer)                                                      \
  do {                                                                         \
    aclError ret = (answer);                                                   \
    if (ret != ACL_SUCCESS) {                                                  \
      return HCCL_E_INTERNAL;                                                  \
    }                                                                          \
  } while (0)

#define HCCL_CHECK(answer)                                                     \
  do {                                                                         \
    HcclResult ret = (answer);                                                 \
    if (ret != HCCL_SUCCESS) {                                                 \
      return ret;                                                              \
    }                                                                          \
  } while (0)

void FillRandom(char *buffer, size_t length);

HcclResult GetLocalIpv4(uint32_t phyId, hccl::HcclIpAddress &localIp);
