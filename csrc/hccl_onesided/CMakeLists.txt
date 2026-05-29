find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

if(NOT DEFINED HCOMM_PUB_INC_DIR AND DEFINED ENV{LMC_HCOMM_PUB_INC_DIR})
    set(HCOMM_PUB_INC_DIR "$ENV{LMC_HCOMM_PUB_INC_DIR}")
endif()
if(NOT DEFINED HCOMM_PKG_INC_DIR AND DEFINED ENV{LMC_HCOMM_PKG_INC_DIR})
    set(HCOMM_PKG_INC_DIR "$ENV{LMC_HCOMM_PKG_INC_DIR}")
endif()

if(NOT HCOMM_PUB_INC_DIR)
    set(HCOMM_PUB_INC_DIR "${CMAKE_SOURCE_DIR}/../test/hcomm/src/pub_inc")
endif()
if(NOT HCOMM_PKG_INC_DIR)
    set(HCOMM_PKG_INC_DIR "${CMAKE_SOURCE_DIR}/../test/hcomm/pkg_inc")
endif()

get_filename_component(HCOMM_PUB_INC_DIR "${HCOMM_PUB_INC_DIR}" ABSOLUTE)
get_filename_component(HCOMM_PKG_INC_DIR "${HCOMM_PKG_INC_DIR}" ABSOLUTE)

if(NOT EXISTS "${HCOMM_PUB_INC_DIR}/transport_pub.h")
    message(WARNING
        "hccl_onesided: transport_pub.h not found under HCOMM_PUB_INC_DIR='${HCOMM_PUB_INC_DIR}'.\n"
        "Set -DHCOMM_PUB_INC_DIR=/abs/path/to/hcomm/src/pub_inc and\n"
        "    -DHCOMM_PKG_INC_DIR=/abs/path/to/hcomm/pkg_inc.\n"
        "Skipping hccl_onesided build.")
    return()
endif()

set(HCCL_ONESIDED_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../hccl_pingpong/pingpong_utils.cpp
    onesided_agent.cpp
    bindings.cpp
)

pybind11_add_module(hccl_onesided_npu_comms ${HCCL_ONESIDED_SRC_FILES})

if(NOT DEFINED ARCH_SUBDIR)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ARCH_SUBDIR "aarch64-linux")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ARCH_SUBDIR "x86_64-linux")
    else()
        message(FATAL_ERROR "hccl_onesided: unsupported arch ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()
set(_OS_ARCH_SUBDIR "${ARCH_SUBDIR}")

target_include_directories(hccl_onesided_npu_comms PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../hccl_pingpong
    ${HCOMM_PUB_INC_DIR}
    ${HCOMM_PUB_INC_DIR}/inner
    ${HCOMM_PKG_INC_DIR}
    ${HCOMM_PKG_INC_DIR}/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util/error_manager
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include/experiment/msprof
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include/experiment/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/${_OS_ARCH_SUBDIR}/include/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/toolchain
)

target_link_directories(hccl_onesided_npu_comms PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

find_library(HCOMM_FOR_ONESIDED_LIB hcomm
  PATHS ${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
  NO_DEFAULT_PATH)

target_link_libraries(hccl_onesided_npu_comms PRIVATE
    -Wl,--no-as-needed
    hccl_plf
    ascendcl
    hccl
)
if(HCOMM_FOR_ONESIDED_LIB)
  target_link_libraries(hccl_onesided_npu_comms PRIVATE ${HCOMM_FOR_ONESIDED_LIB})
endif()
target_link_libraries(hccl_onesided_npu_comms PRIVATE
    -Wl,--as-needed
    runtime
    pthread
)

target_link_options(hccl_onesided_npu_comms PRIVATE
    "LINKER:-rpath,${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64"
    "LINKER:-rpath,${ASCEND_CANN_PACKAGE_PATH}/lib64"
)

set_target_properties(hccl_onesided_npu_comms PROPERTIES
    BUILD_RPATH "${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64;${ASCEND_CANN_PACKAGE_PATH}/lib64"
    INSTALL_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/lib64;${ASCEND_CANN_PACKAGE_PATH}/lib64"
)

target_compile_definitions(hccl_onesided_npu_comms PRIVATE
    OPEN_BUILD_PROJECT
)

install(TARGETS hccl_onesided_npu_comms DESTINATION ${LMC_INSTALL_PATH})
