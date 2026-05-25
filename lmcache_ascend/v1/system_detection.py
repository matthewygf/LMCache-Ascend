# SPDX-License-Identifier: Apache-2.0
# Standard
import os
from typing import Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.system_detection import NUMAMapping
from vllm.platforms import current_platform
import torch

logger = init_logger(__name__)

if torch.npu.is_available():
    try:
        # First Party
        from lmcache_ascend.c_ops import get_gpu_pci_bus_id
    except ImportError:
        # Fallback if c_ops is not available
        get_gpu_pci_bus_id = None


def _parse_first_cpu(cpulist_str: str) -> int:
    """Parse the first CPU number from a cpulist string like '0-23,48-71'."""
    return int(cpulist_str.replace("-", ",").split(",")[0])


def _get_socket_numa_mask(numa_node: int) -> int:
    """
    Find all NUMA nodes on the same physical socket as the given NUMA node,
    and return a bitmask with bits set for each sibling NUMA node.

    On A2 servers, each socket typically has 2 NUMA nodes. This function
    discovers both, allowing memory allocations to use the full DRAM capacity
    of the socket.

    Falls back to a single-node mask (1 << numa_node) if the socket topology
    cannot be determined.
    """
    try:
        # Read the CPU list for the given NUMA node
        cpulist_file = (
            f"/sys/devices/system/node/node{numa_node}/cpulist"
        )
        with open(cpulist_file) as f:
            cpulist_str = f.read().strip()

        if not cpulist_str:
            return 1 << numa_node

        first_cpu = _parse_first_cpu(cpulist_str)

        # Get the physical socket (package) ID for this CPU
        pkg_file = (
            f"/sys/devices/system/cpu/cpu{first_cpu}"
            f"/topology/physical_package_id"
        )
        with open(pkg_file) as f:
            socket_id = int(f.read().strip())

        # Enumerate all NUMA nodes and find those on the same socket
        mask = 0
        node_base = "/sys/devices/system/node"
        for entry in sorted(os.listdir(node_base)):
            if not entry.startswith("node") or not entry[4:].isdigit():
                continue
            node_id = int(entry[4:])

            node_cpulist_file = f"{node_base}/{entry}/cpulist"
            try:
                with open(node_cpulist_file) as f:
                    node_cpulist = f.read().strip()
            except OSError:
                continue

            if not node_cpulist:
                continue

            node_first_cpu = _parse_first_cpu(node_cpulist)
            node_pkg_file = (
                f"/sys/devices/system/cpu/cpu{node_first_cpu}"
                f"/topology/physical_package_id"
            )
            try:
                with open(node_pkg_file) as f:
                    node_socket_id = int(f.read().strip())
            except OSError:
                continue

            if node_socket_id == socket_id:
                mask |= 1 << node_id

        if mask == 0:
            return 1 << numa_node
        return mask

    except Exception as e:
        logger.warning(
            f"Failed to detect sibling NUMA nodes for node {numa_node}: "
            f"{e}. Falling back to single NUMA node."
        )
        return 1 << numa_node


def _read_from_sys() -> Optional[NUMAMapping]:
    """
    Read NUMA mapping from system configuration.
    Returns a NUMAMapping where the value is a bitmask of NUMA nodes
    on the same physical socket as the device, enabling memory allocations
    across all local NUMA nodes.
    """

    try:
        device_index = torch.npu.current_device()
        phy_device_id = current_platform.device_id_to_physical_device_id(device_index)
        pci_bus_id = get_gpu_pci_bus_id(phy_device_id).lower()

        numa_node_file = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
        with open(numa_node_file) as f:
            numa_node = int(f.read())

        # Sanitizing the output as on some hardware setups the numa_node variable
        # appeared to return with -1 value, causing failure.
        if numa_node >= 0:
            numa_mask = _get_socket_numa_mask(numa_node)
            return NUMAMapping(gpu_to_numa_mapping={device_index: numa_mask})
        else:
            logger.warning("No valid NUMA mapping for current device, returning None")
            return None
    except Exception as e:
        logger.warning(f"Failed to auto read NUMA mapping from system: {e}")
        return None
