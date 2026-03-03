# SPDX-License-Identifier: Apache-2.0
"""
Tests for NUMA socket detection logic in system_detection.py.
Uses mocked sysfs directory structures to simulate different NUMA topologies.
"""

# Standard
import os
import tempfile


def _create_mock_sysfs(tmpdir, topology):
    """
    Create a mock sysfs directory structure.

    Args:
        tmpdir: Base temporary directory path.
        topology: Dict mapping (socket_id, numa_node_id) -> cpulist_str.
            Example: {(0, 0): "0-23", (0, 1): "24-47", (1, 2): "48-71"}
    """
    node_base = os.path.join(tmpdir, "sys", "devices", "system", "node")
    cpu_base = os.path.join(tmpdir, "sys", "devices", "system", "cpu")

    for (socket_id, node_id), cpulist_str in topology.items():
        # Create node directory with cpulist
        node_dir = os.path.join(node_base, f"node{node_id}")
        os.makedirs(node_dir, exist_ok=True)
        with open(os.path.join(node_dir, "cpulist"), "w") as f:
            f.write(cpulist_str)

        # Parse cpulist to create CPU topology entries
        for part in cpulist_str.split(","):
            if "-" in part:
                start, end = part.split("-")
                cpus = range(int(start), int(end) + 1)
            else:
                cpus = [int(part)]

            for cpu_id in cpus:
                topo_dir = os.path.join(
                    cpu_base, f"cpu{cpu_id}", "topology"
                )
                os.makedirs(topo_dir, exist_ok=True)
                with open(
                    os.path.join(topo_dir, "physical_package_id"), "w"
                ) as f:
                    f.write(str(socket_id))

    return node_base


def _get_socket_numa_mask_with_base(numa_node, node_base):
    """
    Reimplementation of _get_socket_numa_mask that accepts a base path
    for testing purposes. The logic mirrors the actual implementation.
    """
    try:
        cpulist_file = os.path.join(
            node_base, f"node{numa_node}", "cpulist"
        )
        with open(cpulist_file) as f:
            cpulist_str = f.read().strip()

        if not cpulist_str:
            return 1 << numa_node

        first_cpu = int(cpulist_str.replace("-", ",").split(",")[0])

        # Derive cpu_base from node_base
        # node_base = .../sys/devices/system/node
        # cpu_base  = .../sys/devices/system/cpu
        system_base = os.path.dirname(node_base)
        cpu_base = os.path.join(system_base, "cpu")

        pkg_file = os.path.join(
            cpu_base, f"cpu{first_cpu}", "topology", "physical_package_id"
        )
        with open(pkg_file) as f:
            socket_id = int(f.read().strip())

        mask = 0
        for entry in sorted(os.listdir(node_base)):
            if not entry.startswith("node") or not entry[4:].isdigit():
                continue
            node_id = int(entry[4:])

            node_cpulist_file = os.path.join(node_base, entry, "cpulist")
            try:
                with open(node_cpulist_file) as f:
                    node_cpulist = f.read().strip()
            except OSError:
                continue

            if not node_cpulist:
                continue

            node_first_cpu = int(
                node_cpulist.replace("-", ",").split(",")[0]
            )
            node_pkg_file = os.path.join(
                cpu_base,
                f"cpu{node_first_cpu}",
                "topology",
                "physical_package_id",
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

    except Exception:
        return 1 << numa_node


class TestGetSocketNumaMask:
    """Test the NUMA socket mask detection logic."""

    def test_single_numa_node_per_socket(self):
        """Single NUMA node per socket returns mask with only that node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topology = {
                (0, 0): "0-7",
                (1, 1): "8-15",
            }
            node_base = _create_mock_sysfs(tmpdir, topology)

            # Node 0 on socket 0 -> only node 0
            mask = _get_socket_numa_mask_with_base(0, node_base)
            assert mask == 0b01  # Only node 0

            # Node 1 on socket 1 -> only node 1
            mask = _get_socket_numa_mask_with_base(1, node_base)
            assert mask == 0b10  # Only node 1

    def test_two_numa_nodes_per_socket(self):
        """A2-like topology: 2 NUMA nodes per socket."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topology = {
                (0, 0): "0-23",
                (0, 1): "24-47",
                (1, 2): "48-71",
                (1, 3): "72-95",
            }
            node_base = _create_mock_sysfs(tmpdir, topology)

            # Node 0 on socket 0 -> nodes 0 and 1
            mask = _get_socket_numa_mask_with_base(0, node_base)
            assert mask == 0b0011  # Nodes 0 and 1

            # Node 1 on socket 0 -> nodes 0 and 1
            mask = _get_socket_numa_mask_with_base(1, node_base)
            assert mask == 0b0011  # Nodes 0 and 1

            # Node 2 on socket 1 -> nodes 2 and 3
            mask = _get_socket_numa_mask_with_base(2, node_base)
            assert mask == 0b1100  # Nodes 2 and 3

            # Node 3 on socket 1 -> nodes 2 and 3
            mask = _get_socket_numa_mask_with_base(3, node_base)
            assert mask == 0b1100  # Nodes 2 and 3

    def test_four_sockets_two_nodes_each(self):
        """Full A2 topology: 4 sockets x 2 NUMA nodes = 8 nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topology = {
                (0, 0): "0-23",
                (0, 1): "24-47",
                (1, 2): "48-71",
                (1, 3): "72-95",
                (2, 4): "96-119",
                (2, 5): "120-143",
                (3, 6): "144-167",
                (3, 7): "168-191",
            }
            node_base = _create_mock_sysfs(tmpdir, topology)

            # Socket 0: nodes 0,1
            assert _get_socket_numa_mask_with_base(0, node_base) == 0b00000011
            assert _get_socket_numa_mask_with_base(1, node_base) == 0b00000011

            # Socket 1: nodes 2,3
            assert _get_socket_numa_mask_with_base(2, node_base) == 0b00001100
            assert _get_socket_numa_mask_with_base(3, node_base) == 0b00001100

            # Socket 2: nodes 4,5
            assert _get_socket_numa_mask_with_base(4, node_base) == 0b00110000
            assert _get_socket_numa_mask_with_base(5, node_base) == 0b00110000

            # Socket 3: nodes 6,7
            assert _get_socket_numa_mask_with_base(6, node_base) == 0b11000000
            assert _get_socket_numa_mask_with_base(7, node_base) == 0b11000000

    def test_nonexistent_node_returns_fallback(self):
        """Non-existent NUMA node falls back to single-node mask."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topology = {(0, 0): "0-7"}
            _create_mock_sysfs(tmpdir, topology)

            # Node 99 doesn't exist, should fall back
            mask = _get_socket_numa_mask_with_base(99, "/nonexistent/path")
            assert mask == 1 << 99

    def test_cpulist_comma_separated(self):
        """Handle comma-separated CPU lists like '0-11,24-35'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topology = {
                (0, 0): "0-11,24-35",
                (0, 1): "12-23,36-47",
            }
            node_base = _create_mock_sysfs(tmpdir, topology)

            # Both nodes on socket 0
            mask = _get_socket_numa_mask_with_base(0, node_base)
            assert mask == 0b11

            mask = _get_socket_numa_mask_with_base(1, node_base)
            assert mask == 0b11

    def test_single_cpu_per_node(self):
        """Handle single CPU per NUMA node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            topology = {
                (0, 0): "0",
                (0, 1): "1",
                (1, 2): "2",
            }
            node_base = _create_mock_sysfs(tmpdir, topology)

            mask = _get_socket_numa_mask_with_base(0, node_base)
            assert mask == 0b011  # Nodes 0 and 1

            mask = _get_socket_numa_mask_with_base(2, node_base)
            assert mask == 0b100  # Node 2 only
