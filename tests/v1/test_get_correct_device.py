# SPDX-License-Identifier: Apache-2.0
"""Tests for local NPU device mapping in get_correct_device."""

# Third Party
import pytest

# First Party
from lmcache_ascend.v1 import transfer_channel as tc_mod
from lmcache_ascend.v1.transfer_channel import get_correct_device


class TestGetCorrectDevice:
    def test_cpu_passthrough(self):
        assert get_correct_device("cpu", worker_id=9) == "cpu"

    def test_prefers_current_device_over_global_rank(self, monkeypatch):
        """Multi-node global ranks must not become npu:{worker_id}."""
        monkeypatch.setattr(tc_mod.torch.npu, "device_count", lambda: 8)
        monkeypatch.setattr(tc_mod.torch.npu, "current_device", lambda: 1)

        assert get_correct_device("npu", worker_id=9) == "npu:1"
        assert get_correct_device("npu:0", worker_id=15) == "npu:1"

    def test_fallback_modulo_when_current_device_unavailable(self, monkeypatch):
        """Match CreateNPUConnector: worker_id % device_count."""
        monkeypatch.setattr(tc_mod.torch.npu, "device_count", lambda: 8)

        def _boom():
            raise RuntimeError("no current device")

        monkeypatch.setattr(tc_mod.torch.npu, "current_device", _boom)

        assert get_correct_device("npu", worker_id=9) == "npu:1"
        assert get_correct_device("npu", worker_id=0) == "npu:0"

    def test_single_visible_device_maps_any_rank_to_zero(self, monkeypatch):
        """ASCEND_RT_VISIBLE_DEVICES with one device → always npu:0."""
        monkeypatch.setattr(tc_mod.torch.npu, "device_count", lambda: 1)
        monkeypatch.setattr(tc_mod.torch.npu, "current_device", lambda: 0)

        assert get_correct_device("npu", worker_id=0) == "npu:0"
        assert get_correct_device("npu", worker_id=3) == "npu:0"

    def test_no_devices_raises(self, monkeypatch):
        monkeypatch.setattr(tc_mod.torch.npu, "device_count", lambda: 0)

        with pytest.raises(RuntimeError, match="No NPU devices"):
            get_correct_device("npu", worker_id=0)

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            get_correct_device("cuda", worker_id=0)
