# SPDX-License-Identifier: Apache-2.0
"""Regression: SGLang runtime must rebind CreateGPUConnector to CreateNPUConnector.

Without this, ``init_lmcache_engine`` keeps upstream CUDA ``CreateGPUConnector``
and never installs ``SGLangLayerwiseNPUConnector``. Upstream then calls
``single_layer_kv_transfer_sgl`` / ``TransferDirection`` (absent from Ascend
``c_ops``) or rejects Ascend's layer-concatenated KV layout.
"""

# Standard
import ast
from pathlib import Path


def _init_module_source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "lmcache_ascend" / "__init__.py"
    ).read_text(encoding="utf-8")


def test_sglang_runtime_patches_gpu_connector_factory():
    """Bootstrap must call ``_patch_gpu_connector`` when ``is_sgl`` is true."""
    tree = ast.parse(_init_module_source())
    found = False

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            nonlocal found
            test = node.test
            names = set()
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
                for value in test.values:
                    if isinstance(value, ast.Name):
                        names.add(value.id)
            if names == {"is_vllm", "is_sgl"}:
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "_patch_gpu_connector"
                    ):
                        found = True
            self.generic_visit(node)

    Visitor().visit(tree)
    assert found, (
        "Expected ``_patch_gpu_connector()`` under ``if is_vllm or is_sgl`` "
        "(or equivalent). SGLang must install CreateNPUConnector."
    )


def test_gpu_connector_patch_not_vllm_only():
    """Guard against regressing to ``if is_vllm: _patch_gpu_connector()`` only."""
    tree = ast.parse(_init_module_source())
    vllm_only = False

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            nonlocal vllm_only
            test = node.test
            if isinstance(test, ast.Name) and test.id == "is_vllm":
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "_patch_gpu_connector"
                    ):
                        vllm_only = True
            self.generic_visit(node)

    Visitor().visit(tree)
    assert not vllm_only, (
        "``_patch_gpu_connector()`` must not be gated on ``is_vllm`` alone; "
        "SGLang also requires CreateNPUConnector."
    )
