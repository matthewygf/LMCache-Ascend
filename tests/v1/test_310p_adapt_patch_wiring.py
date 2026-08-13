# SPDX-License-Identifier: Apache-2.0
"""Guard against the 310P adapt patcher being wired to a non-existent module."""

# Standard
import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATCH_DIR = _REPO_ROOT / "lmcache_ascend" / "integration" / "patch"
_APPLY_PATCH = _PATCH_DIR / "apply_patch.py"
_ADAPT_PATCH = _PATCH_DIR / "vllm" / "vllm_ascend_310p_adapt_patch.py"
_STALE_PATCH = _PATCH_DIR / "vllm" / "vllm_ascend_0_10_0_rc1_310p_patch.py"


def test_310p_adapt_patch_file_exists():
    assert _ADAPT_PATCH.is_file()
    assert "class VllmAscend0100rc1Patcher" in _ADAPT_PATCH.read_text(encoding="utf-8")
    assert not _STALE_PATCH.exists()


def test_apply_patch_references_existing_310p_module():
    source = _APPLY_PATCH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    module_refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    parts.append("{base}")
            module_refs.append("".join(parts))

    expected_suffix = ".vllm.vllm_ascend_310p_adapt_patch"
    assert any(ref.endswith(expected_suffix) for ref in module_refs), (
        f"apply_patch.py must reference vllm_ascend_310p_adapt_patch; "
        f"found f-string module refs: {module_refs}"
    )
    assert not any(
        "vllm_ascend_0_10_0_rc1_310p_patch" in ref for ref in module_refs
    ), "stale non-existent 310P patch module path must not remain"
