# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
`LMCache-Ascend` is a plugin/overlay for upstream [LMCache](https://github.com/LMCache/LMCache)
that ports it to Huawei **Ascend NPU** hardware. It monkey-patches LMCache at import
time (`lmcache_ascend/__init__.py`) and ships native CANN/C++ ops (`csrc/`) that are
compiled against the Ascend CANN toolkit. It only runs as a KV connector inside a
serving engine (vLLM-Ascend, SGLang, or MindSpore + vLLM).

### Hardware constraint (important)
The Cursor Cloud VM is **x86_64 with no Ascend NPU and no CANN toolkit**. This means
the following are **not possible in the cloud VM** and should not be attempted as
verification here:
- Building the package (`pip install -e .`): `setup.py` calls `npu-smi` and CMake
  against CANN; it fails with "No available NPU found" and has no C++ toolchain target.
- Importing `lmcache_ascend`: fails because `_version.py`/`_build_info.py` are only
  generated during the NPU build.
- Running the unit tests (`pytest tests/v1`): `tests/conftest.py` hard-requires
  `torch_npu` and a working NPU (`torch.randn(..., device="npu")`) and calls
  `pytest.exit(...)` at collection time otherwise. `tests/bootstrap.py` also clones
  upstream LMCache (tag from `LMCACHE_UPSTREAM_TAG` in `lmcache_ascend/__init__.py`)
  into `/workspace/LMCache`.

Functional build/serve/test work must run on an Ascend host (Atlas 800I A2/A3) using
the official `quay.io/ascend/vllm-ascend` images. See `README.md` and `docs/deployment.md`
for the real build/serve/test commands; do not duplicate them here.

### What CAN run in the cloud VM: Code Quality gate
The only development check that works here is the lint/format/type/spell gate — the
same `pre-commit` suite run by `.github/workflows/code-quality.yml` (SPDX header check,
`isort`, `ruff`, `ruff-format`, `codespell`, `clang-format`, `mypy`). It needs no NPU.

`pre-commit` is installed to the user site (`~/.local/bin`, which is **not on PATH**),
so invoke it as a module:

```bash
python3 -m pre_commit run --all-files
```

Notes / gotchas:
- The first run downloads and builds each hook's isolated environment (needs network
  to github.com/pypi); subsequent runs reuse the cache and are fast.
- `mypy` here passes because the config uses `ignore_missing_imports = true` and
  `follow_imports = silent`, so the missing `torch`/`torch_npu`/`lmcache` deps do not
  break type checking.
- Do not commit a local virtualenv into the repo (`.venv` is not gitignored).
