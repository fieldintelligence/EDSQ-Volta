#!/usr/bin/env python3
"""find_sm70_vllm.py — probe candidate vLLM releases for a compute_70 build.

vLLM dropped Volta when the V1 engine became mandatory. This script lists
candidate tags (V0-engine era), prints the exact build recipe per candidate,
and checks the local torch/nvcc pair so failures are caught before the
build. Network access required for tag listing; build runs locally.

  python3 tools/find_sm70_vllm.py --list
  python3 tools/find_sm70_vllm.py --check-env
  python3 tools/find_sm70_vllm.py --try v0.6.6.post1
"""

import argparse
import json
import subprocess
import sys

# V0-engine era candidates, oldest-last; sm_70 wheels existed through 0.6.x
CANDIDATES = ["v0.5.5", "v0.6.0", "v0.6.3", "v0.6.4.post1", "v0.6.6.post1"]


def list_candidates() -> None:
    print(json.dumps({"candidates": CANDIDATES,
                      "note": "sm_70 support ended when V1 became mandatory; "
                              "verify per tag with the build recipe"}, indent=1))


def check_env() -> int:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    print("nvcc:", nvcc.stdout.strip().splitlines()[-1] if nvcc.returncode == 0 else "ABSENT")
    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    print("device caps:", caps, "sm70 present:", (7, 0) in caps)
    if torch.version.cuda and nvcc.returncode == 0:
        v_torch = tuple(map(int, torch.version.cuda.split(".")[:2]))
        v_nvcc = tuple(int(x) for x in nvcc.stdout.split("release ")[1].split(",")[0].split(".")[:2])
        print("cuda version match:", v_torch == v_nvcc,
              "(mismatch raises in cpp_extension; JIT load tolerated it on node2)")
    return 0


def build_recipe(tag: str) -> None:
    print(f"""# build recipe for {tag} (run inside a venv with the pinned torch)
git clone --depth 1 --branch {tag} https://github.com/vllm-project/vllm vllm-{tag}
cd vllm-{tag}
TORCH_CUDA_ARCH_LIST="7.0" VLLM_TARGET_DEVICE=cuda pip install -e . --no-build-isolation
# smoke: python -c "from vllm import LLM; LLM(model='facebook/opt-125m', dtype='half')"
# record tag + sha in patches/README.md; engine-side fixes go to patches/ as
# FIX_FOR_VLLM_CUSTOM=<sha> .patch files (1Cat convention)""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--try", dest="tag")
    a = ap.parse_args()
    if a.list:
        list_candidates()
    if a.check_env:
        return check_env()
    if a.tag:
        build_recipe(a.tag)
    if not (a.list or a.check_env or a.tag):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
