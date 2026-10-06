# Handoff → benchmark session

Read this first if you are the benchmark chat/session. Everything here is
verified state as of 2026-10-05, post PMem power cycle.

## 1. Hardware state (node2) — all green

| Tier | Device | State |
|---|---|---|
| Model home | `eds1` = `/dev/nvme0n1p3` → `/media/knight2/eds1` | 2.8T free |
| Expert tier | `/mnt/pmem0` (socket0), `/mnt/pmem1` (socket1) | 972G each, ext4 `dax=always`, live |
| TP2 engine pair | 2× V100-SXM2-32GB (NVLink) | paged-attn V1 kernel validated on-hardware |
| Aux GPUs | 4× RTX 4000 Ada 20G | fan manager controls blowers |
| DDR4 | 512 GiB (NOT 832 — verified via ipmctl) | page cache for hot experts |
| Scratch | `/media/knight2/EDS2` (351G) | staging only, delete after run |

Cooling: `~/bin/gpu-temps` (live), `~/bin/gpu-fan-manager` (`--apply` needs
the root crontab entry; see `docs/` and shell history). **Rule: only run
benchmarks with all Ada cards < 78 °C** — GPU0/GPU5 hit 80–82 °C idle-loaded.

## 2. Software state

- Repo: `/home/knight2/repos/1Cat-vLLM-Volta` — 18 merged PRs, tag
  `v0.2.1-volta`. Tests: `python3 -m pytest tests/ -q` → 22/22 green
  (kernel tests need a V100 visible).
- Kernel JIT cache: `~/.cache/torch_extensions` (wipe `volta_ops_test` when
  changing flags).
- EDSQ contract: `vllm_volta/quant.py` (int4/int3, format magic `EDSQ` v1).
- Shard prepper: `tools/prepare_k25_shards.py` (`--dry-run`, `--selftest`).
- Entrypoint: `vllm_volta/entrypoints/kimi_k25.py` (TP2, fp16, single-request).
- Evidence dir: `evidence/` — kernel microbench + correctness records live
  there; raw JSON, no averaged marketing numbers.

## 3. Engine strategy (updated 2026-10-05: colibri installed locally)

**Primary serving path = Colibri** (v1.11.0, `/media/knight2/EDS2/tools/colibri`,
see docs/engine-colibri.md): pure-C, OpenAI-compatible (`coli serve
--model-id x --no-think`), disk/RAM/VRAM as one hierarchy, `qwen38` family
already supported (baseline replicable directly), DeepSeek-V4-Flash served
from tiers today. K2.5 Tower needs a `kimi_k25` family descriptor
(branch `feat/colibri-engine-track`, ours to contribute upstream).

**Research path = vLLM sm_70 pin** (demoted): still wanted for our CUDA
kernels + deferred TP2 reductions; question pending on
[1CatAI/1Cat-vLLM#987](https://github.com/1CatAI/1Cat-vLLM/issues/987)
(fork: `febuz/1Cat-vLLM`, branch `feat/volta-edSQ-k25`).

**Protocol**: two-engine cross-check on identical holdout prompts;
`well_known_suite.py` for reproducibility, our holdout set on top.

**EDSQ int3-tail decision** — DAX bypasses the page cache: PMem-resident
experts are real Optane traffic. Parallel probe tool exists
(`tools/tier_probe.py`); with ~65% of K2.5 per-token expert bytes on PMem,
decode is tier-bound (~1–2 t/s) unless the tail is int3 (≈ −25%) or the hot
fraction grows. Decide with probe numbers.

## 4. Benchmark rules (carried over, non-negotiable)

- Report **PP and TG** separately; hybrid configs shift ranking on PP alone.
- Name results **"Kimi K2.5 Tower (EDSQ-1 …)"** — never "our model"; Kimi
  K2.5 is Moonshot AI's. Credit 1Cat for the ported playbook.
- Contamination-aware weighting: LiveBench/LiveCodeBench-style fresh sets +
  own holdout; publish artifacts (config fingerprint, raw timings) with every
  claim; `evidence/` stays raw.
- One quant on scratch at a time; delete after the run.
- Baseline to beat: Qwen3.8-Flash-Next AP-IQ4_XS · 2×V100 · 42.4 t/s.
- Publishing: GitHub (rename off the 1Cat brand, credit them) and HF upload
  only AFTER the holdout sweep — publication is the contamination moment.

## 5. Quick commands

```bash
cd /home/knight2/repos/1Cat-vLLM-Volta
python3 -m pytest tests/ -q                      # 22/22 expected (V100 visible)
~/bin/gpu-temps                                  # thermal gate before runs
python3 tools/prepare_k25_shards.py <ckpt-on-eds1> <out> --dry-run
python -m vllm_volta.entrypoints.kimi_k25 <ckpt> # fails loudly: engine pin TODO
```
