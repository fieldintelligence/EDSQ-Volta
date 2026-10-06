# Handoff → benchmark session

Read this first if you are the benchmark chat/session. Everything here is
verified state as of 2026-10-05, post PMem power cycle.

## 1. Hardware state (node2) — all green

| Tier | Device | State |
|---|---|---|
| Model home | `eds1` (label) → `/media/knight2/eds1` | **2026-10-06:** device names changed after the power cycle (now `/dev/nvme1n1p3`) and eds1 + claude-data were NOT in fstab, so they did not mount; mounted via `udisksctl mount`. Add by UUID to fstab (eds1 `edf52ef6-5c5b-4c81-9d31-f49493b95dc1`). Free: ~471G (all models moved here) |
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
  `v0.2.1-volta`. Tests: `python3 -m pytest tests/ -q` → 36/36 green
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
python3 -m pytest tests/ -q                      # 36/36 expected (V100 visible)
~/bin/gpu-temps                                  # thermal gate before runs
python3 tools/prepare_k25_shards.py <ckpt-on-eds1> <out> --dry-run
python -m vllm_volta.entrypoints.kimi_k25 <ckpt> # fails loudly: engine pin TODO
```

## 6. Measured inputs added 2026-10-06 (raw files under `evidence/microbench/`)

**PMem DAX parallel read** (`pmem_dax_parallel_read_20261006.json`, dd with N pinned readers; fio still owed):

| case | GB/s |
|---|---|
| local socket, 1 reader | 3.3 per mount |
| local socket, 4–16 readers | **10.1–10.5 per mount** (saturates at 4) |
| remote socket (CPU/mem on the other node) | **0.4–1.0** — more than 10× worse |
| both sockets concurrently, 16 local readers each | ~6.3 aggregate (caveat: a sha256 job ran in parallel; repeat with fio) |

Consequences for the tier model: use **10.5 GB/s per socket, not 13**, as the PMem bandwidth; NUMA-local
placement is mandatory (pmem0 experts read by socket-0 threads, pmem1 by socket-1), never interleave
across sockets. With ~8–9 GB of PMem expert bytes per token (65 % share) the PMem tier alone binds decode to
~0.7 t/s (int4) / ~0.95 t/s (int3 tail): the int3 tail is not sufficient on its own, frequency-aware placement
must shrink the PMem share substantially.

**Kimi K2.5 GGUF baseline** (virtualv_llm, unsloth `UD-IQ3_XXS`, 415 GB, llama.cpp build-v100):
all experts in DDR4 page cache (`--cpu-moe`, attention/shared on 6 GPUs, 48 threads) → TG ~1.2 t/s, PP ~4 t/s;
`--n-cpu-moe 44` OOMs (50 GB requested on one 20 GB Ada). 8-task battery: **0.833 in instant mode, 0.396 with
thinking** — Kimi K2.5 keeps thinking even with llama.cpp `--reasoning off`; disable it per request with
`chat_template_kwargs: {"thinking": false}` (the template emits `<think></think>`), or reasoning eats every fixed
task token budget. Source: virtuanalytica/virtualv_llm PR #19/#22, `docs/LESSONS_LIVE_MIXTURE_20261005.md`.

**Baselines to beat (same box):** Qwen3.8-Flash-Next AP-IQ4_XS 42.4 t/s (above) and, since 2026-10-06,
Qwen3.6-35B-A3B NVFP4 1Cat-vLLM TP2 on the V100 pair: 111.0 t/s B1, 288.5 B4, 980.3 B16 (8-task 0.854).

**Checkpoint:** native INT4 `moonshotai/Kimi-K2.5` (64 safetensors, 595 GB) is already at
`/media/knight2/eds1/Kimi-K2.5-Tower`; sha256 against the HF LFS oids re-run 2026-10-06.

**"colibri" = Colibrì** (github.com/JustVugg/colibri), not Colossal-AI: a pure-C engine that treats
VRAM/RAM/disk as one hierarchy and streams inactive MoE experts — direct prior art for this tier design and
a second reference engine. Installed + verified at `/media/knight2/EDS2/tools/colibri` (OpenAI-compatible;
`coli serve --model <dir> --model-id x --no-think`).
