# Kimi K2.5 GGUF Q3 baseline on llama.cpp (2026-10-06)

Reference for Kimi K2.5 Tower (EDSQ-1): the community `unsloth/Kimi-K2.5-GGUF` `UD-IQ3_XXS` (415 GB,
sha256-verified) on llama.cpp `build-v100` (sm70+sm86), attention/shared weights on the GPUs (`-ngl 99`,
CUDA 3,4,0,1,2,5 in PCI order; GPU 3 shared with an unrelated 17 GB local-chat server), MoE experts in DDR4
page cache. Instant mode (`chat_template_kwargs: {"thinking": false}`), temperature 0, 96 generated tokens,
two probes per config (probe 2 = prompt cached, warm TG). Raw rows: `kimi_k25_q3_gguf_llamacpp_sweep_20261006.jsonl`.

| config | flags | PP t/s | TG t/s (probe 1 / 2) |
|---|---|---|---|
| A | `--cpu-moe`, 48 threads, 45-token prompt | 0.8 | 2.15 / 2.16 |
| B | A + `--numa distribute` | 0.8 | 1.78 / 1.43 |
| C | B + 12 expert layers pinned on the 6 GPUs (`-ot`) | 1.0 | 1.54 / 1.74 |
| A2 | A, 450-token prompt | 4.2 | 1.65 / 1.73 |
| D | A2 + `--no-op-offload` | 7.2 | 1.51 / 1.43 |
| E | D with 32 threads | **13.0** | 0.97 / 1.11 |

Reading: TG is best with the defaults (~1.7–2.2 t/s); `--numa distribute` and pinning expert layers on the
GPUs did not help (per-layer CPU↔GPU activation hops). PP gains 3× without op-offload and with 32 threads,
at a TG cost — a PP/TG trade-off, so report both. Caveats: single probes per config, and other sessions'
CPU jobs (Numerai research/backfill) ran concurrently; TG varies ~±20 % between identical configs (A vs A2).
Repeat with ≥3 probes on an idle box before quoting a single number.

Quality on the same build: 8-task battery 0.833 (instant) / 0.396 (thinking eats the fixed task budgets).
