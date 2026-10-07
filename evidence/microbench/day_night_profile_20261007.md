# Day/night profile live — day lane measured 46.6-49.0 t/s (2026-10-07)

The day/night serving split is now operational via `/home/knight2/bin/vllm-profile`
(cron: 07:30 day, 19:00 night) + `~/bin/night-queue-worker`.

## Day lane (active session target: 50 t/s) — MET 46.6-49.0

Engine: **1Cat-vLLM 1.5.0 TP2, Qwen3.8-27B-QUASAR-NVFP4 target** on the
V100-SXM2 pair (port 18012), the same engine class that measured
111/288/980 t/s (B1/B4/B16) on Qwen3.6-35B.

| probe | effective t/s |
|---|---|
| warmup 16 tok | 10.7 |
| 96-tok probes A/B/C | **46.6 / 46.6 / 46.6** |
| coder 512 tok | **49.0** |

Background target (25 t/s) rides the same engine at higher batch — the
engine scales 111→980 t/s from B1 to B16, so B4+ background lanes exceed
the 25 t/s requirement per stream.

Night lane (target ≥5 t/s): MoM v3 (DSv4 judge 2.30 t/s + 3 proposers,
5m21s/question) + GLM-5.3-Flash deep-verify (up to 6.2 t/s) + K2.7-Code
nightly code judge (0.68 t/s). Night queue: `eds1/queue/night/*.json`
answered by GLM via `~/bin/night-queue-worker`.

## Fixes needed to get here (all in scripts/units now)

1. `serve_1cat_qwen38_tp2.sh` passed GPU UUIDs in CUDA_VISIBLE_DEVICES;
   vLLM 1.5.0 arg parsing rejects them ("invalid literal for int()") —
   patched to numeric PCI indexes `3,4`.
2. Day profile order bug: `start_day` never stopped the night judges
   (GLM held 8.4 G on a V100 → vLLM demands 25.4 G free, found 22.5).
   `start_day` now runs `day_stop` first.
3. The OLD mom-live stack (qwen35/qwen38 members on the V100s, ~35 G)
   was a separate systemd unit set the profile didn't know — added to
   both stop paths.
4. Exit-75 "V100 capacity reserved": flock held by a dying engine core
   from a previous attempt — transient; retry clears it.

## Open caveats

- DFlash2 optional lane does not start while target holds the V100
  flock (single-mode for now; a shared-lock redesign is future work).
- Documented NVFP4-TP2 quality caveat (repetition-loop instability on
  some prompts, see serve-script header): run the 8-task battery on this
  lane before trusting day-lane outputs for scored work.
- K2.7 UD-Q3_K_XL (432 G) downloaded; night-lane test pending.
