#!/usr/bin/env python3
"""calibrate_expert_freq.py — build router-activation weights for
frequency-aware expert placement (PowerInfer-style hot/cold split).

Engine-agnostic by design: this tool consumes a JSONL trace of
router-activation events, NOT a running model. Produce traces with
whichever engine can run the model:

  {"layer": 12, "expert": 107}                      # one line per activation
  {"layer": 12, "counts": [3, 0, 41, ...]}          # or per-layer histograms

Sources: a vLLM hook on the router, HF transformers forward hooks, or any
instrumented engine. Calibration prompts should come from the HOLDOUT-side
distribution you actually serve (docs: contamination discipline — never
calibrate on benchmark items).

Usage:
  python tools/calibrate_expert_freq.py trace1.jsonl trace2.jsonl \
      --num-experts 384 --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys

REPO = os_path = __import__("os").path.dirname(__file__)
sys.path.insert(0, REPO + "/..")

from vllm_volta.moe import merge_calibration  # noqa: E402
from vllm_volta.quant import fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("traces", nargs="+", help="JSONL activation traces")
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    events = []
    for path in args.traces:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    weights = merge_calibration(events)
    missing = args.num_experts - len(weights)
    doc = {
        "format": "edsq-calibration-1",
        "num_experts": args.num_experts,
        "events": len(events),
        "sources": args.traces,
        "expert_weights": {str(e): round(weights.get(e, 0.0), 6)
                           for e in range(args.num_experts)},
        "_note": f"{missing} expert(s) never activated; they receive weight 0 "
                 f"and place into the coldest tier",
    }
    doc["fingerprint"] = fingerprint(doc)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    top = sorted(doc["expert_weights"].items(), key=lambda kv: -kv[1])[:5]
    print(f"[edsq] {len(events)} events -> {len(weights)} activated experts; "
          f"top-5: {top}")
    print(f"[edsq] wrote {args.out} (fp {doc['fingerprint']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
