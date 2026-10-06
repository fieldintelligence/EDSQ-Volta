#!/usr/bin/env python3
"""GSM8K thinking-vs-instant probe on Kimi K2.5 (E-config, llama.cpp).

Question (2026-10-06): can thinking RAISE answer accuracy if the visible-
answer budget is generous, and what does it cost in tokens/time? Context:
the suite battery scored thinking 0.396 vs instant 0.833 because reasoning
ate the fixed per-task budgets — that is a budget artifact, not capability.

Design: 12 fixed GSM8K test questions (seed 7), temperature 0, per question
two requests to the same server: THINKING (default on via --jinja,
max_tokens 1280) and INSTANT (chat_template_kwargs thinking=false,
max_tokens 512). Extraction: text after </think> if present, then last
'#### N' or trailing number. Logs JSONL rows next to this script.
"""
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:18019/v1/chat/completions"
OUT = "/home/knight2/repos/1Cat-vLLM-Volta/evidence/baselines/kimi_k25_gsm8k_thinking_probe_20261006.jsonl"

# 12 GSM8K test questions (deterministic pick, seed 7, skip first 40)
try:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(40, 100))[::5][:12]
    QUESTIONS = [(i, ds[i]["question"], ds[i]["answer"]) for i in idx]
except Exception as e:  # fallback: cached lm-eval copy
    sys.exit(f"dataset load failed: {e}")


def ask(question: str, thinking: bool, max_tokens: int) -> tuple[str, dict]:
    body = {"model": "k", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": question}]}
    if not thinking:
        body["chat_template_kwargs"] = {"thinking": False}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=3600).read())
    text = r["choices"][0]["message"]["content"]
    tm = r.get("timings", {})
    return text, {"gen_tokens": r["usage"]["completion_tokens"],
                  "seconds": round(time.time() - t, 1),
                  "tg_tps": tm.get("predicted_per_second"),
                  "pp_tps": tm.get("prompt_per_second")}


NUM = re.compile(r"(-?[\d,]+\.?\d*)")


def extract(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m[-1].replace(",", "")
    nums = NUM.findall(text.replace(",", ""))
    return nums[-1] if nums else ""


def gold(answer: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer)
    return m.group(1).replace(",", "") if m else ""


rows = []
correct = {"thinking": 0, "instant": 0}
for i, q, a in QUESTIONS:
    row = {"idx": i, "gold": gold(a)}
    for mode, mt in (("thinking", 1280), ("instant", 512)):
        try:
            text, meta = ask(q, thinking=(mode == "thinking"), max_tokens=mt)
        except Exception as e:
            row[mode] = {"error": str(e)[:200]}
            continue
        ans = extract(text)
        ok = ans == row["gold"]
        correct[mode] += int(ok)
        row[mode] = {"answer": ans, "correct": ok, **meta}
        print(f"q{i} {mode}: {'OK ' if ok else 'MISS'} ans={ans!r} gold={row['gold']!r} "
              f"({meta['gen_tokens']} tok, {meta['seconds']}s)", flush=True)
    rows.append(row)
    with open(OUT, "w") as f:  # incremental dump
        json.dump(rows, f, indent=1)

n = len(QUESTIONS)
print(json.dumps({"n": n, "thinking_correct": correct["thinking"],
                  "instant_correct": correct["instant"],
                  "thinking_acc": correct["thinking"] / n,
                  "instant_acc": correct["instant"] / n}))
