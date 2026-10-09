#!/usr/bin/env python3
"""Fixed probe set for tier-sharded Kimi runs on the V100 pair (2026-10-08).

Same prompts, budgets and greedy decoding for every model so Q3, Q4 and K3
are comparable. One JSON line per request with the server's own timings.
"""
import json, sys, time, urllib.request

PORT, ALIAS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CODE = ("Write a Python function tier_ceiling(ram_gb, os_gb, other_cache_gb, headroom_gb) that returns the "
        "largest page-cache tier in GB that fits, never negative. Then write three assert statements that test "
        "it. Output only one Python code block.")
WORD = ("Priya sells tokens for 3 credits each. She pays 15 credits for a stall licence and 9 credits for a "
        "display case. She sells 22 tokens. What is her net profit in credits? End with 'Answer: <number>'.")
PROBES = [("warmup", "Warmup tick.", 16), ("probe-cold", "Context over market microstructure. " * 20 +
          "Name one order-book imbalance signal.", 96), ("probe-warm", "Context over market microstructure. " * 20 +
          "Name one order-book imbalance signal.", 96), ("word-problem", WORD, 2048), ("code-task", CODE, 2048)]

def ask(prompt, budget):
    body = {"model": ALIAS, "temperature": 0, "max_tokens": budget, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=5400) as response:
        reply = json.loads(response.read())
    return reply, time.time() - started

def check(tag, text):
    if tag == "word-problem":
        return "Answer: 42" in text.replace("**", "")
    if tag == "code-task":
        code = text.split("```python")[-1].split("```")[0] if "```" in text else text
        try:
            exec(compile(code, "<probe>", "exec"), {})
            return "assert" in code
        except Exception:
            return False
    return None

with open(OUT, "a") as out:
    for tag, prompt, budget in PROBES:
        try:
            reply, wall = ask(prompt, budget)
        except Exception as exc:
            out.write(json.dumps({"probe": tag, "error": f"{type(exc).__name__}: {exc}"}) + "\n"); out.flush(); break
        msg, t = reply["choices"][0]["message"], reply.get("timings", {})
        text = msg.get("content") or ""
        row = {"probe": tag, "max_tokens": budget, "wall_sec": round(wall, 1),
               "prompt_tokens": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
               "completion_tokens": t.get("predicted_n"), "decode_tps": t.get("predicted_per_second"),
               "finish_reason": reply["choices"][0].get("finish_reason"), "visible_chars": len(text),
               "reasoning_chars": len(msg.get("reasoning_content") or ""), "correct": check(tag, text),
               "content": text[-1500:]}
        out.write(json.dumps(row) + "\n"); out.flush()
        print(f"{tag}: PP {row['prompt_tps']} TG {row['decode_tps']} tok {row['completion_tokens']} "
              f"wall {row['wall_sec']}s correct {row['correct']}", flush=True)
