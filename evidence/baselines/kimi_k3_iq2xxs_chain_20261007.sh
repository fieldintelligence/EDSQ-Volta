#!/usr/bin/env bash
set -u
SRC=/media/knight2/claude-data/knight1/eds1/models/llm/gguf/kimi-k3-ud-iq2xxs/UD-IQ2_XXS
TAIL=/mnt/pmem0/kimi-k3-tail
M=/media/knight2/eds1/models/kimi-k3-ud-iq2xxs

# 1) wacht op download (hf-proces exit + geen .incomplete/grows)
while pgrep -f "hf download unsloth/Kimi-K3" >/dev/null 2>&1; do sleep 60; done
echo "$(date '+%T') K3 download klaar: $(du -sh $SRC | cut -f1)"

# 2) tier-split: ~450G op NVMe (page cache), rest (~212G) kopieren naar PMem
mkdir -p $M "$TAIL"
NVME_BUDGET=$((450 * 1024**3))
acc=0
shards=($(ls $SRC/*.gguf | sort))
: > /tmp/k3_split.log
for f in "${shards[@]}"; do
  sz=$(stat -c%s "$f")
  if [ $((acc + sz)) -le $NVME_BUDGET ]; then
    ln -sf "$f" "$M/$(basename $f)"; acc=$((acc+sz)); echo "NVMe $(basename $f)" >> /tmp/k3_split.log
  else
    cp -n "$f" "$TAIL/" && ln -sf "$TAIL/$(basename $f)" "$M/$(basename $f)"
    echo "PMem $(basename $f)" >> /tmp/k3_split.log; acc=$((acc+sz))
  fi
done
echo "$(date '+%T') split klaar: NVMe-part $(du -shL $M 2>/dev/null | cut -f1) totaal logisch"

# 3) prewarm alleen het NVMe-deel
for f in $M/*.gguf; do
  target=$(readlink -f "$f")
  case "$target" in /media/knight2/claude-data/*) cat "$f" > /dev/null;; esac
done
echo "$(date '+%T') prewarm klaar; free: $(free -g | awk 'NR==2{print $7}')G"

# 4) serveer
systemctl --user reset-failed k3-judge 2>/dev/null
systemd-run --user --unit=k3-judge --collect -p LimitNOFILE=1048576 -E CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -E CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 /media/knight2/EDS2/tools/llama.cpp/build-v100-new/bin/llama-server \
  --model $M/$(basename ${shards[0]}) --alias k3 --host 127.0.0.1 --port 18022 \
  --ctx-size 8192 --parallel 1 -ngl 99 --threads 32 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --cpu-moe --no-op-offload --jinja --special 2>&1 | head -1
for i in $(seq 1 90); do curl -sf -m3 localhost:18022/health >/dev/null 2>&1 && break; sleep 10; done
curl -sf -m3 localhost:18022/health >/dev/null || { echo "K3 niet healthy"; journalctl --user -u k3-judge -o cat -n 8 | grep -iE "error" | head -4; exit 1; }
echo "$(date '+%T') K3 healthy op 18022"

# 5) probe + portfolio
python3 - <<'PY'
import json, time, urllib.request, re
def ask(q, mt, tag):
    body={"model":"k3","temperature":0,"max_tokens":mt,
          "messages":[{"role":"user","content":q}]}
    t=time.time()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:18022/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}), timeout=7200).read())
    tm=r.get("timings",{})
    print(f"{tag}: PP {tm.get('prompt_per_second',0):.1f} t/s | TG {tm.get('predicted_per_second',0):.2f} t/s "
          f"({tm.get('predicted_n')} tok) | {time.time()-t:.0f}s", flush=True)
    m=r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or "")
ask("Warmup tick.", 16, "WARMUP")
ask("Context over market microstructure. "*20 + "Name one order-book imbalance signal.", 96, "PROBE-A")
ask("Context over market microstructure. "*20 + "Name one order-book imbalance signal.", 96, "PROBE-B")
ans, reason = ask("You are the aggregator of a mixture of models. Draft answers:\n"
    "[devstral] Portfolio vol = sqrt(0.6^2*0.12^2 + 0.4^2*0.07^2 + 2*0.6*0.4*0.2*0.12*0.07) = 8.9%\n"
    "[gemma4] Expected return 6.0%, volatility 9.2%\n"
    "[qwen38] Expected return 6.0%, volatility 8.9%\n\n"
    "Question: 60/40 portfolio, E[r1]=8%, vol1=12%, E[r2]=3%, vol2=7%, corr=0.2. "
    "Compute portfolio expected return and volatility (1 decimal). Verify the drafts. "
    "End with: RETURN: X%, VOL: Y%", 2048, "JUDGE-TEST")
allt = (reason or "") + "\n" + (ans or "")
print("uitspraken:", re.findall(r"(RETURN|VOL):[^%\n]*([\d.]+)%", allt))
print("laatste 300:", allt[-300:])
PY
echo "$(date '+%T') K3 TEST KLAAR"
