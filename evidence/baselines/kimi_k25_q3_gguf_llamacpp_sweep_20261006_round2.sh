#!/usr/bin/env bash
# Kimi K2.5 UD-IQ3_XXS llama.cpp tuning sweep (TG/PP reported separately, instant mode, 2 probes per config,
# second = warm). A: all experts in RAM; B: + --numa distribute; C: B + 12 expert layers pinned on the 6 GPUs.
set -uo pipefail
S=/media/knight2/EDS2/tmp/claude-code/claude-1000/-home-knight2/d5c66977-7642-498c-ad30-cf3e853c5265/scratchpad
LOG=$S/k25_q3_sweep2.log; OUT=$S/k25_q3_sweep.jsonl; log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
SRV=/media/knight2/EDS2/tools/llama.cpp/build-v100/bin/llama-server
F=/media/knight2/eds1/models/kimi-k2.5-gguf/UD-IQ3_XXS/Kimi-K2.5-UD-IQ3_XXS-00001-of-00009.gguf
systemctl --user stop mom-proxy mom-qwen38b mom-gemma4 mom-qwen35b mom-devstral mom-qwen35 2>/dev/null; sleep 10
probe() {  # $1 config label
  for i in 1 2; do
    python3 - "$1" "$i" >> "$OUT" <<'PY'
import json,sys,time,urllib.request
body={"model":"k","messages":[{"role":"user","content":"Context: Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns. Numerai Signals scores 20-day forward returns.  Question: Explain in detail how point-in-time validation prevents look-ahead bias in financial machine learning."}],
      "max_tokens":96,"temperature":0,"chat_template_kwargs":{"thinking":False}}
t=time.time(); r=json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18019/v1/chat/completions",
  data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=3600).read())
tm=r.get("timings",{})
print(json.dumps({"config":sys.argv[1],"probe":int(sys.argv[2]),"tg_tps":tm.get("predicted_per_second"),"pp_tps":tm.get("prompt_per_second"),
  "tg_tokens":tm.get("predicted_n"),"pp_tokens":tm.get("prompt_n"),"wall_s":round(time.time()-t,1)}))
PY
  done
}
run() {  # $1 label, rest = extra args
  local label=$1; shift
  systemd-run --user --unit=k25-sweep --collect -p LimitNOFILE=1048576 -E CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -E CUDA_VISIBLE_DEVICES=3,4,0,1,2,5 "$SRV" --model "$F" --alias k --host 127.0.0.1 --port 18019 \
    --ctx-size 8192 --parallel 1 --n-gpu-layers 99 --threads 48 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --jinja "$@"
  if timeout 3600 bash -c 'until curl -sf -m3 localhost:18019/health >/dev/null; do systemctl --user is-active --quiet k25-sweep || exit 1; sleep 10; done'; then
    log "$label healthy"; journalctl --user -u k25-sweep --no-pager -o cat | grep -E 'model buffer size' >> "$LOG"; probe "$label"; log "$label done"
  else log "$label FAILED"; journalctl --user -u k25-sweep -n 25 --no-pager -o cat >> "$LOG"; fi
  systemctl --user stop k25-sweep; sleep 15
}
run A2_cpu_moe_longprompt --cpu-moe
run D_cpu_moe_noopoff --cpu-moe --no-op-offload
run E_cpu_moe_noopoff_t32 --cpu-moe --no-op-offload --threads 32
SRVL=$SRV; G=/media/knight2/claude-data/knight1/eds1/models/llm/gguf
srv() { systemd-run --user --unit="$1" --collect -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES="$2" "$SRVL" --model "$4" --alias "$5" \
  --host 127.0.0.1 --port "$3" --ctx-size "$6" --parallel 2 --gpu-layers 99 --flash-attn on --reasoning off \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja >> "$LOG" 2>&1; }
srv mom-devstral 0 8022 $G/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf devstral 32768
srv mom-qwen35 1 8023 "$G/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf" qwen35 32768
srv mom-qwen38b 4 8026 "$G/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf" qwen38 65536
srv mom-qwen35b 5 8025 "$G/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf" qwen35 32768
srv mom-gemma4 2 8024 "$G/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf" gemma4 32768
timeout 600 bash -c 'for p in 8022 8023 8024 8025 8026; do until curl -sf -m2 localhost:$p/health >/dev/null; do sleep 5; done; done'
systemd-run --user --unit=mom-proxy --collect -p WorkingDirectory=/media/knight2/EDS2/projects/virtualv_llm-wt-mom python3 \
  /media/knight2/EDS2/projects/virtualv_llm-wt-mom/scripts/benchmarks/mixture_proxy.py --port 8030 --name mom-live \
  "--aggregator=qwen38=http://127.0.0.1:8026/v1" --proposer devstral=http://127.0.0.1:8022/v1 \
  "--proposer=qwen35=http://127.0.0.1:8023/v1|http://127.0.0.1:8025/v1" --proposer gemma4=http://127.0.0.1:8024/v1
log "mom-live restored; SWEEP DONE"
