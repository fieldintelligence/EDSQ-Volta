#!/usr/bin/env bash
# 2026-10-08: after the DSv4 battery, Kimi K2.7 Q3 and Q4 and Kimi K3 IQ2_XXS on
# the V100 pair (tier-sharded llama.cpp, identical flags and probes), then the
# GLM-5.3-Flash battery, then restore the day lane and cron.
set -u
W=<EDS2>/projects/virtualv_llm-wt-pub2; cd "$W"
L=$W/reports/benchmark_logs; V=$L/volta_k
OLD=<EDS2>/tools/llama.cpp/build-v100/bin/llama-server
FN=<EDS2>/models/llm/qwen38-flash-next-ap-iq2s/AP-IQ2_S/Qwen3.8-Flash-Next-AP-IQ2_S.gguf
log() { echo "$(date '+%F %T') $*"; }
health() { for i in $(seq 1 "$2"); do curl -sf -m3 "localhost:$1/health" >/dev/null 2>&1 && return 0; sleep 20; done; return 1; }

while kill -0 4106216 2>/dev/null; do sleep 60; done
log "DSv4-keten klaar; Kimi-keten start"
crontab -l | sed -E 's|^([^#].*bin/vllm-profile (day\|night) .*)$|#BATTERY-PAUSE \1|' | crontab -
systemctl --user stop day-flash-next local-chat-qwen38 2>/dev/null
systemctl --user reset-failed day-flash-next-ada 2>/dev/null
systemctl --user is-active -q day-flash-next-ada || systemd-run --user --unit=day-flash-next-ada --collect \
  -p LimitNOFILE=1048576 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=0,1,2,5 "$OLD" --model "$FN" \
  --alias qwen38-flash-next --host 127.0.0.1 --port 18012 --ctx-size 8192 --parallel 1 --split-mode layer \
  --gpu-layers 99 --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --jinja >/dev/null 2>&1
health 18012 60 && log "dag-lane op Ada gezond" || log "dag-lane op Ada NIET gezond"

kimi() { # tag model-shard-1 alias
  local tag=$1 model=$2 unit=volta-$1 port=18050
  log "$tag: prewarm NVMe-deel; vrij RAM vóór: $(free -g | awk 'NR==2{print $7}')G"
  for f in $(dirname "$model")/*.gguf; do t=$(readlink -f "$f"); case "$t" in /mnt/pmem0/*) ;; *) [ -f "$t" ] && cat "$t" >/dev/null;; esac; done
  log "$tag: prewarm klaar; vrij RAM: $(free -g | awk 'NR==2{print $7}')G; cache: $(free -g | awk 'NR==2{print $6}')G"
  ( for f in $(dirname "$model")/*.gguf; do echo "$(basename $f) -> $(readlink -f $f) $(stat -Lc %s $f)"; done ) > "$V/$tag.layout.txt"
  CMD=("$OLD" --model "$model" --alias "$3" --host 127.0.0.1 --port $port --ctx-size 8192 --parallel 1 -ngl 99
       --threads 48 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cpu-moe --no-op-offload --jinja --special)
  printf '%q ' CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4 "${CMD[@]}" > "$V/$tag.cmd.txt"
  systemctl --user reset-failed $unit 2>/dev/null
  systemd-run --user --unit=$unit --collect -p LimitNOFILE=1048576 -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=3,4 "${CMD[@]}" >/dev/null 2>&1
  if health $port 180; then
    log "$tag: server gezond, probes START"
    python3 "$V/probe.py" $port "$3" "$V/$tag.jsonl" > "$V/$tag.probe.log" 2>&1
    log "$tag: probes EIND"; cat "$V/$tag.probe.log"
  else log "$tag: server NIET gezond"; journalctl --user -u $unit --no-pager | grep -iE "error" | tail -3 > "$V/$tag.error.txt"; fi
  journalctl --user -u $unit --no-pager > "$V/$tag.server.log" 2>&1
  systemctl --user stop $unit 2>/dev/null; sleep 20
}

kimi k27-q3 <eds1>/models/kimi-k2.7-ud-q3/UD-Q3_K_XL/Kimi-K2.7-Code-UD-Q3_K_XL-00001-of-00011.gguf k27q3
kimi k27-q4 <eds1>/models/kimi-k2.7-ud-q4/Kimi-K2.7-Code-UD-Q4_K_XL-00001-of-00014.gguf k27q4

# K3: 6 shards on NVMe, 10 downloaded straight to PMem
T=/mnt/pmem0/kimi-k3-tail/UD-IQ2_XXS; N=<eds1>/models/llm/gguf/kimi-k3-ud-iq2xxs/UD-IQ2_XXS
while pgrep -f "hf download unsloth/Kimi-K3-GGUF" >/dev/null || [ "$(ls $T/*.gguf 2>/dev/null | wc -l)" -lt 10 ]; do
  [ "$(ls $T/*.gguf 2>/dev/null | wc -l)" -lt 10 ] && ! pgrep -f "hf download unsloth/Kimi-K3-GGUF" >/dev/null && { log "K3-download onvolledig en gestopt"; break; }
  sleep 120; done
K=$V/k3-model; rm -rf "$K"; mkdir -p "$K"
for f in $N/*.gguf $T/*.gguf; do ln -sf "$f" "$K/$(basename $f)"; done
log "K3: $(ls $K/*.gguf | wc -l)/16 shards gekoppeld"
[ "$(ls $K/*.gguf | wc -l)" -eq 16 ] && kimi k3-iq2xxs "$K/Kimi-K3-UD-IQ2_XXS-00001-of-00016.gguf" k3

log "GLM-batterij START"; bash "$L/battery_glm_rerun.sh"; log "GLM-batterij EIND"
crontab -l | sed -E 's|^#BATTERY-PAUSE ||' | crontab -
log "Kimi-keten klaar; cron: $(crontab -l | grep -c 'bin/vllm-profile') regels actief"
