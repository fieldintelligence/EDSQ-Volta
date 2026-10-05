# Storage map & rules (node2) — verified 2026-10-05

Ground truth from `lsblk`/`df` on node2. This document is the source of truth
for where model artifacts live; it supersedes earlier notes that only knew
about the Intel 750.

## Devices

| Device | Model | Size | Mount | Role |
|---|---|---|---|---|
| sda | Samsung PM800 128GB | 119G | `/` (95% full) | OS only — never models |
| nvme1n1 | Intel SSDPEDMW012T4 (SSD 750, 1.2TB) | 1.1T | `/media/knight2/EDS2` (351G free) | **staging/scratch only** — prepared-shard temp, never the model home |
| nvme2n1 | Intel SSDPECKE064T8 ("6.4TB" class) | 2.9T | `/media/knight2/claude-data` (99% full — 40G) | data volume; cleanup needed before it can hold anything |
| nvme0n1 | Intel SSDPECKE064T8 ("6.4TB" class) | 2.9T | **unconfigured LVM PV** (p3) | **MODEL TIER — activate as `eds1`** (see below) |
| PMem 4×512GB | Optane PMem 100 | 2T | **unconfigured** (no pmem/dax namespace) | expert-shard tier after App Direct setup (see below) |

The "6.4TB" is the SSDPECKE064T8 pair (2× 2.9 TiB ≈ 6.4 TB decimal).

## Rules

1. **Models always live on the 6.4TB pair** (`eds1`), never on the Intel 750
   and never on the root disk.
2. EDS2 (750) is scratch: quantization staging, prepared-shard temp, deletes
   after each run.
3. PMem is the expert-shard tier once configured; until then the model tier
   alone carries the workload.

## Activation runbook (root, once)

### 1. `eds1` model tier (~2.9T on nvme0n1p3)

```bash
sudo vgcreate eds1-vg /dev/nvme0n1p3
sudo lvcreate -n models -l 100%FREE eds1-vg
sudo mkfs.ext4 -L eds1 /dev/eds1-vg/models
sudo mkdir -p /media/knight2/eds1
echo '/dev/eds1-vg/models /media/knight2/eds1 ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a
```

Capacity check after mount: `df -h /media/knight2/eds1` → expect ~2.9T.

### 2. PMem → per-socket App Direct + fsdax (needs tools + POWER CYCLE)

Topology: 4× 512GB PMem100 on 2× Xeon 8259CL — 2 modules per socket.
App Direct interleaving is per-socket (no cross-socket interleave), so this
yields **2 regions × ~1TB** (region0 = socket0/numa0, region1 = socket1/numa1).
That is a feature, not a limitation: NUMA-local shard access.

```bash
# 0) tools (Ubuntu universe)
sudo apt install -y ipmctl ndctl

# 1) goal: 100% App Direct. ipmctl 3.x note: per-socket interleaving is the
#    DEFAULT of PersistentMemoryType=AppDirect (older 'AppDirect1Interleaved'
#    syntax does not exist in 3.x and errors out — verified on node2).
#    WARNING: erases the modules; requires a real POWER CYCLE afterwards
#    (shutdown + remove AC, not a warm reboot)
sudo ipmctl create -dimm all -goal MemoryMode=0 PersistentMemoryType=AppDirect
sudo ipmctl show -goal          # verify the PENDING goal before cycling
# expect per socket: ~960 GiB AppDirect1 (one interleaved set per socket)

# fstab opruimen als de pmem-regels voortijdig toegevoegd waren:
sudo sed -i '\#/dev/pmem[01] #d' /etc/fstab

# --- POWER CYCLE ---

# 2) one fsdax namespace per region
sudo ndctl create-namespace -m fsdax --region=region0 --align=2M --name=pmem0
sudo ndctl create-namespace -m fsdax --region=region1 --align=2M --name=pmem1

# 3) filesystems with DAX
sudo mkfs.ext4 -E lazy_itable_init=0,lazy_init=0 /dev/pmem0
sudo mkfs.ext4 -E lazy_itable_init=0,lazy_init=0 /dev/pmem1
sudo mkdir -p /mnt/pmem0 /mnt/pmem1
echo '/dev/pmem0 /mnt/pmem0 ext4 dax=always,nofail 0 2' | sudo tee -a /etc/fstab
echo '/dev/pmem1 /mnt/pmem1 ext4 dax=always,nofail 0 2' | sudo tee -a /etc/fstab
sudo mount -a

# 4) fstab (pas NA succesvolle namespaces) + verify
echo '/dev/pmem0 /mnt/pmem0 ext4 dax=always,nofail 0 2' | sudo tee -a /etc/fstab
echo '/dev/pmem1 /mnt/pmem1 ext4 dax=always,nofail 0 2' | sudo tee -a /etc/fstab
sudo systemctl daemon-reload && sudo mount -a
ndctl list -RN        # 2 regions, 2 namespaces
df -h /mnt/pmem0 /mnt/pmem1   # ~960G each (1/16 of PMem100 is reserved for metadata)
```

Memory Mode is deliberately rejected: it would consume the 832 GB DDR4 as a
cache and leave us without the page cache the expert tier depends on. App
Direct keeps DDR4 for the OS/page cache and gives ~13 GB/s per socket
region (2× ~6.6 GB/s modules interleaved) behind a DAX filesystem.

NUMA discipline: shards under `/mnt/pmem0` are consumed by socket-0 cores,
`/mnt/pmem1` by socket-1 cores (`numactl --cpunodebind` in the engine).
The placement plan round-robins shards between the two mounts so both
sockets carry half the expert traffic.

## Kimi K2.5 Tower artifact guidance

Kimi K2.5 ships **natively INT4** (~595 GB) — a Q4_K_M GGUF (~621 GB) is
*LARGER than the source* and is rejected as an artifact. Options, in order:

| Artifact | Size | Where | Notes |
|---|---|---|---|
| Native INT4 checkpoint | ~595 G | `eds1` | primary; fits DDR4 page cache (832 G) entirely; PMem becomes cold tier only |
| [unsloth UD-Q2_K_XL](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) | ~375 G | `eds1` | if 595 G proves too heavy for load windows |
| unsloth UD-TQ1_0 | ~245 G | `eds1` | 1.8-bit; quality floor unknown — holdout set must gate it |
| ~~Q4_K_M / Q3 GGUF~~ | — | — | rejected: no smaller than native, Q3 untested vs native INT4 |

Prepared per-expert shards (our format, `tools/prepare_k25_shards.py`) go to
`eds1` as the working copy; the original checkpoint stays as reference.
