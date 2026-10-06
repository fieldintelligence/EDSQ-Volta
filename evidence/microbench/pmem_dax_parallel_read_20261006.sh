#!/usr/bin/env bash
# Parallel PMem DAX read bandwidth: per mount 16 x 2 GiB files, read with N parallel dd (bs=2M), pinned to the local
# socket (pmem0=node0, pmem1=node1) and, for contrast, to the remote socket. DAX bypasses the page cache, so every
# read is real Optane traffic. Writes the files once; removes them at the end.
set -uo pipefail
OUT=/media/knight2/EDS2/tmp/claude-code/claude-1000/-home-knight2/d5c66977-7642-498c-ad30-cf3e853c5265/scratchpad/pmem_bw.tsv
echo -e "mount\tnode\tparallel\tGiB\tseconds\tGBps" > $OUT
prep() { local m=$1 n=$2; mkdir -p $m/bw; for i in $(seq 1 16); do numactl --cpunodebind=$n --membind=$n dd if=/dev/zero of=$m/bw/f$i bs=2M count=1024 oflag=direct status=none & done; wait; sync; }
run() { local m=$1 n=$2 p=$3; local t0=$(date +%s.%N); for i in $(seq 1 $p); do numactl --cpunodebind=$n --membind=$n dd if=$m/bw/f$i of=/dev/null bs=2M status=none & done; wait
  local t1=$(date +%s.%N); python3 -c "s=$t1-$t0; g=2*$p; print(f'$m\t$n\t$p\t{g}\t{s:.2f}\t{g*1.073741824/s:.2f}')" >> $OUT; }
prep /mnt/pmem0 0; prep /mnt/pmem1 1
for p in 1 4 8 16; do run /mnt/pmem0 0 $p; run /mnt/pmem1 1 $p; done
for p in 8 16; do run /mnt/pmem0 1 $p; run /mnt/pmem1 0 $p; done
# both sockets at once, each local, 16 readers per mount
t0=$(date +%s.%N); (run /mnt/pmem0 0 16 &) ; run /mnt/pmem1 1 16; wait; sleep 2
rm -rf /mnt/pmem0/bw /mnt/pmem1/bw
echo DONE >> $OUT
