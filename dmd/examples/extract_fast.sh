#!/usr/bin/env bash
# Run `dmd extract` sharded across all 4 GPUs with the BN48k sweet-spot settings.
# Args: <input-dir> <minutiae-dir> <output-dir>
set -euo pipefail

GPUS=(0 1 2 3)
CORES_PER_GPU=8
BATCH_SIZE=32

TMP=$(mktemp -d) && trap "rm -rf $TMP" EXIT

( cd "$1" && find . -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
       -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
    | sed 's|^\./||' | sort ) > "$TMP/all.txt"

N=$(wc -l < "$TMP/all.txt")
PER=$(( (N + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
split -d -l "$PER" --suffix-length=2 "$TMP/all.txt" "$TMP/shard_"

for i in "${!GPUS[@]}"; do
    SHARD="$TMP/shard_$(printf '%02d' "$i")"
    [[ -f "$SHARD" ]] || continue
    CORE_START=$(( i * CORES_PER_GPU ))
    CORE_END=$(( CORE_START + CORES_PER_GPU - 1 ))
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
    OMP_NUM_THREADS="$CORES_PER_GPU" \
    MKL_NUM_THREADS="$CORES_PER_GPU" \
    taskset -c "$CORE_START-$CORE_END" \
        dmd extract \
            --input-dir "$1" --minutiae-dir "$2" --output-dir "$3" \
            --filter-list "$SHARD" --device cuda:0 --batch-size "$BATCH_SIZE" &
done
wait
