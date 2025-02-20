#!/bin/bash

BATCH_SIZE=128
INPUT_SIZE=128
HIDDEN_SIZE=128
OUTPUT_SIZE=128
OUTPUT_DIR="sparse_mt_results"

mkdir -p "$OUTPUT_DIR"

for w1 in $(seq 0.0 0.2 1.0); do
    OUTPUT_FILE="${OUTPUT_DIR}/flops_w1_${w1}_w2_${w1}.txt"
    python3 ${TORCHSIM_DIR}/tests/test_spmm_scheduler.py \
        --batch_size $BATCH_SIZE \
        --input_size $INPUT_SIZE \
        --hidden_size $HIDDEN_SIZE \
        --output_size $OUTPUT_SIZE \
        --w1_sparsity $w1 \
        --w2_sparsity $w1 > "$OUTPUT_FILE" &
    echo "Started: w1=$w1, w2=$w2 (Output: $OUTPUT_FILE)"
done
wait
echo "All processes completed!"
