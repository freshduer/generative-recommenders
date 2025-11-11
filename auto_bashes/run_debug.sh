#!/bin/bash

# different sequence lengths for debugging
for seq_len in 128 256 512 1024 1536 2048; do
    echo "Running with uih_seq_len=$seq_len ..."
    LOCAL_WORLD_SIZE=1 WORLD_SIZE=1 \
    python3 generative_recommenders/dlrm_v3/inference/main.py \
        --dataset debug \
        --uih_seq_len $seq_len \
        2>&1 | tee logs/debug/debug_${seq_len}.txt
done


# different batchsize for debugging # no impact..
# for batchsize in 4 8 16 32 64 128; do
#     echo "Running with batchsize=$batchsize ..."
#     LOCAL_WORLD_SIZE=1 WORLD_SIZE=1 \
#     python3 generative_recommenders/dlrm_v3/inference/main.py \
#         --dataset debug \
#         --batchsize $batchsize \
#         2>&1 | tee logs/debug/debug_batch/${batchsize}.txt
# done