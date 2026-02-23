#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p preprocess_output

# Launch 8 jobs, one for each node (Total 64 GPUs)
# Each node processes 8 consecutive files (64 total files / 8 nodes = 8 files per node)
for node_id in {0..3}; do
    # Calculate the starting file number for this node
    start_file=$((node_id * 8))
    
    echo "Launching node $node_id with files merge_${start_file}.txt to merge_$((start_file + 7)).txt"
    
    sbatch --job-name=wg-pre-${node_id} \
           --output=preprocess_output/wg-node-${node_id}.out \
           --error=preprocess_output/wg-node-${node_id}.err \
           $(pwd)/FastVideo/examples/training/consistency_finetune/causal_wangame_ode_init/preprocess_worker.slurm $start_file $node_id
done

echo "All 4 nodes (32 GPUs) launched successfully!"
