#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p preprocess_output

# Launch 8 jobs, one for each node (Total 64 GPUs)
# Each node processes 8 consecutive files (64 total files / 8 nodes = 8 files per node)
for node_id in {0..7}; do
    # Calculate the starting file number for this node
    start_file=$((node_id * 8))
    
    echo "Launching node $node_id with files merge_${start_file}.txt to merge_$((start_file + 7)).txt"
    
    sbatch --job-name=mg-vpt-pre-${node_id} \
           --output=preprocess_output/mg-node-${node_id}.out \
           --error=preprocess_output/mg-node-${node_id}.err \
           $(pwd)/examples/training/finetune/MatrixGame2.0/preprocess_worker.slurm $start_file $node_id
done

echo "All 8 nodes (64 GPUs) launched successfully!"
