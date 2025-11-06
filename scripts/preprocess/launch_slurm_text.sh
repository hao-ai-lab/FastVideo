#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p preprocess_output_text

# Launch 8 jobs, one for each node
# Each node processes 8 consecutive files (64 total files / 8 nodes = 8 files per node)
for node_id in {0..7}; do
    # Calculate the starting file number for this node
    start_file=$((node_id * 8 + 1))
    
    echo "Launching text-only node $node_id with files v2m_${start_file}.txt to v2m_$((start_file + 7)).txt"
    echo "sbatch --job-name=text-${node_id} --output=preprocess_output_text/preprocess-text-node-${node_id}.out --error=preprocess_output_text/preprocess-text-node-${node_id}.err scripts/preprocess/syn_text.slurm $start_file $node_id"
    
    sbatch --job-name=text-${node_id} \
           --output=preprocess_output_text/preprocess-text-node-${node_id}.out \
           --error=preprocess_output_text/preprocess-text-node-${node_id}.err \
           scripts/preprocess/syn_text.slurm $start_file $node_id
done

echo "All 8 text-only nodes launched successfully!" 
