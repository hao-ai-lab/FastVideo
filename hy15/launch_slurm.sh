#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p preprocess_output

# Launch 8 jobs, one for each node
# Each node processes 8 consecutive files (64 total files / 8 nodes = 8 files per node)
for node_id in {0..7}; do
    # Calculate the starting file number for this node
    start_file=$((node_id * 4 + 1))
    
    echo "Launching node $node_id with files v2m_${start_file}.txt to v2m_$((start_file + 3)).txt"
    echo "sbatch --job-name=ode-${node_id} --output=preprocess_output/preprocess-node-${node_id}.out --error=preprocess_output/preprocess-node-${node_id}.err slurms/syn.slurm $start_file $node_id"
    
    sbatch --job-name=ode-${node_id} \
           --output=preprocess_output/preprocess-node-${node_id}.out \
           --error=preprocess_output/preprocess-node-${node_id}.err \
           /home/hal-weiz/FastVideo/hy15/syn.slurm $start_file $node_id
done

echo "All 8 nodes launched successfully!"