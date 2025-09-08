#!/bin/bash

counter=0
# for pair in "1e-5 1e-5" "1e-5 8e-6" "1e-5 6e-6" "1e-5 4e-6" "1e-5 2e-6" "1e-5 1e-6"; do
# for pair in "1e-5 8e-6" "1e-5 4e-6" "1e-5 2e-6" "8e-6 8e-6" "8e-6 6e-6" "8e-6 2e-6"; do
for pair in "1e-5 8e-6"; do
# for pair in "2e-6 4e-7" "2e-6 6e-7" "4e-6 6e-7" "4e-6 8e-7" "4e-6 1e-6" "6e-6 4e-7" "6e-6 6e-7" "6e-6 8e-7" "6e-6 1e-6"; do
# for pair in "2e-6 4e-7" "2e-6 6e-7" "4e-6 6e-7" "4e-6 8e-7" "4e-6 1e-6" "6e-6 4e-7" "6e-6 6e-7" "6e-6 8e-7" "6e-6 1e-6"; do
    port=$((29500 + counter))
    read -r lr critic_lr <<< "$pair"
    echo "$lr $critic_lr"
    echo "sbatch --job-name=sf-${lr}-c${critic_lr} --output=sf_output/sf-${lr}-c${critic_lr}_%j.out --error=sf_output/sf-${lr}-c${critic_lr}_%j.err examples/distill/SFWan2.1-T2V/distill_dmd_t2v_1.3B.slurm $port $lr $critic_lr"
    sbatch --job-name=n-${lr}-c${critic_lr} --output=sf_output/sf-${lr}-c${critic_lr}.out --error=sf_output/sf-${lr}-c${critic_lr}.err examples/distill/SFWan2.1-T2V/distill_dmd_t2v_1.3B.slurm $port $lr $critic_lr
    ((counter++))
done
#     port=$((29500 + counter))
    
#     echo "sbatch --job-name=sf-${i} --output=sf_output/sf-${i}_%j.out --error=sf_output/sf-${i}_%j.err distill_sf_1_3B.slurm 7 $port $i"
#     sbatch --job-name=sf-${i} --output=sf_output/sf-${i}_%j.out --error=sf_output/sf-${i}_%j.err distill_sf_1_3B.slurm $i $port 5
    
#     ((counter++))
# done