#!/bin/bash
sbatch ./data_preperation/run_generate_videos_0_100.slurm
sbatch ./data_preperation/run_generate_videos_100_200.slurm
sbatch ./data_preperation/run_generate_videos_200_300.slurm
sbatch ./data_preperation/run_generate_videos_300_400.slurm
sbatch ./data_preperation/run_generate_videos_400_500.slurm
sbatch ./data_preperation/run_generate_videos_500_600.slurm
sbatch ./data_preperation/run_generate_videos_600_700.slurm
sbatch ./data_preperation/run_generate_videos_700_800.slurm
sbatch ./data_preperation/run_generate_videos_800_900.slurm
sbatch ./data_preperation/run_generate_videos_900_1000.slurm