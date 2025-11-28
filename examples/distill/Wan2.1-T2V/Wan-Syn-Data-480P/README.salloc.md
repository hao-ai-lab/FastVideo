

### Standard way to use `salloc + srun`
```bash
salloc -N 8 -G 64 --cpus-per-task=128 --mem=512G --exclusive --partition=lowprio --qos=lowprio
# Then you enter a special space where you can srun and get all the slurm env var
# 1. To check slurm env var try
env | grep SLURM
# 2. To run the srun command, try
bash ./distill_dmd_VSA_t2v_1.3B.salloc
```

### Non-standard way to use `salloc + srun`

In a tmux, run this command to reserve a terminal space
```bash
salloc -N 8 -G 64 --cpus-per-task=128 --mem=512G --exclusive --partition=lowprio --qos=lowprio
```

Get the job id from the output of the `squeue` command:
```bash
squeue -u $USER
```

Now in another terminal, run
```bash
conda activate <your_conda_env>

# Customize the job id, number of nodes, and number of gpus per node
export JOBID=1045228
export NUM_NODES=8
export NUM_GPUS_PER_NODE=8
bash ./distill_dmd_VSA_t2v_1.3B.salloc
```