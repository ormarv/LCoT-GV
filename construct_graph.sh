#!/bin/bash
#SBATCH --job-name=graphconstruction
#SBATCH --output=<your_logfile_dir>/%x_%j_%a.out
#SBATCH --error=<your_logfile_dir>/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --array=<your_job_nb>
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Load your environment

DATA_DIR="<your_data_dir>"
MAX_NLI_CONTEXT=5

FILE="$(awk -v S="$SLURM_ARRAY_TASK_ID" 'FNR==S {print $1}' graph_production_array.txt)"

chmod +x src/build_graphs.py
srun src/build_graphs.py -f $FILE -d $DATA_DIR

echo "Job finished at: $(date)"
