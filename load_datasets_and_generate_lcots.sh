#!/bin/bash
#SBATCH --job-name=load_and_generate
#SBATCH --output=<your_logfile_dir>/%x_%j.out
#SBATCH --error=<your_logfile_dir>/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -C h100
#SBATCH --array=<your_job_nbs>
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Initiate the environment


# Modify the permissions

chmod +x load_datasets.py

# Get the necessary variables

DATA_DIR="<your_data_dir>"
NB_SAMPLES_PER_FILE=15
HF_DIR="<your_huggingface_cache_dir>"
read -r DATASET LRM N_SAMPLES N_ITERATIONS <<< "$(awk -v S="$SLURM_ARRAY_TASK_ID" 'FNR == S {print $1,$2,$3,$4}' array_lcot_production.txt)"
echo "DATASET $DATASET"
echo "LRM $LRM"
echo "N_SAMPLES $N_SAMPLES"
echo "N_ITERATIONS $N_ITERATIONS"
# Launch the job

srun load_datasets.py -s $DATA_DIR -d $DATASET -m $LRM -l $N_SAMPLES -i $N_ITERATIONS

echo "Job ended at: $(date)"
