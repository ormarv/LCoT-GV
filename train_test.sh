#!/bin/bash
#SBATCH --job-name=traintest
#SBATCH --output=<your_logfile_dir>/%x_%j.out
#SBATCH --error=<your_logfile_dir>/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Load your environment

# Change the file permission
chmod +x src/train_test.py

# Run the Python script
srun src/train_test.py train test -v -g "<your_graphs_dir>" -f nb_parents nb_children node_index distance_to_end nb_words_before nb_nodes_per_depth -d <your_training_results_dir> -a "text"

echo "Job ended at: $(date)"
