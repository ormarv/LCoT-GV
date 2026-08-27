
# Long Chain-of-Thought Analysis with GNN

This repository contains code to analyze long chain-of-thought reasoning using Graph Neural Networks.
The sbatch scripts provided need to be filled in depending on the user's system, or replaced entirely if you do not use Slurm.

## Environment Installation

The environment used by the developers was created using Miniforge 24.9.0 and CUDA 12.6.3.

```bash
pip install -r requirements.txt
```

## Pipeline

The pipeline consists of the following steps:

1. **Generate LCoTs and evaluate**
   - Configure model and evaluation tasks:
   ```bash
   sbatch load_datasets_and_generate_lcots.sh
   ```
   - This generates model outputs for reasoning tasks with chain-of-thought generation. 
   - Select the datasets and LRMs of your choice by changing the job numbers in the --array line of the script.

2. **Build graphs**
   ```bash
   sbatch build_graph.sh
   ```
   - This script requires a file called graph\_production\_array.txt. It can be produced with the following command.
   ```bash
   ls -l <your_data_dir>/lcots/ | tail -n +2 | awk '{print $9}' > graph_production_array.txt
   ```
   - This generates one graph for each file in the "your\_data\_dir/lcots" directory.

3. **Train and/or test the GAT model**

   ```bash
   sbatch construct_graph.sh
   ```
   - This can train and test the GAT model on the previously generated graphs.
