#!/bin/bash
#SBATCH -N 1                    # Request 1 node
#SBATCH -c 8                    # Request 8 CPU cores 
#SBATCH --ntasks-per-node=1     # Run 1 task per node 
#SBATCH -t 03:00:00             # Max runtime 
#SBATCH --gres=gpu:H200:2       # Request 2 H200s GPUs
#SBATCH --mem-per-gpu=142G      # Request 142GB memory per GPU
#SBATCH -J hyperparam_tune_model        # Job name
#SBATCH -o /home/hice1/%u/scratch/cs7643/deepfake-detection/slurm_outs/%x_%j.out # Output file for stdout and stderr

mkdir -p /home/hice1/$USER/scratch/cs7643/deepfake-detection/slurm_outs

module load anaconda3/2023.03

echo "Setting up environment paths..."
export CONDA_ENV_PATH="/home/hice1/$USER/scratch/cs7643"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.12/site-packages:$PYTHONPATH"

# Activate the Conda environment
echo "Activating Conda environment..."
source activate /home/hice1/$USER/scratch/cs7643

cd /home/hice1/$USER/scratch/cs7643/deepfake-detection

echo "Starting training script..."
export PYTHONUNBUFFERED=TRUE
python hyperparam_tune.py 