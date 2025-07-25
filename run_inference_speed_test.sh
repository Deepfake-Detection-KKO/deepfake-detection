#!/bin/bash
#SBATCH -N 1                    # Request 1 node
#SBATCH -c 8                    # Request 8 CPU cores
#SBATCH --ntasks-per-node=1     # Run 1 task per node
#SBATCH -t 00:45:00             # Max runtime (reduced as inference is faster)
#SBATCH --gres=gpu:H200:2       # Request 2 H200s GPUs (keeping as per your request)
#SBATCH --mem-per-gpu=142G      # Request 142GB memory per GPU
#SBATCH -J inference_speed_test # Job name changed
#SBATCH -o /home/hice1/%u/scratch/cs7643/deepfake-detection/slurm_outs/%x_%j.out # Output file for stdout and stderr

# Create the output directory if it doesn't exist
mkdir -p /home/hice1/$USER/scratch/cs7643/deepfake-detection/slurm_outs

# Load necessary modules
module purge
module load anaconda3/2023.03 
module load cuda/11.8        

echo "Setting up environment paths..."
# Assuming your conda environment 'deepfake-detection' is directly under /home/hice1/$USER/scratch/cs7643/deepfake-detection/
# If your environment is named differently or located elsewhere (e.g., in /home/hice1/$USER/.conda/envs/), adjust this.
export CONDA_ENV_PATH="/home/hice1/$USER/scratch/cs7643"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.12/site-packages:$PYTHONPATH"

# Activate the Conda environment
echo "Activating Conda environment..."
source activate /home/hice1/$USER/scratch/cs7643

# Navigate to the directory containing your Python script
cd /home/hice1/$USER/scratch/cs7643/deepfake-detection

echo "Starting inference speed test script..."
export PYTHONUNBUFFERED=TRUE
python inference_speed_test.py 

echo "Job finished!"