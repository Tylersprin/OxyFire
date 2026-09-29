#!/bin/bash
#SBATCH --job-name=OxyFire
#SBATCH --output=OxyFire_%j.out
#SBATCH --error=OxyFire_%j.err
#SBATCH --time=04:00:00               # Estimated time limit (HH:MM:SS)
#SBATCH --partition=gpu              # Name of the GPU partition on your cluster (check your HPC docs)
#SBATCH --gres=gpu:1                 # Request 1 GPU (change to 2 if multi-GPU)
#SBATCH --cpus-per-task=8            # Number of CPU cores for data loading
#SBATCH --mem=32G                    # RAM allocation
#SBATCH --nodes=1

module load GCC/12.3.0
module load OpenMPI/4.1.5

echo "Job started on $(hostname) at $(date)"
python YOLOTrainer.py
echo "Job finished at $(date)"