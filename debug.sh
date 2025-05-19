#!/bin/bash
#SBATCH --chdir /home/perello/cs302-lab-3
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 1G
#SBATCH --account cs-302
#SBATCH --output=/scratch/izar/perello/cs302-lab-3/debug-%j.out
#SBATCH --error=/scratch/izar/perello/cs302-lab-3/debug-%j.err

module load gcc cuda

echo "Starting debug run at $(date)"
echo "----------------------------------------"

# Compile the code
make clean
make all

# Run with small matrix size (16x16x16) and debug mode enabled (1)
echo "Running with 16x16x16 matrix size and debug mode enabled"
echo "----------------------------------------"
./assignment3 16 16 16 1

echo "----------------------------------------"
echo "Finished debug run at $(date)" 