#!/bin/bash
#SBATCH --chdir /home/perello/cs302-lab-3
#SBATCH --partition=gpu
#SBATCH --qos=cs-302
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 1G
#SBATCH --account cs-302
#SBATCH --output=/scratch/izar/perello/cs302-lab-3/sweep-%j.out
#SBATCH --error=/scratch/izar/perello/cs302-lab-3/sweep-%j.err

module load gcc cuda

echo "Starting matrix size sweep at $(date)"
echo "----------------------------------------"

# Compile the code
make all

# Array of matrix sizes to test
sizes=(16 64 256 1024 4096)

# Run tests for each size
for size in "${sizes[@]}"; do
    echo "Testing matrix size: ${size}x${size}x${size}"
    echo "----------------------------------------"
    ./assignment3 $size $size $size 0
    echo "----------------------------------------"
done

echo "Finished matrix size sweep at $(date)" 