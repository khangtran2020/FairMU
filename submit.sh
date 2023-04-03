#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J FairMU
#SBATCH -p datasci
#SBATCH --output=results/logs/adult_ns_0.8.out
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
module load python
conda activate torch

for MODE in clean random target
do
    for RUN in 1 2 3 4 5
    do
        python main.py --mode clean --submode $MODE --dataset bank --batch_size 64 --lr 0.1 --epochs 200 --seed $RUN
    done
done