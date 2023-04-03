#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J FairMU
#SBATCH -p datasci
#SBATCH --output=results/logs/adult_ns_0.8.out
#SBATCH --mem=24G
#SBATCH --gres=gpu:0
module load python
conda activate torch

for MODEL in lr nn
do
    for MODE in clean random targeted
    do
        for RATIO in 0.1 0.25 0.5
        do
            for RUN in 1 2 3 4 5
            do
                python main.py --mode clean --submode $MODE --dataset bank --batch_size 64 --lr 0.05 --ratio $RATIO --epochs 400 --seed $RUN --model_type $MODEL
            done
        done
    done
done

