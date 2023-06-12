#!/bin/bash
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=10
#SBATCH --time=47:58:58

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote


python main.py runs/train-superdataset-test-4/ train \
-p X.SpeakerDiarization.DATASET \
--classes dataset \
--model_type pyannet \
--epoch 200
