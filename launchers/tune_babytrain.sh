#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10
#SBATCH --time=47:58:58

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote


python main.py runs/train-superdataset-test-4/ tune \
-p X.SpeakerDiarization.DATASET \
--model_path runs/train-superdataset-test-4/checkpoints/last.ckpt \
-nit 50 \
--classes dataset \
--metric fscore
