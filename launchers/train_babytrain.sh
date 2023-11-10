#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10
#SBATCH --time=47:58:58

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote-new


python /scratch2/sdas/modules/pyannote-vtc-testing/main.py runs/train-superdataset-test-2_4-1/ train \
-p X.SpeakerDiarization.DATASET \
--classes babytrain \
--model_type pyannet \
--epoch 100
