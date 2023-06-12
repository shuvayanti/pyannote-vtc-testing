#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10
#SBATCH --time=47:58:58

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate pyannote


python main.py runs/train-superdataset-test-4/ apply \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/train-superdataset-test-4/checkpoints/last.ckpt \
--classes dataset \
--apply_folder runs/train-superdataset-test-4/apply/ \
--params runs/train-superdataset-test-4/best_params.yml
