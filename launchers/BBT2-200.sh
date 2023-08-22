#!/bin/bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=10
#SBATCH --time=47:59:58

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate /scratch2/sdas/conda/envs/pyannote

echo "Training started"

python main.py runs/babytrain-130-2/ train \
-p X.SpeakerDiarization.BBT2 \
--classes babytrain \
--model_type pyannet \
--epoch 130

echo "Tuning started"

python main.py runs/babytrain-130-2/ tune \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/babytrain-130-2/checkpoints/last.ckpt \
-nit 50 \
--classes babytrain \
--metric fscore

echo "Generating inference"

python main.py runs/babytrain-130-2/ apply \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/babytrain-130-2/checkpoints/last.ckpt \
--classes babytrain \
--apply_folder runs/babytrain-130-2/apply/ \
--params runs/babytrain-130-2/best_params.yml

echo "Score"

python main.py runs/babytrain-130-2/ score \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/babytrain-130-2/checkpoints/last.ckpt \
--classes babytrain \
--metric fscore \
--apply_folder runs/babytrain-130-2/apply/ \
--report_path runs/babytrain-130-2/results/fscore.csv

echo "All done"