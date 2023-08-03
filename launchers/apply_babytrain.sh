#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --time=47:58:58

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate /scratch2/sdas/conda/envs/pyannote


python main.py runs/train-superdataset-test-3-1/ apply \
-p X.SpeakerDiarization.BBT2 \
--model_path runs/train-superdataset-test-3-1/checkpoints/epoch=30-MultiLabelSegmentation-XSpeakerDiarizationDATASET-MultilabelAUROC=0.773830.ckpt \
--classes babytrain \
--apply_folder runs/train-superdataset-test-3-1/apply/ \
--params runs/train-superdataset-test-3-1/best_params.yml
