#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=puck5
#SBTACH --gres=gpu:rtx8000:1

# loading modules and activating the right conda env
source ~htiteux/.bashrc
module load cuda/10.0 anaconda
conda activate pyannote-vtc-v2

python main.py runs/basal_voice/ train \
-p BasalVoice.SpeakerDiarization.InterviewDiarizationProtocol \
--classes basalvoice \
--model_type pyannet \
--epoch 200
