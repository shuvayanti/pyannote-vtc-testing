# pyannote-vtc-testing
Testing scripts for pyannote VTC

## Setup

First of all, create your environment and install pyannote audio :
```shell
conda create -n pyannote python=3.8
conda activate pyannote
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch

git clone https://github.com/pyannote/pyannote-audio.git
cd pyannote-audio
pip install -e .
```

Then, install this experimental repository :
```shell
git clone https://github.com/shuvayanti/pyannote-vtc-testing.git
```

Make sure you have a `database.yml` file in `~/.pyannote`.


## Usage

The `main.py` script does all you need. It is used to:
- train a model on a given dataset's train-set
- tune the pipeline's hyperparameters on the dataset's dev-set
- apply the tuned pipeline on a dataset's test-set
- score the test-set's inference files with either IER or average Fscore

Run `python main.py -h` to get help or look at the launchers script to get an 
idea of the arguments for each command.
