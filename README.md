# pyannote-vtc-testing
Testing scripts for pyannote VTC

## Setup

This requires the pyannote version on branch `develop` of the repo 
`git@github.com:bootphon/pyannote-audio/`.

```shell
git clone git@github.com:hadware/pyannote-vtc-testing/
cd pyannote-vtc-testing/
# create a venv (tested on python3.8)
python3.8 -m venv venv/
. venv/bin/activate
pip install -r requirements.txt
```

## Usage

The `main.py` script does all you need. It is used to:
- train a model on a given dataset's train-set
- tune the pipeline's hyperparameters on the dataset's dev-set
- apply the tuned pipeline on a dataset's test-set
- score the test-set's inference files with either IER or average Fscore

Run `python main.py -h` to get help or look at the launchers script to get an 
idea of the arguments for each command.