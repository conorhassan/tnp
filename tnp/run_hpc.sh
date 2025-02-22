#!/bin/bash
#SBATCH --job-name=test_run 
#SBATCH --output=%x-%j.out 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32G 
#SBATCH --gpus=2 
#SBATCH --time=00:20:00

module purge
module load scicomp-python-env 

export WRKDIR="/scratch/work/$hassanc1"

VENV_PATH="$WRKDIR/myenv"

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
else
    source $VENV_PATH/bin/activate
fi

python old_nnx_train_script.py --config-path configs --config-name config

deactivate

