#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1                             # specify gpu

#SBATCH --cpus-per-task=4                                # specify cpu

#SBATCH --mem=12G                                        # specify memory

#SBATCH --time=00:50:00                                  # set runtime

#SBATCH -o /home/mila/j/julia.kaltenborn/slurm-%j.out        # set log dir to home

EPOCHS=$1
BATCH_SIZE=$2
LEARNING_RATE=$3
MODEL=$4
DL_FRAMEWORK="torch"

echo "Beginning experiment with $EPOCHS epochs, $BATCH_SIZE batch size, $LEARNING_RATE learning rate and $MODEL model."

# 1. Load Python

module load python/3.7


# 2. Load DL Framework

if [[ $DL_FRAMEWORK == "torch" ]]; then

    #module load python/3.7/cuda/10.2/cudnn/7.6/pytorch/1.5.1
    module load python/3.7/cuda/11.1/cudnn/8.0/pytorch/1.8.1

elif [[ $DL_FRAMEWORK == "tf" ]]; then

    module load cuda/10.0/cudnn/7.6
    module load tensorflow/1.15

fi


# 3. Create or Set Up Environment

if [ -a env/bin/activate ]; then

    source env/bin/activate

else

    python -m venv env
    source env/bin/activate
    pip install -U pip wheel setuptools

fi


# 4. Install requirements.txt if it exists

if [ -a requirements.txt ]; then

    pip install -r requirements.txt

fi


# 5. Copy data and code from scratch to $SLURM_TMPDIR/

cp -r /network/scratch/j/julia.kaltenborn/caiclone/ $SLURM_TMPDIR/
rm -r $SLURM_TMPDIR/caiclone/results/
cp -r /network/scratch/j/julia.kaltenborn/data/ $SLURM_TMPDIR/

# 6. Set Flags

export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/caiclone/

# 8. Run Python

echo "Running python caiclone_predictor.py ..."
python caiclone_predictor.py -m $MODEL -e $EPOCHS -b $BATCH_SIZE -l $LEARNING_RATE -i $SLURM_TMPDIR/data/precursor_vort_64x64.npy -t $SLURM_TMPDIR/data/labels_vort_64x64.npy


# 9. Copy output to scratch
cp -r $SLURM_TMPDIR/caiclone/results/* /network/scratch/j/julia.kaltenborn/caiclone/results/


# 10. Experiment is finished

echo "Experiment with $EPOCHS epochs, $BATCH_SIZE batch size, $LEARNING_RATE learning rate and $MODEL model is concluded."
