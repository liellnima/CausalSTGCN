#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1                             # specify gpu

#SBATCH --cpus-per-task=4                                # specify cpu

#SBATCH --mem=12G                                        # specify memory

#SBATCH --time=00:50:00                                  # set runtime

#SBATCH -o /home/mila/j/julia.kaltenborn/slurm-%j.out        # set log dir to home

COUNTER=$1
epochs=$2
stgcn=$3
tpcnn=$4
lr=$5
lr_scheduler=$6 # must write "true"
k=$7

# fixed for the moment
dataset_num=4

DL_FRAMEWORK="torch"

echo "Beginning experiment $COUNTER for Dataset $dataset_num, with $epochs epochs, $stgcn stgcn, $tpcnn tpcnn, $lr learning rate, $lr_scheduler scheduler, $k kernel_size is concluded."
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

cp -r /network/scratch/j/julia.kaltenborn/CausalSTGCN/ $SLURM_TMPDIR/
# rm -r $SLURM_TMPDIR/caiclone/results/
# cp -r /network/scratch/j/julia.kaltenborn/data/ $SLURM_TMPDIR/

# 6. Set Flags

export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/CausalSTGCN/

# 8. Run Python
echo "Training CausalSTGCN ..."
if [[ $lr_scheduler = "true" ]]
then
  echo "With learning scheduler:"
  python main.py --exp_id $COUNTER --dataset_num $dataset_num --epochs $epochs --kernel_size $k --lr $lr --stgcn $stgcn --tpcnn $tpcnn --store_csv --lr_scheduler
else
  echo "Without learning scheduler:"
  python main.py --exp_id $COUNTER --dataset_num $dataset_num --epochs $epochs --kernel_size $k --lr $lr --stgcn $stgcn --tpcnn $tpcnn --store_csv
fi

# 9. Copy output to scratch

# make the dir you need
dir=/network/scratch/j/julia.kaltenborn/CausalSTGCN/checkpoint/exp_$COUNTER/
if [ -d "$dir" -a ! -h "$dir" ]
then
   echo "Dir exists already"
else
   echo "Make checkpoint dir"
   mkdir /network/scratch/j/julia.kaltenborn/CausalSTGCN/checkpoint/exp_$COUNTER/
fi

cp -r $SLURM_TMPDIR/CausalSTGCN/checkpoint/exp_$COUNTER/* /network/scratch/j/julia.kaltenborn/CausalSTGCN/checkpoint/exp_$COUNTER/
cp -r $SLURM_TMPDIR/CausalSTGCN/tuning/results.csv /network/scratch/j/julia.kaltenborn/CausalSTGCN/checkpoint/exp_$COUNTER/


# 10. Experiment is finished
echo "Experiment $COUNTER for Dataset $dataset_num, with $epochs epochs, $stgcn stgcn, $tpcnn tpcnn, $lr learning rate, $k kernel_size is concluded."
