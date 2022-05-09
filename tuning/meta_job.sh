#!/bin/bash

# counter=$1
# epochs=$2
# stgcn=$3
# tpcnn=$4
# lr=$5
# lr_scheduler=$6 # must write "true"
# k=$7
counter=0
repeat=1 #5
epochs=500

for i in $(seq $repeat); do
  for lr in 0.01 0.001; do
    for k in 3 5; do
      for tpcnn in 0 2 10; do
        for stgcn in 0 2 10; do
          sbatch --partition=long tuning/job.sh $counter $epochs $stgcn $tpcnn $lr "true" $k
          let counter++
          sbatch --partition=long tuning/job.sh $counter $epochs $stgcn $tpcnn $lr "false" $k
          let counter++
        done
      done
    done
  done
done
