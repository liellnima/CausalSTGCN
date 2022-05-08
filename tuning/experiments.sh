#!/bin/bash

# LATER different datasets (adapt code!)

# NOW only one dataset, but run each experiment 5 times

# when running this script:
# $1 -> stgcn blocks (0, 2, 10)
# $2 -> tpcnn blocks (0, 2, 10)
# $3 -> counter (0)

counter=$3
repeat=5
epochs=3

for i in $(seq $repeat); do
  for lr in 0.01 0.001; do
    for k in 3 5; do
      python main.py --exp_id $counter --dataset_num 4 --epochs $epochs --kernel_size $k --lr $lr --stgcn $1 --tpcnn $2 --store_csv
      let counter++
      python main.py --exp_id $counter --dataset_num 4 --epochs $epochs --kernel_size $k --lr $lr --lr_scheduler --stgcn $1 --tpcnn $2 --store_csv
      let counter++
    done
  done
done

# other script for setting up stuff on cluster:
# python graph_generator.py --exp-id 4 --n 1000 --num-features 5 -g 20 -p 0.6 --neighborhood 2 --timewindow 100 --eta 1.5

# csv must save: (one row == one exp)
# best metrics (mse, mae, kld) for val and train, training time
# all tuning params used
# all data params
