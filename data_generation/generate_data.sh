#!/bin/bash

# main data:
# python graph_generator.py --exp-id 4 --n 1000 --num-features 5 -g 20 -p 0.6 --neighborhood 2 --timewindow 100 --eta 1.5

i=0

for timewindow in 100 500; do
  for p in 0.3 0.6 0.9; do
    echo $i
    python graph_generator.py --exp-id $i --n 1000 --num-features 5 -g 20 -p $p --neighborhood 2 --timewindow $timewindow --eta 1.5
    let i++
    python graph_generator.py --exp-id $i --n 1000 --num-features 5 -g 20 -p $p --neighborhood 5 --timewindow $timewindow --eta 1.5
    let i++
    python graph_generator.py --exp-id $i --n 1000 --num-features 5 -g 100 -p $p --neighborhood 10 --timewindow $timewindow --eta 1.5
    let i++
    python graph_generator.py --exp-id $i --n 1000 --num-features 5 -g 100 -p $p --neighborhood 25 --timewindow $timewindow --eta 1.5
    let i++
  done
done
