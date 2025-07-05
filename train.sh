# !/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.001 --n_sit 1 --n_txpcnn 7  --dataset eth --tag sit-eth --use_lrschd --num_epochs 250 && echo "eth Launched." &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.001 --n_sit 1 --n_txpcnn 7  --dataset hotel --tag sit-hotel --use_lrschd --num_epochs 250 && echo "hotel Launched." &
P1=$!
CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.001 --n_sit 1 --n_txpcnn 7  --dataset univ --tag sit-univ --use_lrschd --num_epochs 250 && echo "univ Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.001 --n_sit 1 --n_txpcnn 7  --dataset zara1 --tag sit-zara1 --use_lrschd --num_epochs 250 && echo "zara1 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.001 --n_sit 1 --n_txpcnn 7  --dataset zara2 --tag sit-zara2 --use_lrschd --num_epochs 250 && echo "zara2 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.001 --n_sit 1 --n_txpcnn 7  --dataset zara2 --tag sdd --use_lrschd --num_epochs 250 && echo "sdd Launched." &
P4=$!

wait $P0 $P1 $P2 $P3 $P4
