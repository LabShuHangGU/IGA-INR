#!/bin/bash
#使用空格进行分格

export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

CUDA_VISIBLE_DEVICES=9 python run_nerf.py \
    --config configs/hotdog.txt 