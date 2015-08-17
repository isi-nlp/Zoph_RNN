#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -l gpus=2

cd /home/nlg-05/zoph/MT_Experiments/Data/Stride_XX/single_layer_gpu_google_model/
source /usr/usc/cuda/7.0/setup.sh
./a.out