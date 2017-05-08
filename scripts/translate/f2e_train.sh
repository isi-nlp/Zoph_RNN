#!/usr/bin/env bash
#PBS -q isi80
#PBS -l walltime=300:00:00
#PBS -l gpus=4

SCRIPTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT=$SCRIPTDIR/../../

EXEC=$ROOT/executable/ZOPH_RNN_XING

echo "$EXEC -h";

$EXEC -h

# script

PY_FORMAT=$ROOT/scripts/bleu_format_valid.py
PERL_BLEU=$ROOT/scripts/multi-bleu.perl

# model

model_dir=$ROOT/scripts/translate/Fre_Eng_lc_d_att/
mkdir -p $model_dir
fe_nn_attention=$model_dir/best.nn
model=$model_dir/model.nn

# data

data_folder=$ROOT/sample_data/

TGT_TST=$data_folder/test_english.tok.lc
SRC_TST=$data_folder/test_french.tok.lc

EN_DEV=$data_folder/dev_english.txt.tok.lc
FR_DEV=$data_folder/dev_french.txt.tok.lc

EN_TRN=$data_folder/train_english.txt.tok.lc.10k
FR_TRN=$data_folder/train_french.txt.tok.lc.10k

# train
LOG=$model_dir/train.log


$EXEC --logfile $LOG -B $fe_nn_attention --screen-print-rate 300 -N 2 -M 0 1 2 --attention-model true --feed-input true -H 1000 -a $FR_DEV $EN_DEV -t $FR_TRN $EN_TRN $model -v 200000 -V 40000 -L 100 -n 8 -A 1 --fixed-halve-lr 6 -l 0.35 -d 0.8 -m 128 -w 5
