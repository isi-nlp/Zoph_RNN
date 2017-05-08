#!/usr/bin/env bash
#PBS -q isi
#PBS -l walltime=1:00:00
#PBS -l gpus=2

SCRIPTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT=$SCRIPTDIR/../../

EXEC=$ROOT/executable/ZOPH_RNN_XING

# script

PY_FORMAT=$ROOT/scripts/bleu_format_valid.py
PERL_BLEU=$ROOT/scripts/multi-bleu.perl

# model

model_dir=$ROOT/scripts/translate/Fre_Eng_lc_d_att/
fe_nn_attention=$model_dir/best.nn

# data

data_folder=$ROOT/sample_data/

TGT_TST=$data_folder/test_english.tok.lc
SRC_TST=$data_folder/test_french.tok.lc

EN_DEV=$data_folder/dev_english.txt.tok.lc
FR_DEV=$data_folder/dev_french.txt.tok.lc

EN_TRN=$data_folder/train_english.txt.tok.lc.10k
FR_TRN=$data_folder/train_french.txt.tok.lc.10k

# output

output_folder=$ROOT/scripts/translate/Fre_Eng_lc_d_att/decode
mkdir -p $output_folder

id=FE
LOG=$output_folder/${id}.log
OUTPUT=$output_folder/${id}.kbest
REF=$output_folder/${id}.ref
BLEU=$output_folder/${id}.bleu

# decode

cd $output_folder

$EXEC --decode-main-data-files $SRC_TST -b 12 -L 100 -k 1 $fe_nn_attention $OUTPUT --logfile $LOG

# calculate BLEU

python $PY_FORMAT $OUTPUT $TGT_TST $REF
perl $PERL_BLEU -lc $REF < $OUTPUT.bleu > $BLEU
