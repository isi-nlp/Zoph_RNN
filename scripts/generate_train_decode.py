import sys
import os

head = """
#!/bin/bash
#PBS -q isi
#PBS -l walltime=300:00:00
#PBS -l gpus=2

PREFIX=__PREFIX__
model_folder=/home/nlg-05/xingshi/lstm/model/$PREFIX/
data_folder=/home/nlg-05/xingshi/lstm/ghdata/Eng_Uzb/
output_folder=/home/nlg-05/xingshi/lstm/syntaxNMT/decode/
EXEC=/home/nlg-05/xingshi/lstm/exec/ZOPH_RNN

PY_FORMAT=/home/nlg-05/xingshi/lstm/single_layer_gpu_google_model/Scripts/bleu_format_valid.py
PERL_BLEU=/home/nlg-05/xingshi/workspace/tools/mosesdecoder/scripts/generic/multi-bleu.perl

SRC_TRN=$data_folder/training.tok.lc.uzb
TGT_TRN=$data_folder/training.tok.lc.eng

SRC_DEV=$data_folder/dev.tok.lc.uzb
TGT_DEV=$data_folder/dev.tok.lc.eng

SRC_TST=$data_folder/test.tok.lc.uzb
TGT_TST=$data_folder/test.tok.lc.eng
OUTPUT=$output_folder/$PREFIX.kbest
REF=$output_folder/$PREFIX.ref
BLEU=$output_folder/$PREFIX.bleu

mkdir $model_folder
cd $model_folder

__cmd__

"""

cmd_train = "$EXEC --logfile HPC_OUTPUT_NEW.txt -a $SRC_DEV $TGT_DEV -t $SRC_TRN $TGT_TRN model.nn -B best.nn -v 50000 -V 25000 --screen-print-rate 300 -N 2 -M 0 0 1 -n 40 -w 5 -L 200 --attention-model true --feed-input true -m 64"
# -A 0.9 -l 0.5 -d 0.5 -H 1000 

cmd_decode = """ $EXEC -k 1 $model_folder/best.nn $OUTPUT --decode-main-data-files $SRC_TST -L 100 -b 12
python $PY_FORMAT $OUTPUT $TGT_TST $REF
perl $PERL_BLEU -lc $REF < $OUTPUT.bleu > $BLEU

"""

def main():
    def A(val):
        return "A{}".format(val), "-A {}".format(val)
    def l(val):
        return "l{}".format(val), "-l {}".format(val)
    def d(val):
        return "d{}".format(val), "-d {}".format(val)
    def H(val):
        return "H{}".format(val), "-H {}".format(val)

    funcs = [H,l,d,A]
    template = [300,0.5,0.5,0.5]
    params = []
    
    _Hs = [300,500,1000]
    _ls = [0.5,1.0]
    _ds = [0.5,0.8]
    
    gen = ((x,y,z) for x in _Hs for y in _ls for z in _ds)
    for _H, _l, _d in gen:
        temp = list(template)
        temp[0] = _H
        temp[1] = _l
        temp[2] = _d
        params.append(temp)
    
    def get_name_cmd(paras):
        name = "Uz_En_"
        cmd = [cmd_train]
        for func, para in zip(funcs,paras):
            n, c = func(para)
            name += n
            cmd.append(c)
            
        name = name.replace(".",'')
        
        cmd = " ".join(cmd)
        return name, cmd

    # train
    for para in params:
        name, cmd = get_name_cmd(para)
        fn = "../sh_ue/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd).replace("__PREFIX__",name)
        f.write(content)
        f.close()

    # decode
    for para in params:
        name, cmd = get_name_cmd(para)
        fn = "../sh_ue/{}.decode.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd_decode).replace("__PREFIX__",name)
        f.write(content)
        f.close()


if __name__ == "__main__":
    main()

