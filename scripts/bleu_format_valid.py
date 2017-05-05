#pass the output of the kbest from the RNN toolkit into this to strip it of the extra padding

# The difference between this and bleu_format.py is that: 
# 1. this script will take the refence file as input and generate an temp reference file which have the same lines with the formatted output file. 

# python bleu_format_valid.py <RNN_OUTPUT> <reference file> <reference output file>

import codecs
import sys
import re

input_file_name = str(sys.argv[1])
output_file_name = input_file_name + '.bleu'
input_file = codecs.open(input_file_name,'r','utf-8')
output_file = codecs.open(output_file_name,'w','utf-8')

reference_fn = str(sys.argv[2])
output_reference_fn = str(sys.argv[3])

rf = codecs.open(reference_fn,'r','utf-8')
orf = codecs.open(output_reference_fn,'w','utf-8')

while True:
    line = input_file.readline()
    refline = rf.readline()
    if not line:
        break

    if line.startswith('------'):
        line = input_file.readline()
        line = line.strip()
        if line == "":
            pass
        elif line.startswith("<START>"):
            ll = line.split()
            ll = ll[1:-1]
            output_file.write(" ".join(ll) + '\n')
            orf.write(refline)
            input_file.readline()

output_file.close()
orf.close()
