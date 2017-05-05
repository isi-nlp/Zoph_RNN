#pass the output of the kbest from the RNN toolkit into this to strip it of the extra padding

import codecs
import sys
import re

input_file_name = str(sys.argv[1])
output_file_name = input_file_name + '.bleu'
input_file = codecs.open(input_file_name,'r','utf-8')
output_file = codecs.open(output_file_name,'w','utf-8')

for line in input_file:
        re.sub('\n','',line)
        line = line.split(' ')
        if line[0]=="<START>":
                del line[0]
                del line[-1]
                output_file.write(' '.join(line)+'\n')