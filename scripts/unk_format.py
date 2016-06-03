import codecs
import sys
import re

if len(sys.argv) != 3:
    print("format: <input file> <output file name>")
    sys.exit()

input_file_name = str(sys.argv[1])
output_file_name = str(sys.argv[2])
input_file = codecs.open(input_file_name,'r','utf-8')
output_file = codecs.open(output_file_name,'w','utf-8')

for line in input_file:
	re.sub('\n','',line)
	line = line.split(' ')
	if line[0]=="<START>":
		del line[0]
		del line[-1]
		output_file.write(' '.join(line)+'\n')

