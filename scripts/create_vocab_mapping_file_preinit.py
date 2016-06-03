# -*- coding: utf-8 -*-
import codecs
import sys
from itertools import izip
from collections import defaultdict as dd

if len(sys.argv)!=6:
	print("USAGE: <child source data file> <child target data file> <count threshold> <output model file> <parent source data file>")
	sys.exit()
source_file = codecs.open(sys.argv[1],'r','utf-8') #child source data file 
target_file = codecs.open(sys.argv[2],'r','utf-8') #child target data file
count_threshold = int(sys.argv[3]) #count threshold to unk all words that do not occur this many times
output_model_file = codecs.open(sys.argv[4],'w','utf-8') #output mapping file name
source_file_big = codecs.open(sys.argv[5],'r','utf-8') #parent source data file

source_counts = dd(int)
source_big_counts = dd(int)
target_counts = dd(int)
source_words = set([])
source_words_big = set([])
target_words = set([])

for line_s,line_t in izip(source_file,target_file):
	line_s = line_s.replace('\n','').split(' ')
	line_t = line_t.replace('\n','').split(' ')
	for word in line_s:
		source_counts[word]+=1
	for word in line_t:
		target_counts[word]+=1

for line in source_file_big:
	line = line.replace('\n','').split(' ')
	for word in line:
		source_big_counts[word]+=1

for tup in source_counts:
	if source_counts[tup] >= count_threshold:
		source_words.add(tup)
for tup in target_counts:
	if target_counts[tup] >= count_threshold:
		target_words.add(tup)

print("Number of unique source words above count threshold:",len(source_words))
print("Number of unique target words above count threshold:",len(target_words))

import operator
sorted_big_counts = sorted(source_big_counts.items(), key=operator.itemgetter(1))[::-1][:len(source_words)]

for tup in sorted_big_counts:
	source_words_big.add(tup[0])



index = 1
output_model_file.write('1 1 '+ str(len(target_words)+3) + ' ' + str(len(source_words)+1) +'\n')
output_model_file.write('==========================================================\n')
output_model_file.write('0 <UNK>\n')
for word in source_words_big:
	output_model_file.write(str(index) + ' ' + word + '\n')
	index+=1

index = 3
output_model_file.write('==========================================================\n')
output_model_file.write('0 <START>\n')
output_model_file.write('1 <EOF>\n')
output_model_file.write('2 <UNK>\n')
for word in target_words:
	output_model_file.write(str(index) + ' ' + word + '\n')
	index+=1
output_model_file.write('==========================================================\n')

