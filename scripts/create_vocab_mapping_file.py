import codecs
import sys
from itertools import izip
from collections import defaultdict as dd

if len(sys.argv)!=5:
	print("USAGE: <source data file> <target data file> <count threshold> <output model file>")
	sys.exit()
source_file = codecs.open(sys.argv[1],'r','utf-8') #the source data file
target_file = codecs.open(sys.argv[2],'r','utf-8') #the target data file
count_threshold = int(sys.argv[3]) #all words with less than this count frequency will be replaced by <UNK>
output_model_file = codecs.open(sys.argv[4],'w','utf-8') #the output model file

source_counts = dd(int)
target_counts = dd(int)
source_words = set([])
target_words = set([])

for line_s,line_t in izip(source_file,target_file):
	line_s = line_s.replace('\n','').split(' ')
	line_t = line_t.replace('\n','').split(' ')
	for word in line_s:
		source_counts[word]+=1
	for word in line_t:
		target_counts[word]+=1


for tup in source_counts:
	if source_counts[tup] >= count_threshold:
		source_words.add(tup)
for tup in target_counts:
	if target_counts[tup] >= count_threshold:
		target_words.add(tup)

print("Number of unique source words above count threshold:",len(source_words))
print("Number of unique target words above count threshold:",len(target_words))

index = 1
output_model_file.write('1 1 '+ str(len(target_words)+3) + ' ' + str(len(source_words)+1) +'\n')
output_model_file.write('==========================================================\n')
output_model_file.write('0 <UNK>\n')
for word in source_words:
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

