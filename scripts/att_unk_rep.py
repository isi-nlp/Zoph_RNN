#Run this with the output of the neural MT program
import sys
import codecs
import re

src_file_name = str(sys.argv[1])
tgt_file_name = str(sys.argv[2]) #This is the output of the NMT system
dict_name = str(sys.argv[3])    #This is from the berkely aligner
tgt_unk_locations = codecs.open(sys.argv[4],'r','utf-8') #output from the unk replacement flag during decoding

##############################################################################################
dict_file = codecs.open(dict_name,'r','utf-8')
src_data = [line.replace('\n','') for line in codecs.open(src_file_name,'r','utf-8')]
tgt_data = [line.replace('\n','') for line in codecs.open(tgt_file_name,'r','utf-8')]
tgt_output_file = codecs.open(tgt_file_name+'.output','w','utf-8')

unk_locations = [[] for i in range(0,len(src_data))]
index = 0
for line in tgt_unk_locations:
	line = line.split(' ')
	del line[-1]
	unk_locations[index] = line
	index+=1


#Load in the dictionary
t_table_lookups = {} #For a source word, look up its max prob word, so store as tuple (word,prob)

ignore = True #set to false once certain line is found
curr_word = ''
for line in dict_file:
	#line = re.sub('[\n]', '', line)
	line = line[:-1] #to remove newline
	line = line.split(' ')
	orig_line = line
	if line[0]=='#' and line[1]=='Translation' and line[2]=='probabilities':
		ignore = False

	if ignore:
		continue
	
	try:
		if line[0] !='':
	 		curr_word = line[0].split('\t')[0]
		else: 	
			line[2] = line[2][:-1]	
			tup = (line[2],float(line[3]))
			if curr_word not in t_table_lookups:
				t_table_lookups[curr_word] = tup
			else:
				if tup[1] > t_table_lookups[curr_word][1]:
					t_table_lookups[curr_word] = tup
	except:
		print(line)
		print(orig_line)

if len(src_data) != len(tgt_data):
	print("ERROR: source and target data are of different lengths")
	sys.exit()


for i in range(0,len(tgt_data)):
	line = tgt_data[i]
	line = line.split(' ')
	src_sent = src_data[i].split(' ')
	final_sent = []

	for j in range(0,len(line)):
		word = line[j]
		if word=='<UNK>':
			source_index = int(unk_locations[i][0])
			source_word = src_sent[source_index]
			final_word = '<UNK>'
			if source_word in t_table_lookups:
				final_word = t_table_lookups[source_word][0]
			else:
				final_word = source_word
			final_sent.append(final_word)
			del unk_locations[i][0]
		else:
			final_sent.append(word)

	tgt_output_file.write(' '.join(final_sent)+'\n')


