# we have a simple task, translate number into letters

# first run generate_fsa.py to generate the fsa file and training and validation data set

python generate_fsa.py

# it will generate the following: 

# source.train.txt : 6 number sentences.
# target.train.txt : 6 letter sentences.

# source.valid.txt : 2 number sentences.
# target.valid.txt : 2 letter sentences.

# and a simple fsa file which forces the output to have following words:
# ["lstm","is","great","slow","luckily","we","make","it","fast","enough","and","with","fsa"]

EXEC=../../../executable/ZOPH_RNN_XING

# [Train] train the translation model 
$EXEC -t source.train.txt target.train.txt model.nn -L 15 -H 40 -N 1 -M 0 1 -B best.nn -a source.valid.txt target.valid.txt -d 0.5 -P -0.05 0.05 -w 5 -m 20 -n 20 -l 1 -A 0.8

# [Decode] decode the top 10 for the source.valid.txt
$EXEC -k 10 best.nn kbest.txt --print-score 1 -b 20 --decode-main-data-files source.valid.txt

# [Decode + Fsa] decode the top 10 with fsa integration
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --decode-main-data-files source.valid.txt

# [Decode + Fsa + Beam Info] To see the beam cells during decoding: --print-beam 1
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt

# [Decode + Fsa + Beam Info + encourage-list + repeat-penalty + adjacent-repeat-penalty + alliteration + wordlen ]
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt --repeat-penalty -1.0 --adjacent-repeat-penalty -1.0 --alliteration-weight 1.0 --wordlen-weight 1.0 --encourage-list enc1.txt enc2.txt --encourage-weight 1.0,-1.0

# [Interactive mode] : --interactive 1
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt --repeat-penalty -1.0 --adjacent-repeat-penalty -1.0 --interactive 1
# it will print the following commend:
# Please input k:<k> source_file:<source_file> fsa_file:<fsa_file> repetition:<repetition_weight> alliteration:<alliteration_weight> wordlen:<wordlen_weight> encourage_list_files:<file1>,<file2> encourage_weights:<weight1>,<weight2>
# Note:
# <repetition> is the same as --repeat-penalty, and it will add these two weights;
# the command line should contains --fsa <fsa_file> and --decode-main-data-files <source_file>, both fsa_file and source_file should exist and are valid fsa_file and source file, although you don't really use them in the interactive mode.

# [Interactive-line mode] : --interactive 1 --interactive-line 1
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt --interactive-line 1 --interactive 1

# 1. `source <source_file>` : process the source-side forward propagation.
# 2. `words word1 word2 word3` feed the target-side RNN with words sequence `word1 owrd2 word3`. This is supposed to be the line that human composed. 
# 3. `fsaline <fsa_file> encourage_list_files:enc1.txt,enc2.txt encourage_weights:1.0,-1.0 repetition:0.0 alliteration:0.0 wordlen:0.0` Let the RNN to continue decode with FSA.


# [Interactive-line mode + ensemble ] : --interactive 1 --interactive-line 1
$EXEC -k 10 best.nn best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt source.valid.txt --interactive-line 1 --interactive 1

# 1. `source <source_file>` : process the source-side forward propagation.
# 2. `words word1 word2 word3` feed the target-side RNN with words sequence `word1 owrd2 word3`. This is supposed to be the line that human composed. 
# 3. `words_ensemble word11 word12 word13 ___sep___ word21 word22 word23 ___sep___` feed the target-side RNN with words sequence `word11 owrd12 word13` for `best.nn.1` and `word21 word22 word23` for `best.nn.2` These are supposed to be the lines human composed. 
# 4. `fsaline <fsa_file> encourage_list_files:enc1.txt,enc2.txt encourage_weights:1.0,-1.0 repetition:0.0 alliteration:0.0 wordlen:0.0` Let the RNN to continue decode with FSA.






