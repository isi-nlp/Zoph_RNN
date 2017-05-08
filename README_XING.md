# Decoding with Finate State Acceptor (FSA), Locality Sensitive Hashing (LSH) and Word Alignemnt (WA)

This is the description for additional features built on top of Zoph\_RNN by Xing Shi.
Please contact [Xing Shi](http://xingshi.me)(shixing19910105@gmail.com) for any questions

All the following papers are based on this code:

1. [Why Neural Translations are the Right Length](http://xingshi.me/data/pdf/EMNLP2016short.pdf)
2. [Does String-Based Neural MT Learn Source Syntax?](http://xingshi.me/data/pdf/EMNLP2016long.pdf)
3. [Generating Topical Poetry](http://xingshi.me/data/pdf/EMNLP2016poem.pdf) (for FSA decoding)
4. [Hafez: an Interactive Poetry Generation System](http://xingshi.me/data/pdf/ACL2017demo.pdf) (for FSA decoding)
5. [Speeding up Neural Machine Translation Decoding by Shrinking Run-time Vocabulary](http://xingshi.me/data/pdf/ACL2017short.pdf) (for LSH and WA decoding)

# Instructions for compilation/using the code
The source code is provided in `src/` directory. You can compile the code into a standalone executable by:

```
bash scripts/compile.xing.sh
```

The executable `ZOPH_RNN_XING` will appear in the root folder. A pre-compiled executable `ZOPH_RNN_XING` is in folder `executable\`.

You should set the 7 environment variables and have the required libraries mentioned in `README.md` before compilation.
Two additional requirements:

* CUDA version greater than 8.0
* gcc version greater than 4.9.3

# Decoding with FSA

This section will describe how you can constrain the RNN decoding with a FSA so that the every output will be accepted by the FSA.

The FSA file format should follow the format defined by [Carmel](http://www.isi.edu/licensed-sw/carmel/).

All the script and sampled data file related with this section are in folder `scripts/fsa/`.

## Example data preparation

We will focus on a simple task : Translate numbers into letters, and use a fsa file to force the output to have following words: `["lstm","is","great","slow","luckily","we","make","it","fast","enough","and","with","fsa"]`

To generate fsa file and coresponding train and dev set:

```
cd scripts/fsa/
python generate_fsa.py 
```

It will generate the following:

* source.train.txt : 6 number sentences.
* target.train.txt : 6 letter sentences.
* source.valid.txt : 2 number sentences.
* target.valid.txt : 2 letter sentences.
* fsa.txt

Here, we define `EXEC=../../../executable/ZOPH_RNN_XING`.

### [Train] Train the translation model

```
$EXEC -t source.train.txt target.train.txt model.nn -L 15 -H 40 -N 1 -M 0 1 -B best.nn -a source.valid.txt target.valid.txt -d 0.5 -P -0.05 0.05 -w 5 -m 20 -n 20 -l 1 -A 0.8
```

### [Decode] decode the top 10 for the source.valid.txt

```
$EXEC -k 10 best.nn kbest.txt --print-score 1 -b 20 --decode-main-data-files source.valid.txt
```

## Batch Mode
`Batch Mode` means we will translate all the sentences in the `source.valid.txt` file with the same FSA file `fsa.txt`

### [Decode + FSA] Decode the top 10 with fsa

```
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --decode-main-data-files source.valid.txt
```

### [Decode + FSA + Beam Info] 
To see the beam cells during decoding, use flag `--print-beam 1`
```
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt
```

### [Decode + Fsa + Beam Info + encourage-list + repeat-penalty + adjacent-repeat-penalty + alliteration + wordlen ]

Beside FSA, we also provide several other weights to control the output:

1. Encourage Lists and Encouange Weights: Each encourage list is a file contains either a list of words (like `enc1.txt`) or a list of words with weights (like `enc2.txt`). You feed the encourange lists by flag `--encourage-list enc1.txt enc2.txt` For each encourage list, you should also assign a weight by flag `--encourage-weight 1.0,-1.0`.
2. Repeat penalty: to prevent producing repeated words during decoding. Use flag `--repeat-penalty -1.0`.
3. Adjacent repeat penalty: to prevent producing consectuive repeated wrods. Use flag `--adjacent-repeat-penalty -1.0`.
4. Alliteration: Use flag `--alliteration-weight 1.0`.
5. Word length weight: Use flag `--wordlen-weight 1.0`.

Please refer the paper [Hafez: an Interactive Poetry Generation System](http://xingshi.me/data/pdf/ACL2017demo.pdf) for detailed description of these style controls.

Put all the flags together, we got:

```
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt --repeat-penalty -1.0 --adjacent-repeat-penalty -1.0 --alliteration-weight 1.0 --wordlen-weight 1.0 --encourage-list enc1.txt enc2.txt --encourage-weight 1.0,-1.0
```

To reproduce the results in [Hafez: an Interactive Poetry Generation System](http://xingshi.me/data/pdf/ACL2017demo.pdf), please also checkout the [Rhyme Generation code](https://github.com/Marjan-GH/Topical_poetry) and [Web Interface code](https://github.com/shixing/poem).


## Interactive Mode

Usually, you want to decode different sentence with different FSA. With `Batch Mode`, you'll have to create a `soruce.txt` and reload the whole RNN model from disk for each sentence, which can takes 1-2 minutes.

Thus, we provide the `Interactive Mode`, where you can provide different `source.txt` and `fsa.txt` without reloading the RNN model.

To enable `Interactive Mode`, use flag `--interactive 1`

```
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt --repeat-penalty -1.0 --adjacent-repeat-penalty -1.0 --interactive 1
```
Once loaded the RNN model, it will print the following instructions:
```
Please input k:<k> source_file:<source_file> fsa_file:<fsa_file> repetition:<repetition_weight> alliteration:<alliteration_weight> wordlen:<wordlen_weight> encourage_list_files:<file1>,<file2> encourage_weights:<weight1>,<weight2>
```
Follow the instruction and input the following in STDIN:
```
k:1 soruce_file:source.single.txt fsa_file:fsa repetition:-1.0 alliteration:1.0 wordlen:1.0 encourage_list_files:enc1.txt,enc2.txt encourage_weights:1.0,-1.0
```
It will decode the print the results in STDOUT. Then you can type another commend into STDIN to decode another sentence. NOTE: `source.single.txt` should contains only one sentence. 

## Interactive Line Mode

This mode is specially for the interactive poem generation task where human and machine compose a poem line by line alternatively. Use flag `--interactive-line 1` to enable this mode.

```
$EXEC -k 10 best.nn kbest_fsa.txt --print-score 1 -b 5 --fsa fsa.txt --print-beam 1 --decode-main-data-files source.valid.txt --interactive-line 1 --interactive 1
```
You can choose one of the following three commend to type in STDIN:

1. `source <source_file>` : process the source-side forward propagation.
2. `words word1 word2 word3` feed the target-side RNN with words sequence `word1 owrd2 word3`. This is supposed to be the line that human composed. 
3. `fsaline <fsa_file> encourage_list_files:enc1.txt,enc2.txt encourage_weights:1.0,-1.0 repetition:0.0 alliteration:0.0 wordlen:0.0` Let the RNN to continue decode with FSA.

Both step 2 and 3 will start from the previous hidden states and cell states of target-side RNN.

# Decoding with Word Alignment

Suppose we are translating from French to English, we could use the word alignment information to speed up the decoding. Please find details in 5. [Speeding up Neural Machine Translation Decoding by Shrinking Run-time Vocabulary](http://xingshi.me/data/pdf/ACL2017short.pdf).

The commend to decode with word alignment information is:

```
$EXEC --decode-main-data-files $SRC_TST -b 12 -L 200  -k 1 $fe_nn_attention $OUTPUT  --target-vocab-shrink 2 --f2e-file $ALIGNMENT_FILE --target-vocab-cap $NCAP
```
where `$ALIGNMETN_FILE` is the file that contains word alignment information with the following format:
```
<Source Word Id> <Target Candidate Word Id 1> <Target Candidate Word Id 1> ... <Target Candidate Word Id 10>
```
Each line starts with `<Source Word Id>` and follows with `$NCAP` candidate target word ids. The source word id and target word id should be consistant with the word ids used by model `$fe_nn_attention`.

# Decoding with Locality Sensitive Hashing

To decode with Winer-Take-All LSH:

```
$EXEC --decode-main-data-files $SRC_TST -L 100  -k 1 $fe_nn $OUTPUT --lsh-type 1 --WTA-K 16 --WTA-units-per-band 3 --WTA-W 500 --WTA-threshold 5 --target-vocab-policy 3 -b 12 > $LOG
```
