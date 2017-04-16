# Zoph\_RNN: A C++/CUDA toolkit for training sequence and sequence-to-sequence models across multiple GPUs

This is [Barret Zoph's](http://barretzoph.github.io/) code for Zoph\_RNN  
Send any questions or comments to barretzoph@gmail.com

This toolkit can successfully replicate the results from the following papers (the multi-gpu parallelism, which is explained in the tutorial, is similar to 6)

1. [Multi-Source Neural Translation](http://www.isi.edu/natural-language/mt/multi-source-neural.pdf)
2. [Simple, Fast Noise Contrastive Estimation for Large RNN Vocabularies](http://www.isi.edu/natural-language/mt/simple-fast-noise.pdf)
3. [Transfer Learning for Low-Resource Neural Machine Translation](http://arxiv.org/pdf/1604.02201v1.pdf)
4. [Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf)
5. [Addressing the Rare Word Problem in Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/acl15_nmt.pdf)
6. [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215.pdf)
7. [Recurrent Neural Network Regularization](http://arxiv.org/pdf/1409.2329.pdf)

# Instructions for compilation/using the code
The code for Zoph\_RNN is provided in the `src/` directory. Additionally, a precompiled binary (named `ZOPH_RNN`) is provided that will work on 64 bit linux for cuda 7.5, so it is not necessary to compile the code. 

If you just want to use the executable, then run the following command `cat executable/ZOPH_RNN_1 executable/ZOPH_RNN_2 executable/ZOPH_RNN_3 executable/ZOPH_RNN_4 > ZOPH_RNN`. Then `ZOPH_RNN` will be the executable that you can use.  To run the executable you need to be sure your path variable includes the location to CUDA. This is a sample command of putting cuda into your PATH variable `export PATH=/usr/cuda/7.5/bin:$PATH`

If you want to compile the Zoph\_RNN code run `bash scripts/compile.sh`, which will compile the code given you set a few environmental variables. The variables that need to be set are below:

1. `PATH_TO_CUDA_INCLUDE` (example value: `/usr/cuda/7.5/include/` ) 
2. `PATH_TO_BOOST_INCLUDE` (example value: `/usr/boost/1.55.0/include/` )
3. `PATH_TO_CUDA_LIB_64` (example value: `/usr/cuda/7.5/lib64/` )
4. `PATH_TO_BOOST_LIB` (example value: `/usr/boost/1.55.0/lib/` )
5. `PATH_TO_CUDNN_V4_64` (example value: `/usr/cudnn_v4/lib64/` )
6. `PATH_TO_EIGEN` (example value: `/usr/eigen/` )
7. `PATH_TO_CUDNN_INCLUDE` (example value: `/usr/cudnn_v4/include/` ) 

### The acceptable versions for the libraries above

Note that cuda version greater than 7.0 is required to run the code, while the rest are required to compile the code

* cuda version greater than 7.0
* gcc version greater than  4.8.1, but not greater than 4.9
* CuDNN version = 4
* Boost version = 1.51.0 or 1.55.0 
* Any version of Eigen


# Tutorial
For this tutorial `ZOPH_RNN` represents the executable to run the code. Also all the scripts in the `scripts` folder require python 3 to run.

This command will bring up the program's help menu showing all the flags that the program can be run with:

```
./ZOPH_RNN -h
```

There are two different kinds of models this code can train

1. Sequence models (Ex: Language Modeling)
2. Sequence-to-Sequence models (Ex: Machine Translation)

The commands for these two different architectures are almost the same, all that needs to change is adding a `-s` flag if you want to use the sequence model. The sequence-to-sequence model is used by default.

In the `sample_data` directory there is sample data provided that shows the proper formatting for files.


### Training a seq-to-seq model:
Lets step through an example that trains a basic sequence-to-sequence model. The following code will train a sequence-to-sequence model with the source training data `/path/to/source_train_data.txt` and the target training data `/path/to/target_train_data.txt`. These are placeholder names that will be replaced with your data files when you are training your own model. The resulting model will be saved to `model.nn`, but this can be named whatever the user wants. Training data always needs to consist of one training example per line, with tokens separated by spaces.

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn
```

By default the source sentences will always be fed in the reversed direction as in [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215.pdf). If you want to feed in the source sentences in the forward direction then simply preprocess your source data, so that it is in the reversed direction.

There are many flags that can be used to train more specific architectures. Lets say we want to train a model with 3 layers (default is 1), 500 hiddenstates (default is 100), and a minibatch of size 64 (default is 8). The following command does this:

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64
```

Lets also make the model have 20,000 source vocabulary and 10,000 target vocabulary (by default the code makes the source vocabulary equal to the number of unique tokens in the source training data, and the target vocab does the same). Also lets apply dropout with a keep probability of 0.8 to the model, where dropout is applied as specified in [Recurrent Neural Network Regularization](http://arxiv.org/pdf/1409.2329.pdf). 

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8
```

Additionally, lets change the learning rate to 0.5 (default is 0.7), add the local-p attention model with feed input as in [Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf).

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true
```

To monitor the training we also want to be able to monitor the performance of the model during training on some held out set of data (developement/validation). Lets do this in the code and also add the option that if perplexity (better is lower) on the held out set of data increased since it was previously checked, then we multiply the current learning rate by 0.5.

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5
```

During training the code needs to produce temporary files. By default these will be put in the directory where the code is launched from, but we can change this to whatever we want. Additionally, we can make all of the output that is typically printed to standard out (the screen) also be printed to a file.

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt
```

Typically during training only one model will be output at the end of training. To make the code output the best model during training according to the perplexity on your heldout data specificed by the `-a` flag we can add the `-B` flag. 

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt -B best.nn
```

Or if you want to save all models every half epoch we can do that with the `--save-all-models` flag.

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt --save-all-models true
```

By default the code will throw away any sentences in training and in the held out data longer than some fixed length which is 100 by default. We can change this to whatever we want, but be careful as it will greatly increase memory usage. Lets change it to 500.

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt --save-all-models true -L 500
```

By default the code uses an MLE objective function, which can be very computationally expensive if the target vocabulary is big. To alleviate this issue we can train with NCE instead of MLE by using the `--NCE` flag. This is the same NCE as in [Simple, Fast Noise Contrastive Estimation for Large RNN Vocabularies](http://www.isi.edu/natural-language/mt/simple-fast-noise.pdf). A good number of noise samples is usually around 100. Note that the `--NCE` flag only has to be specified during training and not during force-decode or decode. 

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt --save-all-models true -L 500 --NCE 100
```

One feature of this code is that is supports model parallelism across multiple gpus. To see the number of available gpu's on your node you can type `nvidia-smi`. The `-M` flag allows our model to put each layer on a gpu of our choosing along with the softmax. -M 0 1 2 3 means put layer 1 on GPU 0, layer 2 on GPU 1, layer 3 on GPU 2 and the softmax on GPU 3. By default the code does -M 0 0 0 0, putting everything on the default GPU 0. We can also change up the specification depending how many gpus we have on the node, so we could do `-M 0 0 1 1` if we only have 2 gpus on our node.

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn -N 3 -H 500 -m 64 --source-vocab-size 20000 --target-vocab-size 10000 -d 0.8 -l 0.5 --attention-model true --feed-input true -a /path/to/source_dev_data.txt /path/to/target_dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt --save-all-models true -L 500 --NCE 100 -M 0 1 2 3
```


### Supplying your own vocabulary mapping file
The `--source-vocab-size N` and the `--target-vocab-size N` flags create a vocabulary mapping file that will replace all words not in the top N most frequent words with <unks>'s. The code will create an integer mapping that is stored in the top of the model file. If you want to supply your own mapping file you can do this using the  `--vocab-mapping-file /path/to/my_mapping.nn`. The `my_mapping.nn` can be a previously trained model, in that case it will use the exact same vocabulary mapping as that model. This is useful because if you want to ensemble models using the `--decode` flag, then the models must have exactly the same target vocabulary mapping file for it to work. In the `scripts/` directory there is a python script called `create_vocab_mapping_file.py`. We can use this to create a mapping file, which then gets fed into the training using the following command:

```
python scripts/create_vocab_mapping_file.py /path/to/source_training_data.txt /path/to/target_training_data.txt 5 my_mapping.nn
``` 

This will create a mapping file named `my_mapping.nn`, which we can then use for training a model using the following command.

```
./ZOPH_RNN -t /path/to/source_training_data.txt /path/to/target_training_data.txt model.nn --vocab-mapping-file my_mapping.nn
```

Instead of using the `create_vocab_mapping_file.py` script, we can also use an existing model as the input for the `--vocab-mapping-file` flag

```
./ZOPH_RNN -t /path/to/source_train_data.txt /path/to/target_train_data.txt model.nn --vocab-mapping-file my_old_model.nn
```

This `--vocab-mapping-file` flag only needs to be specified during training. This can also be used for sequence models in the same way.


### Force-Decoding a seq-to-seq model
Once the model finished training we can use the model file (model.nn, best.nn or any of the models output from `--save-all-best` in the previous training example) for getting the perplexity for a set of source/target pairs or do beam decoding to get the best target outputs given some source sentences. Lets do the former first. We will specify the source and target data we want to get the perplexity for along with the per line log probabilities of each sentece. The output file we specify (`/path/to/output/perp_output.txt`) will contain the per line log probabilities and the total perplexity will be output to standard out. Additionally, we can use the `--logfile` flag as before if we also want standard out to be put to a file too and the `-L` flag to change what the longest sentence the code will accept.

```
./ZOPH_RNN -f /path/to/source_perp_data.txt /path/to/target_perp_data.txt model.nn /path/to/output/perp_output.txt --logfile /path/to/log/logfile.txt -L 500
```

If we trained the model using NCE then we can use the `--NCE-score` flag, which will make the model get the per line log probabilities using an unnormalized softmax. This greatly speeds up force-decode as now a normalization over the softmax does not have to be done, but now it does not represent a distribution that sums to 1. The reason we can do this is because the NCE training objective makes the normalization constant close to 1, so we can get a reasonably good approximation.


### Kbest Decoding for a seq-to-seq model
Lets have the model output the most likely target translation given the source using beam decoding. This can be done with the `--decode` (`-k`) flag. The `model.nn` file will be the trained neural network, `kbest.txt` is where we want the output to be put to and and `source_data.txt` is the file containing the source sentences that we want to be decoded. Once again short sentences are thrown out, so we can change that using the `-L` flag. 

```
./ZOPH_RNN -k 1 model.nn kbest.txt --decode-main-data-files /path/to/source_data.txt -L 500 
```

By default the model uses beam decoding with a beam size of 12. We can change this using the `-b` flag.

```
./ZOPH_RNN -k 1 model.nn kbest.txt --decode-main-data-files /path/to/source_data.txt -L 500 -b 25 
```

We can also output the log probabilities of each sentence being decoded and have it saved in kbest.txt using the `--print-score` flag

```
./ZOPH_RNN -k 1 model.nn kbest.txt --decode-main-data-files /path/to/source_data.txt -L 500 -b 25 --print-score true 
```

Another default during decoding is that the output sentences can only be within the length range [0.5\*(length of source sentence),1.5\*(length of source sentence)]. This can be changed with this --dec-ratio flag.

```
./ZOPH_RNN -k 1 model.nn kbest.txt --decode-main-data-files /path/to/source_data.txt -L 500 -b 25 --print-score true --dec-ratio 0.2 1.8
```


### Ensemble decoding for a seq-to-seq model
In the above example we only decoded a single model. In this code you have the option of ensembling multiple outputs using the `--decode` flag. All of the models you want to ensemble must have the same target vocabulary mappings, so you must use the `--vocab-mapping-file` flag as specified above. We can ensemble together 8 models below, but any number of models can be specified by the user. 

```
./ZOPH_RNN -k 1 model1.nn model2.nn model3.nn model4.nn model5.nn model6.nn model7.nn model8.nn kbest.txt --decode-main-data-files /path/to/source_data1.txt /path/to/source_data2.txt /path/to/source_data3.txt /path/to/source_data4.txt /path/to/source_data5.txt /path/to/source_data6.txt /path/to/source_data7.txt /path/to/source_data8.txt -L 500 -b 25 --print-score true --dec-ratio 0.2 1.8
```

Note that now we pass in 8 different model files and 8 different source data files. The reason for the 8 different source files is that the source vocabularies could be different for all 8 models, so different types of data can be passed in. If you want the same data passed in for all 8 model, then simply copy `/path/to/source_data.txt` 8 times as the input to `--decode-main-data-files`.

### Training a seq model
Training a sequence model is much like training a sequence-to-sequence model. Now we must employ the `-s` flag to denote that we want to train a sequence model. Lets train a model with slighly different parameters from the sequence-to-sequence model above. This model will have a hiddenstate size of 1000, minibatch size of 32, 2 layers, dropout rate of 0.3 and a target vocabulary size of 15K. Also note that now we only need to pass in one data file for training and for dev since it is only a sequence model and not a sequence-to-sequence model.

```
./ZOPH_RNN -s -t /path/to/training_data.txt model.nn -H 1000 -m 32 -l 0.2 -N 2 -M 0 1 2 -d 0.7 --target-vocab-size 15000 -a /path/to/dev_data.txt -A 0.5 --tmp-dir-location /path/to/tmp/ --logfile /path/to/log/logfile.txt --save-all-models true -L 500
```


### Force-Decoding a seq model
To force decode the model it is almost the same as force-decoding a sequence-to-sequence model. In the seq model you can also use the `-m` flag to speedup the batching process, but it will no longer output the per line log probability if `-m` is not set to 1.

```
./ZOPH_RNN -s -f /path/to/dev_data.txt model.nn /path/to/output/perp_output.txt -L 500 --logfile /path/to/log/logfile.txt
```

### Decoding a seq model
This is not a feature in the code.


### Training Multi-Source model
Lets train a multi-source model like in [Multi-Source Neural Translation](http://www.isi.edu/natural-language/mt/multi-source-neural.pdf). In this model we will have two source encoders and one target encoder. This means we need to have 3-way parallel data. The two source training files in this example are: `source1_train_data.txt` and `source2_train_data`.txt. The target training file is: `target_train_data.txt`. All 3 of these files must have the same number of lines. Notice that we must now add the `--multi-source` flag which specifies the second source training file. Additionally, we must specify it a second neural network file name that will be created just like `model.nn`.

```
./ZOPH_RNN -t /path/to/source1_train_data.txt /path/to/target_train_data.txt model.nn --multi-source /path/to/source2_train_data.txt src.nn
```

By default the model combines the two source encoders using the "Basic" method as specified in [Multi-Source Neural Translation](http://www.isi.edu/natural-language/mt/multi-source-neural.pdf). To use the "Child-Sum" method we can add the following flag `--lstm-combine 1`.

```
./ZOPH_RNN -t /path/to/source1_train_data.txt /path/to/target_train_data.txt model.nn --multi-source /path/to/source2_train_data.txt src.nn --lstm-combine 1
```

Additionally, we can use the multi-source attention model from the above paper by adding the three following flags `--attention-model 1 --feed-input 1 --multi-attention 1`. All three flags must be specified to use the multi-source attention model.

```
./ZOPH_RNN -t /path/to/source1_train_data.txt /path/to/target_train_data.txt model.nn --multi-source /path/to/source2_train_data.txt src.nn --lstm-combine 1 --attention-model 1 --feed-input 1 --multi-attention 1 
```

Now lets have the model use a dev set for learning rate monitoring like before. 

```
./ZOPH_RNN -t /path/to/source1_train_data.txt /path/to/target_train_data.txt model.nn --multi-source /path/to/source2_train_data.txt src.nn --lstm-combine 1 --attention-model 1 --feed-input 1 --multi-attention 1 -a /path/to/source1_dev_data.txt /path/to/target_dev_data.txt /path/to/source2_dev_data.txt  -A 0.5 
```
 
### Force-Decoding a Multi-Source model
To force-decode a multi-source model the `--multi-source` flag must be specified when using the -f flag.

```
./ZOPH_RNN -f /path/to/source1_perp_data.txt /path/to/target_perp_data.txt model.nn /path/to/output/perp_output.txt --logfile /path/to/log/logfile.txt -L 500 --multi-source /path/to/source2_perp_data.txt src.nn
```

### Kbest Decoding a Multi-Source model
To decode a multi-source model two additional flags needs to be specified.

```
./ZOPH_RNN -k 1 model.nn kbest.txt --decode-main-data-files /path/to/source1_data.txt --decode-multi-source-data-files /path/to/source2_data.txt --decode-multi-source-vocab-mappings src.nn
```


### Training a Preinit Model
Lets train a model using tranfer learning as specified in [Transfer Learning for Low-Resource Neural Machine Translation](http://arxiv.org/pdf/1604.02201v1.pdf). First we need to have parent data (source and target) and child data (source and target) where the parent and child models must have the same target language. In the paper the shared target language was English. 

Also note that this can only be done with seq-to-seq models and not seq models or multi-source models.

First we must make a mapping file that was shown in the "Supplying your own vocabulary mapping file" section. We can use the script `create_vocab_mapping_file_preinit.py` in the `scripts/` folder. Run the following command to create a vocabulary mapping file where all words that appear less than 5 times will be replaced by <unk> (the 5 can be changed to whatever the user wants):

```
python scripts/create_vocab_mapping_file_preinit.py /path/to/source_child_data.txt /path/to/target_child_data.txt 5 my_mapping.nn /path/to/source_parent_data.txt
```

Now we have created a mapping file `mapping.nn`, which can now be used for training. Now lets train the parent model

```
./ZOPH_RNN -t /path/to/source_parent_data.txt /path/to/target_parent_data.txt parent_model.nn --vocab-mapping-file my_mapping.nn 
```

Once the parent model finished training, we can now train the child model using the following script in the `scripts/` folder.

```
python scripts/pretrain.py --parent parent_model.nn --trainsource /path/to/source_child_data.txt --traintarget /path/to/target_child_data.txt --devsource /path/to/source_child_dev_data.txt --devtarget /path/to/target_child_dev_data.txt --rnnbinary ZOPH_RNN --child child.nn
```

Once the above arguements are supplied other normal parameter flags can be added just like in the `ZOPH_RNN` executable.

```
python scripts/pretrain.py --parent parent_model.nn --trainsource /path/to/source_child_data.txt --traintarget /path/to/target_child_data.txt --devsource /path/to/source_child_dev_data.txt --devtarget /path/to/target_child_dev_data.txt --rnnbinary ./ZOPH_RNN --child child.nn -d 0.8 -l 0.5 -A 0.5 -P 0.01 -w 5 -L 200 -m 32 -n 15 --attention_model True --feed_input True 
```


### Unk Replacement in seq-to-seq model
To do Unk replacement as specified in [Addressing the Rare Word Problem in Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/acl15_nmt.pdf) there is a python script provided in the `scripts/` directory. The following commands need to be run if you want to do unk replacement. This can only be done with attention seq-to-seq models.

1. When decoding (-k or --decode flags) add in the following flag `--UNK-decode /path/to/unks.txt`.

The `unks.txt` file will be generated during decoding, so save it somewhere that it can be accessed later.

2. Run the Berkeley aligner to in order to generate a t-table. The Berkeley aligner is available at: https://code.google.com/archive/p/berkeleyaligner/. A sample parameter file is provided in the scripts/berk_aligner directory. The run_aligner.sh script will run the berkeley aligner, what needs to be changed on the user's end is the unk_replace.conf file. The `execDir` field must be changed to the directory that you want the Berkeley aligner to output all of its files to. The `trainSources` field must give a path to the source training data. The `testSources` field must be changed to the path of the dev data for both source and target. The suffixes must also be changed accordinly depending on what the files were named. 

A sample of what the data/train/ and the data/test/ directories should contain are below (if the foreign and english suffixes are u and e respectively):

`ls data/train`
results in
`train.e  train.u`

`ls data/test`
results in
`test.e test.u`


Once the Berkeley aligner finishes running you need to take the ttable (there are two given by the berkeley aligner, one of P(source | target) and the other P(target | source)) that corresponds to P(target | source). If you make the target language english then the name of the ttable is: `stage2.2.params.txt` else `stage2.1.params.txt`.

Now we can run the following command to decode a seq-to-seq model using unk replacement. `/path/to/unks.txt` is where additional information will be stored during decoding when using unk replacement. 

```
./ZOPH_RNN -k 1 model.nn kbest.txt --decode-main-data-files /path/to/source_data.txt -L 500 -b 25 --print-score true --dec-ration 0.2 1.8 --UNK-decode /path/to/unks.txt
```

Next we will run the `scripts/unk_format.py` script to convert the output of the ZOPH_RNN code into correct format for the `scripts/att_unk_rep.py` script.  

```
python scripts/unk_format.py kbest.txt kbest.txt.formatted
```

Next we will run the final `scripts/att_unk_rep.py` script.

```
python scripts/att_unk_rep.py /path/to/source_data.txt kbest.txt.formatted stage2.2.params.txt kbest.txt.formatted.unkrep
```

Now the `kbest.txt.formatted.unkrep` will contain the decoded sentences with the rare words replaced. The format is 1 output per line.

### Models from papers:
Here are sample commands that can be run to create models in the papers above:

For the paper [Multi-Source Neural Translation](http://www.isi.edu/natural-language/mt/multi-source-neural.pdf). Here is the command to train a multi-source attention model with german and french as the input and english as the output. If you dont want attention remove `--attention-model 1 --feed-input 1 --multi-attention 1` and if you want to use the basic method of combination instead of the child-sum method then change `--lstm-combine 1` to `--lstm-combine 0`.

```
./ZOPH_RNN -t german_train_data.txt train_english_data.txt model.nn -n 15 -B best.nn -m 128 -H 1000 -l 0.7 -w 5 -a german_dev_data.txt english_dev_data.txt french_dev_data.txt -A 1 -v 50000 -V 50000 --clip-cell 50 1000 -N 4 -M 0 1 1 2 3 --multi-source french_train_data.txt src.nn -d 0.8 -L 65 --logfile log.txt --screen-print-rate 15 --fixed-halve-lr-full 11 -P -0.08 0.08 --lstm-combine 1 --attention-model 1 --feed-input 1 --multi-attention 1
``` 

For the paper [Simple, Fast Noise Contrastive Estimation for Large RNN Vocabularies](http://www.isi.edu/natural-language/mt/simple-fast-noise.pdf) the command below will train the billion word language model.

```
./ZOPH_RNN --logfile log.txt -a english_dev_data.txt -s -t english_train_data.txt model.nn -B best.nn --NCE 100 --screen-print-rate 300 -N 4 -M 0 1 2 3 3 -l 0.7 -P -0.08 0.08 -A 0.5 -d 0.8 -n 20  -c 5 -H 2048 --vocab-mapping-file my_mapping.nn -L 205
```

For the paper [Transfer Learning for Low-Resource Neural Machine Translation](http://arxiv.org/pdf/1604.02201v1.pdf) the following command wil train the parent model and the child model (with the child language being Uzbek).


```
python scripts/create_vocab_mapping_file_preinit.py uzbek_train_data.txt english_child_train_data.txt 5 my_mapping.nn french_train_data.txt

./ZOPH_RNN -t french_train_data.txt english_parent_train_data.txt -H 750 -N 2 -d 0.8 -m 128 -l 0.5 -P -0.08 0.08 -w 5 --attention-model 1 --feed-input 1 --screen-print-rate 30 --logfile log.txt -B best.nn -n 10 -L 100 -A 0.5 -a french_dev_data.txt english_parent_dev_data.txt --vocab-mapping-file my_mapping.nn 
```

Once the parent model finishes training then run:

```
python scripts/pretrain.py --parent best.nn --trainsource uzbek_train_data.txt --traintarget english_child_train_data.txt --devsource uzbek_dev_data.txt --devtarget english_child_dev_data.txt --rnnbinary ZOPH_RNN --child child.nn -d 0.5 -l 0.5 -A 0.9 -P -0.05 0.05 -w 5 -L 100 -m 128 -n 100 --attention_model True --feed_input True 
```



For the paper [Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf) the command below will train the "Base + reverse + dropout + local-p attention (general) + feed input" from Table 1.

```
./ZOPH_RNN --logfile log.txt -a english_dev_data.txt german_dev_data.txt -t english_train_data.txt german_train_data.txt model.nn -B best.nn --screen-print-rate 300 -N 4 -M 0 1 2 2 3 -L 50 -l 1 -P -0.1 0.1 --fixed-halve-lr-full 9 -A 1 -d 0.8 -n 12 -w 5 --attention-model 1 --feed-input 1 --attention-width 10 -v 50000 -V 50000
```

For the paper [Sequence to Sequence Learning with Neural Networks](http://arxiv.org/pdf/1409.3215.pdf) the following command will train the "Single reversed LSTM"

```
./ZOPH_RNN -t source_train_data.txt target_train_data.txt model.nn  -H 1000 -N 4 -v 160000 -V 80000 -P -0.08 0.08 -l 0.7 -n 8 --fixed-halve-lr 6 -m 128 -w 5 -L 100 
```

# Changes from previous version
- The flag (`--HPC-output`) has been renamed to (`--logfile`)
- The flag (`--source-vocab`) has been renamed to (`--source-vocab-size`)
- The flag (`--target-vocab`) has been renamed to (`--target-vocab-size`)
- The flag (`--random-seed`) now takes in an integer to use as the fixed random seed, or by default now seeds with the time
- The flag (`--save-all-best`) has been renamed to (`--save-all-models`)
- The flag (`--feed_input`) has been renamed to (`--feed-input`)
- The default minibatch size was changed from 128 to 8
- The default hiddenstate size was changed from 1000 to 100
- Added attention, multi-source, NCE, unk-replacement, transfer learning


# License
MIT
