
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cudnn.h>

#include "math_constants.h"

//Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

//Boost
#include "boost/program_options.hpp" 
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

//My own includes
#include "add_model_info.h"
#include "logger.h"
#include "global_params.h"
#include "prev_states.h"
#include "input_file_prep.h"
#include "BZ_CUDA_UTIL.h"
#include "conv_char.h"
#include "encoder_multi_source.h"
#include "bi_encoder.h"
#include "attention_layer.h"
#include "attention_node.h"
#include "attention_combiner.h"
#include "decoder_model_wrapper.h"
#include "ensemble_factory.h"
#include "base_layer.h"
#include "NCE.h"
#include "gpu_info_struct.h"
#include "custom_kernels.h"
#include "Hidden_To_Hidden_Layer.h"
#include "LSTM_HH.h"
#include "model.h"
#include "fileHelper.h"
#include "fileHelper_source.h"
#include "Eigen_Util.h"
#include "model.hpp"
//#include "base_layer.hpp"
#include "LSTM.hpp"
#include "softmax.hpp"
#include "Input_To_Hidden_Layer.hpp"
#include "Hidden_To_Hidden_Layer.hpp"
#include "LSTM_HH.hpp"
#include "decoder_model_wrapper.hpp"
#include "ensemble_factory.hpp"
#include "attention_layer.hpp"
#include "attention_node.hpp"
#include "NCE.hpp"
#include "bi_encoder.hpp"
#include "encoder_multi_source.hpp"
#include "tree_LSTM.hpp"
#include "input_file_prep.hpp"
#include "attention_combiner.hpp"
#include "attention_combiner_node.hpp"
#include "conv_char.hpp"
#include "highway_network.hpp"
#include "memory_util.h"


//parse the command line from the user
void command_line_parse(global_params &params,int argc, char **argv) {
		
	//files for keeping the user input
	//if not s, 1st source, 2nd target, 3rd output weights name
	//if s, 1st target, 2nd output weights name
	std::vector<std::string> train_files;

	//files for force decoding
	//if not s, 1. source input file 2. target input file  3. neural network file name 4. output file name
	//if s, 1. target input file  2. neural network file name 3. output file name
	std::vector<std::string> test_files;

	//stuff for adaptive learning rate schedule
	//if not seq , 1st is source dev, 2nd is target dev
	//if seq 1st is target dev
	std::vector<std::string> adaptive_learning_rate;

	//lower and upper range for parameter initialization
	std::vector<precision> lower_upper_range;

	//for the kbest flag, 4 arguements must be entered for kbest, 1. number of best paths 2 input file name 
	//3. neural network file name (this is the output file you get after training the neural network)4. output file name
	std::vector<std::string> kbest_files;

	//for stoic gen, 1st neural network file, 2nd is output file name
	std::vector<std::string> stoicgen_files;

	//truncated softmax
	std::vector<std::string> trunc_info;

	//for decoding ratios 
	std::vector<precision> decoding_ratio;

	//for continuing to train
	std::vector<std::string> cont_train;

	//for multi gpu training
	std::vector<int> gpu_indicies;

	std::vector<precision> clip_cell_vals;

	std::vector<double> NCE_vals;

	//for multisource
	std::vector<std::string> multi_source;

  //for char-mt
  std::vector<int> char_mt_vec;

	//basic format setup
	namespace po = boost::program_options; 
	po::options_description desc("Options");
	desc.add_options() 
  		("help,h", "Run to get help on how to use the program. This is version 1.0")
  		("train,t",po::value<std::vector<std::string> > (&train_files)->multitoken(),"Train a model with input data file(s) and a name for the neural network output file"\
  			". \nFORMAT (if sequence to sequence): <source file name> <target file name> <neural network output name> "\
  			" \nFORMAT (if sequence): <target file name> <neural network output name>")
  		("cont-train,C",po::value<std::vector<std::string>> (&cont_train)->multitoken(),"Resume training of a model (THIS WILL OVERWRITE THE MODEL FILE PASSED IN)\n"\
  			"FORMAT: (if sequence to sequence): <source file name> <target file name> <neural network file name>\n"\
  			"FORMAT: (if seq): <target file name> <neural network file name>")
  		("vocab-mapping-file",po::value<std::string> (&params.ensemble_train_file_name),"Train a model with the same integerization mappings as another model. This is needed to do ensemble decoding\n"\
  			"FORMAT: <neural network file name>")
      ("train-source-RNN",po::value<bool>(&deniz::train_source_RNN),"train source RNN. DEFAULT: True")
      ("train-target-RNN",po::value<bool>(&deniz::train_target_RNN),"train target RNN. DEFAULT: True")
      ("train-source-input-embedding",po::value<bool>(&deniz::train_source_input_embedding),"train source input embeddings. DEFAULT: True")
      ("train-target-input-embedding",po::value<bool>(&deniz::train_target_input_embedding),"train target input embeddings. DEFAULT: True")
      ("train-target-output-embedding",po::value<bool>(&deniz::train_target_output_embedding),"train target output embeddings. DEFAULT: True")
      ("train-attention-target-RNN",po::value<bool>(&deniz::train_attention_target_RNN),"train target attention. DEFAULT: True")
  		("vocab-mapping-file-multi-source",po::value<std::string> (&params.multi_src_params.ensemble_train_file_name),"specify multi-source mapping for vocab")
  		("multi-source",po::value<std::vector<std::string>> (&multi_source)->multitoken(),"Specify the second source training file and mapping file for the multi-source model")
  		//("multi-attention",po::value<bool>(&params.multi_src_params.multi_attention),"for attention model with multiple sources\n")
  		("multi-attention",po::value<bool>(&params.multi_src_params.multi_attention_v2),"Make the multi-source seq-to-seq model use attention\n")
  		//("char-mt",po::value<std::vector<int>> (&char_mt_vec)->multitoken(),"<filter_size> <char_emb_size> <num highway layers> \n")
      //("add-ht",po::value<bool>(&params.multi_src_params.add_ht),"add hiddenstates for both attention models instead of sending through neural network\n")
  		//("print-norms",po::value<bool>(&BZ_CUDA::print_norms),"Print out norms of all matrices\n")
  		("lstm-combine",po::value<bool>(&params.multi_src_params.lstm_combine),"For multi source seq-to-seq model, use the child-sum combination method if set to true, else use the basic method. DEFAULT: false\n")
  		("num-layers,N",po::value<int>(&params.num_layers),"Set the number of LSTM layers you want for your model\n DEFAULT: 1")
  		("multi-gpu,M",po::value<std::vector<int>> (&gpu_indicies)->multitoken(), "Train the model on multiple gpus.\nFORMAT: <gpu for layer 1> <gpu for layer 2> ... <gpu for softmax>\n"\
  			"DEFAULT: all layers and softmax lie on gpu 0")
  		("force-decode,f",po::value<std::vector<std::string> > (&test_files)->multitoken(), "Get per line probability of dataset plus the perplexity\n"\
  			"FORMAT: (if sequence to sequence): <source file name> <target file name> <trained neural network file name> <output file name>\n"\
  			"FORMAT: (if sequence): <target file name> <trained neural network file name> <output file name>")
  		// ("stoch-gen,g", po::value<std::vector<std::string> > (&stoicgen_files)->multitoken(),"Do random generation for a sequence model, such as a language model\n"\
  		// 	"FORMAT: <neural network file name> <output file name>")
  		// ("stoch-gen-len",po::value<int>(&params.sg_length) ,"How many sentences to let stoch-gen run for\n"\
  		// 	"FORMAT: <num sentences>\n"
  		// 	"DEFAULT: 100")
  		//("dump-alignments",po::value<bool>(&params.attent_params.dump_alignments),"Dump the alignments to a file")
  		// ("temperature",po::value<double>(&params.temperature) ,"What should the temperature be for the stoch generation"\
  		// 	"FORMAT: <temperature>  where temperature is typically between [0,1]. A lower temperature makes the model output less and less from what it memorized from training\n"\
  		// 	"DEFAULT: 1")
  		("sequence,s", "Train model that learns a sequence,such as language modeling. Default model is sequence to sequence model")
  		("tmp-dir-location",po::value<std::string>(&params.tmp_location),"For all modes in the code, a tmp directiory must be created for data preparation. Specify the location of where you want this to be created. DEFAULT: Current directory")
      //("bi-directional",po::value<bool>(&params.bi_dir_params.bi_dir),"Have the source sequence be encoded bi-diretionally\n")
  		//("combine-bi-directional",po::value<bool>(&params.bi_dir_params.bi_dir_comb),"send a nonlinear tranformation of the rev and nonrev hidden states from the source encoders to the decoder\n")
  		//("share-embedding",po::value<bool>(&params.bi_dir_params.share_embeddings),"For the bidirectional encoder, share the embeddings")
  		("dropout,d",po::value<precision>(&params.dropout_rate),"Use dropout and set the dropout rate. This value is the probability of keeping a node. FORMAT: <dropout rate>. DEFAULT: 1.0")
      ("learning-rate,l",po::value<precision>(&params.learning_rate),"Set the learning rate. DEFAULT: 0.5")
  		("random-seed",po::value<int>(&params.random_seed_int),"Specify a random seed, instead of the model being seeded with the current time\n")
  		("longest-sent,L",po::value<int>(&params.longest_sent),"Set the maximum sentence length for training/force-decode/decode. DEFAULT: 100")
  		("hiddenstate-size,H",po::value<int>(&params.LSTM_size),"Set hiddenstate size. DEFAULT: 100")
  		//("UNK-replacement",po::value<int>(&params.unk_aligned_width),"Set unk replacement to be true and set the wideth\n FORMAT: <alignment width>")
  		// ("truncated-softmax,T",po::value<std::vector<std::string>> (&trunc_info)->multitoken(),"Use truncated softmax\n DEFAULT: not being used\n"\
  		// 	"FORMAT: <shortlist size> <sampled size>")
      ("UNK-decode",po::value<std::string>(&BZ_CUDA::unk_rep_file_name),"Use unk replacement at decoding time if you have an attention model. Specify a file that the system will output information to. \
        This file will then need to be passed to the python script")
  		("NCE",po::value<int>(&params.num_negative_samples),"Use an NCE loss function, specify the number of noise samples you want (these are shared across the minibatch for speed). DEFAULT: uses MLE not NCE")
  		("NCE-share-samples",po::value<bool>(&params.share_samples),"Share the noise samples across the minibatch when using NCE for a speed increase. DEFAULT: True ")
      //("NCE-leg-dump",po::value<bool>(&BZ_CUDA::nce_legacy_dump),"Dont use this option")
  		("NCE-score",po::value<bool>(&BZ_CUDA::nce_score),"Bool for using unnormalized softmax outputs for force decoding. This will make the probabilities not sum to 1, but makes decoding significanly faster. You must have trained the model with NCE for this to work. DEFAULT: false")
      //("ASHISH-NCE-STATS",po::value<bool>(&BZ_CUDA::dump_NCE_stats),"for ashish")
      ("attention-model",po::value<bool>(&params.attent_params.attention_model),"Bool for whether you want to train with the attention mode. DEFAULT: False\n")
  		("attention-width",po::value<int>(&params.attent_params.D),"How many words do you want to look at around the alignment position on one half. DEFAULT: 10\n")
  		("feed-input",po::value<bool>(&params.attent_params.feed_input),"Bool for wether you want feed input for the attention model. DEFAULT: False\n")
  		("source-vocab-size,v",po::value<int>(&params.source_vocab_size),"Set source vocab size\n DEFAULT: number of unique words in source training corpus")
  		("target-vocab-size,V",po::value<int>(&params.target_vocab_size),"Set target vocab size\n DEFAULT: number of unique words in target training corpus")
  		("shuffle",po::value<bool>(&params.shuffle),"true if you want to shuffle the train data. DEFAULT: True")
  		("parameter-range,P",po::value<std::vector<precision> > (&lower_upper_range)->multitoken(),"parameter initialization range\n"\
  			"FORMAT: <Lower range value> <Upper range value>\n DEFAULT: -0.08 0.08")
  		("number-epochs,n",po::value<int>(&params.num_epochs),"Set number of epochs. DEFAULT: 10")
  		("matrix-clip-gradients,c",po::value<precision>(&params.norm_clip),"Set gradient clipping threshold\n DEFAULT: 5")
  		//("ind-clip-gradients,i",po::value<precision>(&BZ_CUDA::ind_norm_clip_thres),"CURRENT THIS DOES NOT WORK!!!!!!!!!!!!!!!!!!! \nSet gradient clipping threshold for individual elements\n DEFAULT: 0.1")
  		("whole-clip-gradients,w",po::value<precision>(&params.norm_clip),"Set gradient clipping threshold for all gradients\n DEFAULT: 5")
  		("adaptive-halve-lr,a",po::value<std::vector<std::string>> (&adaptive_learning_rate)->multitoken(),"change the learning rate"\
  			" when the perplexity on your specified dev set increases from the previous half epoch by some constant, so "\
  			" new_learning_rate = constant*old_learning rate, by default the constant is 0.5, but can be set using adaptive-decrease-factor\n"
  			"FORMAT: (if sequence to sequence): <source dev file name> <target dev file name>\n"\
  			"FORMAT: (if sequence): <target dev file name>")
  		("clip-cell",po::value<std::vector<precision>>(&clip_cell_vals)->multitoken(),"Specify the cell clip threshold and the error threshold in backprop.\n FORMAT: <Cell clip threshold> <Error clip Threshold> . Recommended values: <50> <1000>. DEFAULT: not used\n")
  		("adaptive-decrease-factor,A",po::value<precision>(&params.decrease_factor),"To be used with adaptive-halve-lr"\
  			" it\n DEFAULT: 0.5")
  		("fixed-halve-lr",po::value<int> (&params.epoch_to_start_halving),"Halve the learning rate"\
  			" after a certain epoch, every half epoch afterwards by a specific amount. FORMAT: <epoch number>")
  		("fixed-halve-lr-full",po::value<int> (&params.epoch_to_start_halving_full),"Halve the learning rate"\
  			" after a certain epoch, every epoch afterwards by a specific amount. FORMAT: <epoch number>")
  		("minibatch-size,m",po::value<int>(&params.minibatch_size),"Set minibatch size. DEFAULT: 8.")
  		("screen-print-rate",po::value<int>(&params.screen_print_rate),"Set after how many minibatches you want to print info to the stdout and/or the logfile\n DEFAULT: 5")
  		("logfile",po::value<std::string>(&params.HPC_output_file_name),"Dump the terminal output to a" \
  			"file \n FORMAT: <file name>")
  		("best-model,B",po::value<std::string>(&params.best_model_file_name),"During train have the best model (determined by validation perplexity) be written to a file\nFORMAT: <output file name>")
  		("save-all-models",po::value<bool>(&BZ_CUDA::dump_every_best),"Save the every model every half epoch")
      ("decode,k",po::value<std::vector<std::string> > (&kbest_files)->multitoken(),"Get top decoding outputs using beam search in sequence to sequence model. You can specify more than one model for ensemble decoding\n"\
  			"FORMAT: <how many outputs> <neural network file 1> <neural network file 2> ... <output file>")
  		("decode-main-data-files",po::value<std::vector<std::string> > (&params.decode_user_files)->multitoken(),"FORMAT: <data file 1> <data file 2> ... ")
  		("decode-multi-source-data-files",po::value<std::vector<std::string> > (&params.decode_user_files_additional)->multitoken(),"FORMAT: <multi-source data file 1> <multi-source data file 2> ... ")
  		("decode-multi-source-vocab-mappings",po::value<std::vector<std::string> > (&params.model_names_multi_src)->multitoken(),"FORMAT: <multi-source vocab mapping 1> <multi-source vocab mapping 2> ... ")
  		("pre-norm-ensemble",po::value<bool>(&BZ_CUDA::pre_norm),"For --decode, ensemble the models before they are normalized to probabilities")
        ("beam-size,b",po::value<int>(&params.beam_size),"Set beam size for --decode paths\n DEFAULT: 12")
  		("penalty,p",po::value<precision>(&params.penalty),"Set penalty for --decode decoding. The value entered"\
  			" will be added to the log probability score per target word decoded. This can make the model favor longer sentences for decoding\n DEFAULT: 0")
  		("print-score",po::value<bool>(&params.print_score),"Set if you want to print out the unnormalized log prob for each path when using --decode"\
  			"FORMAT: <bool> \nthe bool is 1 if you want to print the score or 0 otherwise.\n DEFAULT: false")
  		("dec-ratio",po::value<std::vector<precision>>(&decoding_ratio)->multitoken(),"Set the min and max decoding length rations when using --decode\n"\
  			"This means that a target decoded sentence must be at least min_dec_ratio*len(source sentence)"\
  			" and not longer than max_dec_ratio*len(source sentence)\nFORMAT: <min ration> <max ratio>\n"\
  			"DEFAULT: 0.5, 1.5")
        // for fsa
        ("interactive",po::value<bool>(&params.interactive),"Interactive Mode. FORMAT: <bool> \n DEFAULT: false")
        ("interactive-line",po::value<bool>(&params.interactive_line),"Interactive line by line Mode. FORMAT: <bool> \n DEFAULT: false")
        ("print-beam",po::value<bool>(&params.print_beam),"Set if you want to print out the beam cells, mainly used for debug. FORMAT: <bool> \n DEFAULT: false")
        ("repeat-penalty",po::value<precision>(&params.repeat_penalty),"Set penalty for kbest decoding. The value entered will be added to the log probability score per target word decoded. This can make the model favor sentences for less repeating words\n DEFAULT: 0")
        ("adjacent-repeat-penalty",po::value<precision>(&params.adjacent_repeat_penalty),"Set penalty for kbest decoding. The value entered will be added to the log probability score per target word decoded. This will disencourage adjacent word copying.\n DEFAULT: 0")
  		("fsa",po::value<std::string>(&params.fsa_file),"the fsa file for the decoder, should be in carmel format\nFORMAT: <fsa file name>")
        ("fsa-weight",po::value<float>(&params.fsa_weight),"the fsa weight for the decoder, \nDEFAULT: 0.0")
        ("fsa-log",po::value<bool>(&params.fsa_log),"Whether the probability in fsa file is in log space, DEFAULT: false\n")
        ("encourage-list",po::value<std::vector<std::string>>(&params.encourage_list)->multitoken(),"provide encourage word list files for the decoding, each line should contain a encourage word \nFORMAT: <file1> <file2>")
("encourage-weight",po::value<std::string>(&params.encourage_weight_str)->multitoken(),"The encourage weights. The weight is in log(e) space, and will be added to the corresponding word probability during decoding\nFORMAT: <weight1>,<weight2> e.g. 1.0,-0.5\n DEFAULT: ")
("wordlen-weight",po::value<precision>(&params.wordlen_weight),"wordlen weight\n DEFAULT: 0")
("alliteration-weight",po::value<precision>(&params.alliteration_weight),"alliteration weight\n DEFAULT: 0")
    
    
    
    
        ;
    //   ("tsne-dump",po::value<bool>(&BZ_STATS::tsne_dump),"for dumping multi-source hiddenstates during decoding")
  		// ("Dump-LSTM",po::value<std::string>(&params.LSTM_dump_file),"Print the output at each timestep from the LSTM\nFORMAT: <output file name>\n"\
  		// 	"The file lines that are output are the following: 1.input word, embedding   2.Forget gate   3.input gate"\
  		// 	"   4.c_t   5.output gate    6.h_t     7.probabilities");

    po::variables_map vm; 
//kbest should be changed to decode. train-emsemble should be changed to vocab-mapping-file. screen-print-rate should be changed 
//Declare license for the code. LGPL license or MIT license?. 

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

        std::cout << "------------- Printing options that have currently being set by the user -------------\n";
    //now try to loop over all the boost program options
    for (auto it=vm.begin(); it != vm.end(); it++) {
      std::cout << "Variable: " << it->first << "   Value: ";
      auto& value = it->second.value();
      if (auto v = boost::any_cast<int>(&value)) {
        std::cout << *v << "\n";
      }
      else if (auto v = boost::any_cast<bool>(&value)) {
        std::cout << *v << "\n";
      }
      else if (auto v = boost::any_cast<float>(&value)) {
        std::cout << *v << "\n";
      }
      else if(auto v = boost::any_cast<double>(&value)) {
        std::cout << *v << "\n";
      }
      else if(auto v = boost::any_cast<std::string>(&value)) {
        std::cout << *v << "\n";
      }
      else if(std::vector<std::string> *v = boost::any_cast<std::vector<std::string>>(&value)) {
        std::vector<std::string> vv = *v;
        for(int i=0; i<vv.size(); i++) {
          std::cout << " " << vv[i] << " ";
        }
        std::cout << "\n";
      }
      else {
        std::cout << "Not Printable\n";
      }
    }
    std::cout << "--------------------------------------------------------------------------------------\n\n";

		//see if the user specified the help flag
		if ( vm.count("help") ) {

			std::cout << "\n------------------------------\n";
			std::cout << "This is Barret Zoph's GPU RNN library\n"
            << "The flags for the command line interface are below\n" 
            << "Look at the README for an indepth tutorial and example commands\n"  
            << "" << "\n";

			std::cout << desc << "\n";
			exit (EXIT_FAILURE);
		}

    if (vm.count("random-seed") ) {
      params.random_seed = true;
    }

    if (vm.count("tmp-dir-location")) {
      if (params.tmp_location != "") {
        if (params.tmp_location[params.tmp_location.size()-1]!='/') {
          params.tmp_location+="/";
        }
      }
    }

    if(vm.count("shuffle")) {
        BZ_CUDA::shuffle_data = params.shuffle;
    }
    
    if(vm.count("logfile")) {
      params.HPC_output = true;
      //BZ_CUDA::HPC_output = true;
    }

    BZ_CUDA::logger.SetOutputLogger(params.HPC_output_file_name,params.HPC_output);

//error checks to be sure only once of these options is set
	if (vm.count("train") && vm.count("decode")) {
		BZ_CUDA::logger << "ERROR: you cannot train and get decode at the same time\n";
		exit (EXIT_FAILURE);
	}
	if (vm.count("train") && vm.count("force-decode")) {
		BZ_CUDA::logger << "ERROR: you cannot train and force-decode at the same time\n";
		exit (EXIT_FAILURE);
	}
	if (vm.count("force-decode") && vm.count("decode")) {
		BZ_CUDA::logger << "ERROR: you cannot force-decode and get decode at the same time\n";
		exit (EXIT_FAILURE);
	}
	if (!(vm.count("train") || vm.count("force-decode") || vm.count("decode")||vm.count("stoch-gen") || vm.count("cont-train") )) {
		BZ_CUDA::logger << "ERROR: you must either train,continue training,get decode,stoch generate data or force-decode\n";
		exit (EXIT_FAILURE);
	}

    if(vm.count("parameter-range")) {
      BZ_CUDA::lower = lower_upper_range[0];
      BZ_CUDA::upper = lower_upper_range[1];
    }

    if(vm.count("cont-train")) {
        BZ_CUDA::cont_train = true;
    }
    else {
        BZ_CUDA::cont_train = false;
    }

    //this is for making sure dev_synch_all only loops over current GPU's specified
//    if(vm.count("multi-gpu")) {
//      if(gpu_indicies.size()==0) {
//        gpu_info::device_numbers.push_back(0);
//      }
//      else {
//        gpu_info::device_numbers = gpu_indicies;
//      }
//    }



		if(vm.count("clip-cell")) {
			if(clip_cell_vals.size()!=2) {
				BZ_CUDA::logger << "ERROR: clip-cell must have exactly two arguement\n";
				exit (EXIT_FAILURE);
			}
			BZ_CUDA::clip_cell = true;
			BZ_CUDA::cell_clip_threshold = clip_cell_vals[0];
			BZ_CUDA::error_clip_threshold = clip_cell_vals[1];
		}

		params.longest_sent+=4; //because it is really 4 less

    if(vm.count("UNK-decode")) {
      BZ_CUDA::unk_replacement = true;
      BZ_CUDA::unk_rep_file_stream.open(BZ_CUDA::unk_rep_file_name.c_str());
      for(int i=0; i<params.beam_size; i++) {
        BZ_CUDA::viterbi_alignments.push_back(-1);
      }
      for(int i=0; i<params.beam_size * params.longest_sent; i++) {
        BZ_CUDA::alignment_scores.push_back(0);
      }

      BZ_CUDA::h_align_indicies = (int*)malloc((2*params.attent_params.D+1)*params.beam_size*sizeof(int));
      BZ_CUDA::h_alignment_values = (precision*)malloc((2*params.attent_params.D+1)*params.beam_size*sizeof(precision));
    }

    if(vm.count("char-mt")) {
      params.char_params.char_cnn = true;
      params.char_params.filter_size = char_mt_vec[0];
      params.char_params.char_emb_size = char_mt_vec[1];
      params.char_params.num_highway_layers = char_mt_vec[2];
      extract_char_info(params.char_params.longest_word,params.char_params.num_unique_chars_source,
        params.char_params.num_unique_chars_target,params.source_vocab_size,params.target_vocab_size,
        params.char_params.char_mapping_file,params.char_params.word_mapping_file);
    }

		if(vm.count("train") || vm.count("cont-train")) {

			if(vm.count("multi-source")) {
				if(multi_source.size()!=2) {
					BZ_CUDA::logger << "ERROR only two arguements for the multi-source flag\n";
					exit (EXIT_FAILURE);
				}
				params.multi_src_params.multi_source = true;
				params.multi_src_params.file_name = multi_source[0];
				params.multi_src_params.source_model_name = multi_source[1];
			}


			//some basic error checks to parameters
			if(params.learning_rate<=0) {
				BZ_CUDA::logger << "ERROR: you cannot have a learning rate <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.minibatch_size<=0) {
				BZ_CUDA::logger << "ERROR: you cannot have a minibatch of size <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.LSTM_size<=0) {
				BZ_CUDA::logger << "ERROR: you cannot have a hiddenstate of size <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.source_vocab_size<=0) {
				if(params.source_vocab_size!=-1) {
					BZ_CUDA::logger << "ERROR: you cannot have a source_vocab_size <=0\n";
					exit (EXIT_FAILURE);
				}
			}
			if(params.target_vocab_size<=0) {
				if(params.target_vocab_size!=-1) {
					BZ_CUDA::logger << "ERROR: you cannot have a target_vocab_size <=0\n";
					exit (EXIT_FAILURE);
				}
			}
			if(params.norm_clip<=0) {
				BZ_CUDA::logger << "ERROR: you cannot have your norm clip <=0\n";
				exit (EXIT_FAILURE);
			}

			if(params.num_epochs<=0) {
				BZ_CUDA::logger << "ERROR: you cannot have num_epochs <=0\n";
				exit (EXIT_FAILURE);
			}

			// if(vm.count("logfile")) {
			// 	params.HPC_output = true;
   //      BZ_CUDA::HPC_output = true;
			// }

			if(vm.count("dropout")) {
				params.dropout = true;
				if(params.dropout_rate < 0 || params.dropout_rate > 1) {
					BZ_CUDA::logger << "ERROR: dropout rate must be between 0 and 1\n";
					exit (EXIT_FAILURE);
				}
			}

			if(vm.count("matrix-clip-gradients")) {
				BZ_CUDA::global_clip_flag = false;
				params.clip_gradient = true;
				BZ_CUDA::individual_grad_clip = false;
			}

			if(vm.count("whole-clip-gradients")) {
				BZ_CUDA::global_clip_flag = true;
				params.clip_gradient = false;
				BZ_CUDA::individual_grad_clip = false;
			}

			if(vm.count("ind-clip-gradients")) {
				BZ_CUDA::global_clip_flag = false;
				params.clip_gradient = false;
				BZ_CUDA::individual_grad_clip = true;
			}

			if(vm.count("NCE")) {
				params.NCE = true;
				params.softmax = false;
				//BZ_CUDA::print_partition_function = true;
			}

			if(vm.count("UNK-replacement")) {
				params.unk_replace = true;
			}

			boost::filesystem::path unique_path = boost::filesystem::unique_path();
      if(vm.count("tmp-dir-location")) {
        unique_path = boost::filesystem::path(params.tmp_location + unique_path.string());
      }
			BZ_CUDA::logger << "Temp directory being created named: " << unique_path.string() << "\n\n";
			boost::filesystem::create_directories(unique_path);
			params.unique_dir = unique_path.string();
      //BZ_CUDA::logger << "Unique_dir: " << params.unique_dir << "\n";
			params.train_file_name = params.unique_dir+"/train.txt";

			//number of layers
			//error checking is done when initializing model
			if(vm.count("multi-gpu")) {
				params.gpu_indicies = gpu_indicies;
			}



			if(vm.count("cont-train")) {

				//sequence model
				if(vm.count("sequence")) {
					if(cont_train.size()!=2) {
						BZ_CUDA::logger << (int)cont_train.size() << "\n";
						BZ_CUDA::logger << "ERROR: two arguements to be supplied to the continue train flag\n"\
						" 1. train data file name, 2. neural network file name\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					params.attent_params.attention_model = false;
					params.target_file_name = cont_train[0];
					params.input_weight_file = cont_train[1];
					params.output_weight_file = cont_train[1];
					params.LM = true;
					params.load_model_train = true;
					params.load_model_name = params.input_weight_file;

					input_file_prep input_helper;

					input_helper.integerize_file_LM(params.input_weight_file,params.target_file_name,params.train_file_name,
						params.longest_sent,params.minibatch_size,true,params.LSTM_size,params.target_vocab_size,params.num_layers);

				}
				else {
					if(cont_train.size()!=3) {
						BZ_CUDA::logger << "ERROR: three arguements to be supplied to the continue train flag\n"\
						" 1. source train data file name  2. target train data file name  3. neural network file name  \n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					params.LM = false;
					params.source_file_name = cont_train[0];
					params.target_file_name = cont_train[1];
					params.input_weight_file = cont_train[2];
					params.output_weight_file = cont_train[2];
					params.load_model_train = true;
					params.load_model_name = params.input_weight_file;
          BZ_CUDA::logger << "Load model name: " << params.load_model_name << "\n";

					if(params.source_file_name == params.target_file_name) {
						BZ_CUDA::logger << "ERROR: do not use the same file for source and target data\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					input_file_prep input_helper;

					if(vm.count("multi-source")) {
						params.multi_src_params.int_file_name = params.unique_dir + params.multi_src_params.int_file_name;
					}
					
          if(params.char_params.char_cnn) {
            params.train_file_name = params.char_params.word_train_file;
            params.test_file_name = params.char_params.word_dev_file;
            params.output_weight_file = params.char_params.word_mapping_file;
          }
          else {
  					input_helper.integerize_file_nonLM(params.input_weight_file,params.source_file_name,
  						params.target_file_name,params.train_file_name,params.longest_sent,params.minibatch_size,params.LSTM_size,
  						params.source_vocab_size,params.target_vocab_size,params.num_layers,params.attent_params.attention_model,
  						params.multi_src_params.multi_source,params.multi_src_params.file_name,params.multi_src_params.int_file_name,
  						params.multi_src_params.source_model_name);
          }
				}
			}
			else {

				if(vm.count("num-layers")) {
					if(params.num_layers <=0) {
						BZ_CUDA::logger << "ERROR: you must have >= 1 layer for your model\n";
						exit (EXIT_FAILURE);
					}
				}

				//now create the necessary files
				if(vm.count("sequence")) {
					
					if(train_files.size()!=2) {
						BZ_CUDA::logger << "ERROR: two arguements to be supplied to the train flag"\
						" 1. train data file name, 2. neural network output name\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					params.attent_params.attention_model = false;
					params.LM = true;
					params.target_file_name = train_files[0];
					params.output_weight_file = train_files[1];

					input_file_prep input_helper;

					if(vm.count("vocab-mapping-file")) {
						params.ensemble_train = true;
					}


					//this outputs the train.txt file along with the mappings and first line
					bool success=true;
					if(!params.ensemble_train) {

						success = input_helper.prep_files_train_LM(params.minibatch_size,params.longest_sent,
							params.target_file_name,
							params.train_file_name,params.target_vocab_size,
							params.shuffle,params.output_weight_file,params.LSTM_size,params.num_layers);
					}
					else {
						success = input_helper.prep_files_train_LM_ensemble(params.minibatch_size,params.longest_sent,
							params.target_file_name,
							params.train_file_name,params.target_vocab_size,
							params.shuffle,params.output_weight_file,params.LSTM_size,params.num_layers,params.ensemble_train_file_name);
					}



					//clean up if error
					if(!success) {
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}
				}
				else {
					//then sequence to sequence model
					if(train_files.size()!=3) {
						BZ_CUDA::logger << (int)train_files.size() <<"\n";
						BZ_CUDA::logger << "ERROR: three arguements to be supplied to the train flag for the sequence to sequence model\n"\
						" 1. source train data file name\n 2. target train data file name \n3. neural network output name\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					params.LM = false;
					params.source_file_name = train_files[0];
					params.target_file_name = train_files[1];
					params.output_weight_file = train_files[2];

					if(params.source_file_name == params.target_file_name) {
						BZ_CUDA::logger << "ERROR: do not use the same file for source and target data\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					//see if ensemble training
					if(vm.count("vocab-mapping-file")) {
						params.ensemble_train = true;
					}

					input_file_prep input_helper;

					bool success=true;

          //check if char
          if(params.char_params.char_cnn) {
            params.train_file_name = params.char_params.word_train_file;
            params.test_file_name = params.char_params.word_dev_file;
            params.output_weight_file = params.char_params.word_mapping_file;
          }
          else {
  					if(params.multi_src_params.multi_source) {
  						params.multi_src_params.int_file_name = params.unique_dir + params.multi_src_params.int_file_name;
  						if(params.ensemble_train) {
  							input_helper.prep_files_train_nonLM_multi_source_ensemble(params.minibatch_size,params.longest_sent,
  								params.source_file_name,params.target_file_name,
  								params.train_file_name,params.source_vocab_size,params.target_vocab_size,
  								params.shuffle,params.output_weight_file,params.LSTM_size,
  								params.num_layers,params.multi_src_params.file_name,params.multi_src_params.int_file_name,
  								params.multi_src_params.source_model_name,params.ensemble_train_file_name,params.multi_src_params.ensemble_train_file_name);
  						}
  						else {
  							input_helper.prep_files_train_nonLM_multi_source(params.minibatch_size,params.longest_sent,
  								params.source_file_name,params.target_file_name,
  								params.train_file_name,params.source_vocab_size,params.target_vocab_size,
  								params.shuffle,params.output_weight_file,params.LSTM_size,
  								params.num_layers,params.multi_src_params.file_name,params.multi_src_params.int_file_name,
  								params.multi_src_params.source_model_name);
  						}
  					}
  					else if(!params.ensemble_train) {
  						success = input_helper.prep_files_train_nonLM(params.minibatch_size,params.longest_sent,
  							params.source_file_name,params.target_file_name,
  							params.train_file_name,params.source_vocab_size,params.target_vocab_size,
  							params.shuffle,params.output_weight_file,params.LSTM_size,params.num_layers,params.unk_replace,params.unk_aligned_width,params.attent_params.attention_model);
  					}
  					else {
  						success = input_helper.prep_files_train_nonLM_ensemble(params.minibatch_size,params.longest_sent,
  							params.source_file_name,params.target_file_name,
  							params.train_file_name,params.source_vocab_size,params.target_vocab_size,
  							params.shuffle,params.output_weight_file,params.LSTM_size,params.num_layers,params.ensemble_train_file_name,params.attent_params.attention_model);
  					}
          }

					//clean up if error
					if(!success) {
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}
				}
			}

			if(vm.count("parameter-range")) {

				if(lower_upper_range.size()!=2) {
					BZ_CUDA::logger << "ERROR: you must have two inputs to parameter-range\n1.lower bound\n2. upper bound\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}

				BZ_CUDA::lower = lower_upper_range[0];
				BZ_CUDA::upper = lower_upper_range[1];
				if(BZ_CUDA::lower >= BZ_CUDA::upper) {
					BZ_CUDA::logger << "ERROR: the lower parameter range cannot be greater than the upper range\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}
			}

			if(vm.count("fixed-halve-lr-full")) {
				params.stanford_learning_rate = true;
			}
				
			if(vm.count("fixed-halve-lr")) {
				params.google_learning_rate = true;
				if(params.epoch_to_start_halving<=0) {
					BZ_CUDA::logger << "ERROR: cannot halve learning rate until 1st epoch \n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}
			}

			if(vm.count("adaptive-halve-lr")) {
				params.learning_rate_schedule = true;
				if(vm.count("sequence")) {
					if(adaptive_learning_rate.size()!=1) {
						BZ_CUDA::logger << "ERROR: adaptive-halve-lr takes one arguement\n1.dev file name\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}
					params.dev_target_file_name = adaptive_learning_rate[0];
					params.test_file_name = params.unique_dir + "/validation.txt";

					input_file_prep input_helper;

          if(!params.char_params.char_cnn) {
  					input_helper.integerize_file_LM(params.output_weight_file,params.dev_target_file_name,params.test_file_name,
  						params.longest_sent,params.minibatch_size,true,params.LSTM_size,params.target_vocab_size,params.num_layers); 
          }
				}
				else {
					if(adaptive_learning_rate.size()!=2 && !params.multi_src_params.multi_source) {
						BZ_CUDA::logger << "ERROR: adaptive-halve-lr takes two arguements\n1.source dev file name\n2.target dev file name\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					if(adaptive_learning_rate.size()!=3 && params.multi_src_params.multi_source) {
						BZ_CUDA::logger << "ERROR: adaptive-halve-lr takes three arguements with multi-source\n1.source dev file name\n2.target dev file name\n3.other source dev file name\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					if(params.multi_src_params.multi_source) {
						params.multi_src_params.test_file_name = adaptive_learning_rate[2];
					}

					params.dev_source_file_name = adaptive_learning_rate[0];
					params.dev_target_file_name = adaptive_learning_rate[1];
					params.test_file_name = params.unique_dir + "/validation.txt";
					params.multi_src_params.int_file_name_test = params.unique_dir + params.multi_src_params.int_file_name_test;

          if(params.char_params.char_cnn) {
            params.train_file_name = params.char_params.word_train_file;
            params.test_file_name = params.char_params.word_dev_file;
          }

					if(params.dev_source_file_name == params.dev_target_file_name) {
						BZ_CUDA::logger << "ERROR: do not use the same file for source and target data\n";
						boost::filesystem::path temp_path(params.unique_dir);
						boost::filesystem::remove_all(temp_path);
						exit (EXIT_FAILURE);
					}

					input_file_prep input_helper;
          if(!params.char_params.char_cnn) {
  					input_helper.integerize_file_nonLM(params.output_weight_file,params.dev_source_file_name,
  						params.dev_target_file_name,params.test_file_name,
  						params.longest_sent,params.minibatch_size,params.LSTM_size,params.source_vocab_size,params.target_vocab_size,params.num_layers,
  						params.attent_params.attention_model,params.multi_src_params.multi_source,params.multi_src_params.test_file_name,params.multi_src_params.int_file_name_test,params.multi_src_params.source_model_name);
          }
				}


				if(vm.count("best-model")) {
					params.best_model = true;
				}
			}

			if(vm.count("truncated-softmax")) {
				params.shortlist_size = std::stoi(trunc_info[0]);
				params.sampled_size = std::stoi(trunc_info[1]);
				params.truncated_softmax = true;
				if(params.shortlist_size + params.sampled_size > params.target_vocab_size) {
					BZ_CUDA::logger << "ERROR: you cannot have shortlist size + sampled size >= target vocab size\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}
			}
            
            //put in the first line of the model file with the correct info
            //format:
            //0:     num_layers
            //1:     LSTM_size
            //2:     target_vocab_size
            //3:     source_vocab_size
            //4:     attention_model
            //5:     feed_input 
            //6:     multi_source
            //7:     combine_LSTM 
            //8:     char_cnn 
            
            add_model_info(params.num_layers,params.LSTM_size,params.target_vocab_size,params.source_vocab_size,params.attent_params.attention_model,params.attent_params.feed_input,\
                    params.multi_src_params.multi_source,params.multi_src_params.lstm_combine,params.char_params.char_cnn,params.output_weight_file);
			params.train= true;
			params.decode=false;
			params.test = false;
			params.stochastic_generation = false;
			return;
		}
        else { //checks here for things that should only be specified during training
            if(vm.count("train-source-RNN")) {
                std::cout << "Error train-source-RNN should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("train-target-RNN")) {
                std::cout << "Error train-target-RNN should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("train-source-input-embedding")) {
                std::cout << "Error train-source-input-embedding should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("train-target-input-embedding")) {
                std::cout << "Error train-target-input-embedding should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("train-target-output-embedding")) {
                std::cout << "Error train-target-output-embedding should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("train-train-attention-target-RNN")) {
                std::cout << "Error train-train-attention-target-RNN should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("vocab-mapping-file-multi-source")) {
                std::cout << "Error vocab-mapping-file-multi-source should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("multi-source")) {
                std::cout << "Error train-target-RNN should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("train-target-RNN")) {
                std::cout << "Error train-target-RNN should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("multi-attention")) {
                std::cout << "Error multi-attention should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("lstm-combine")) {
                std::cout << "Error lstm-combine should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("num-layers")) {
                std::cout << "Error num-layers should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("dropout")) {
                std::cout << "Error dropout should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("learning-rate")) {
                std::cout << "Error learning-rate should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("random-seed")) {
                std::cout << "Error random-seed should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("hiddenstate-size")) {
                std::cout << "Error hiddenstate-size should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("NCE")) {
                std::cout << "Error NCE should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("NCE-share-samples")) {
                std::cout << "Error NCE-share-samples should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("attention-model")) {
                std::cout << "Error attention-model should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("attention-width")) {
                std::cout << "Error attention-width should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("feed-input")) {
                std::cout << "Error feed-input should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("source-vocab-size")) {
                std::cout << "Error source-vocab-size should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("target-vocab-size")) {
                std::cout << "Error target-vocab-size should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("parameter-range")) {
                std::cout << "Error parameter-range should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("number-epochs")) {
                std::cout << "Error number-epochs should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("matrix-clip-gradients")) {
                std::cout << "Error matrix-clip-gradients should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("whole-clip-gradients")) {
                std::cout << "Error whole-clip-gradients should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("adaptive-halve-lr")) {
                std::cout << "Error adaptive-halve-lr should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("clip-cell")) {
                std::cout << "Error clip-cell should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("adaptive-decrease-factor")) {
                std::cout << "Error adaptive-decrease-factor should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("fixed-halve-lr")) {
                std::cout << "Error fixed-halve-lr should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("fixed-halve-lr-full")) {
                std::cout << "Error fixed-halve-lr-full should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("screen-print-rate")) {
                std::cout << "Error screen-print-rate should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
            if(vm.count("best-model")) {
                std::cout << "Error best-model should only be used during training (-t) or continue-training (-C)\n";
                exit (EXIT_FAILURE);
            }
    
        }

		if(vm.count("decode")) {
			if (kbest_files.size()<3) {
				BZ_CUDA::logger << "ERROR: at least 4 arguements must be entered for --decode, 1. number of best outputs\n"\
				" 2. neural network file name (this is the output file you get after training the neural network)\n"\
				" 3. output file name\n"\
				"Additionally more neural network file names can be added to do ensemble decoding\n";
				exit (EXIT_FAILURE);
			}

			//fill into NULL if the user did not specify anything
			if(params.decode_user_files_additional.size()==0) {
				for(int i=0; i<params.decode_user_files.size(); i++) {
					params.decode_user_files_additional.push_back("NULL");
				}
			}

			//once again fill in NULL if user did not specify
			if(params.model_names_multi_src.size()==0) {
				for(int i=0; i<params.decode_user_files.size(); i++) {
					params.model_names_multi_src.push_back("NULL");
				}	
			}

			boost::filesystem::path unique_path = boost::filesystem::unique_path();
      if(vm.count("tmp-dir-location")) {
        unique_path = boost::filesystem::path(params.tmp_location + unique_path.string());
      }
			BZ_CUDA::logger << "Temp directory being created named: " << unique_path.string() << "\n";
			boost::filesystem::create_directories(unique_path);
			params.unique_dir = unique_path.string();
      // if(vm.count("tmp-dir-location")) {
      //   params.unique_dir = params.tmp_location + params.unique_dir;
      // }

			//for ensembles
			for(int i=1; i<kbest_files.size()-1; i++) {
				params.model_names.push_back(kbest_files[i]);
				std::string temp_path = params.unique_dir+ "/kbest_tmp_" + std::to_string(i-1);
				params.decode_temp_files.push_back(temp_path);
				temp_path = params.unique_dir+ "/kbest_tmp_additional_" + std::to_string(i-1);
				params.decode_temp_files_additional.push_back(temp_path);
			}

      //BZ_CUDA::logger << "params.model_names: " << (int)params.model_names.size() << "\n";
      //BZ_CUDA::logger << "decode_user_files: " << (int)params.decode_user_files.size() << "\n";
      //BZ_CUDA::logger << "model_names_multi_src: " << (int)params.model_names_multi_src.size() << "\n";
			if(params.model_names.size() != params.decode_user_files.size() || params.model_names.size() != params.model_names_multi_src.size()) {
				BZ_CUDA::logger << "ERROR: the same number of inputs must be specified as models\n";
				exit (EXIT_FAILURE);
			}

			//params.decode_file_name = params.unique_dir+"/decoder_input.txt";
			params.decoder_output_file = params.unique_dir+"/decoder_output.txt";

			params.num_hypotheses =std::stoi(kbest_files[0]);
			//params.decode_tmp_file = kbest_files[1];
			//params.input_weight_file = model_names[0];
			params.decoder_final_file = kbest_files.back();

			input_file_prep input_helper;

			// input_helper.integerize_file_LM(params.input_weight_file,params.decode_tmp_file,"tmp/decoder_input.txt",
			// 	params.longest_sent,1,false,params.LSTM_size,params.target_vocab_size,true,params.source_vocab_size);
			for(int i=0; i<params.decode_temp_files.size(); i++) {
				input_helper.integerize_file_kbest(params.model_names[i],params.decode_user_files[i],params.decode_temp_files[i],
					params.longest_sent,params.target_vocab_size,false,"NULL");

				if(params.decode_user_files_additional[i]!= "NULL") {
					input_helper.integerize_file_kbest(params.model_names[i],params.decode_user_files_additional[i],params.decode_temp_files_additional[i],
						params.longest_sent,params.target_vocab_size,true,params.model_names_multi_src[i]);
				}
			}
		
			if(vm.count("multi-gpu")) {
				if(gpu_indicies.size()!=params.model_names.size()) {
					BZ_CUDA::logger << "ERROR: for decoding, each model must be specified a gpu\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}
				params.gpu_indicies = gpu_indicies;
			}
			else {
				for(int i=0; i<params.model_names.size(); i++) {
					params.gpu_indicies.push_back(0);
				}
			}

			if(params.beam_size<=0) {
				BZ_CUDA::logger << "ERROR: beam size cannot be <=0\n";
				boost::filesystem::path temp_path(params.unique_dir);
				boost::filesystem::remove_all(temp_path);
				exit (EXIT_FAILURE);
			}
			if(params.penalty<0) {
				BZ_CUDA::logger << "ERROR: penalty cannot be less than zero\n";
				boost::filesystem::path temp_path(params.unique_dir);
				boost::filesystem::remove_all(temp_path);
				exit (EXIT_FAILURE);
			}

			if(vm.count("Dump-LSTM")) {
				params.dump_LSTM=true;
			}

			if(vm.count("dec-ratio")) {
				if(decoding_ratio.size()!=2) {
					BZ_CUDA::logger << "Decoding ratio size: " << (int)decoding_ratio.size() << "\n";
					BZ_CUDA::logger << decoding_ratio[0] << "\n";
					BZ_CUDA::logger << "ERROR: only two inputs for decoding ratio\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}
				params.min_decoding_ratio = decoding_ratio[0];
				params.max_decoding_ratio = decoding_ratio[1];
				if(params.min_decoding_ratio >= params.max_decoding_ratio) {
					BZ_CUDA::logger << "ERROR: min decoding ratio must be <= max_decoding_ratio\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}
			}

			params.train = false;
			params.decode = true;
			params.test = false;
			params.stochastic_generation = false;
			params.LM = false;
			return;
		}

		if(vm.count("force-decode")) {
			BZ_CUDA::force_decode = true;
			if(vm.count("multi-gpu")) {
				params.gpu_indicies = gpu_indicies;
			}
			boost::filesystem::path unique_path = boost::filesystem::unique_path();
      if(vm.count("tmp-dir-location")) {
        unique_path = boost::filesystem::path(params.tmp_location + unique_path.string());
      }
			BZ_CUDA::logger << "Temp directory being created named: " << unique_path.string() << "\n";
			boost::filesystem::create_directories(unique_path);
			params.unique_dir = unique_path.string();
      // if(vm.count("tmp-dir-location")) {
      //   params.unique_dir = params.tmp_location + params.unique_dir;
      // }
			params.test_file_name = params.unique_dir + "/validation.txt";

			if(vm.count("sequence")) {
				if(test_files.size()!=3) {
					BZ_CUDA::logger << "ERROR: force-decode takes three arguements 1.input file name (input sentences)"\
					"2. neural network file name 3.output file name \n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}

				params.attent_params.attention_model = false;
				params.target_file_name = test_files[0];
				params.input_weight_file = test_files[1];
				params.output_force_decode = test_files[2];
				params.LM = true;

				input_file_prep input_helper;

				input_helper.integerize_file_LM(params.input_weight_file,params.target_file_name,params.test_file_name,
					params.longest_sent,params.minibatch_size,false,params.LSTM_size,params.target_vocab_size,params.num_layers);

			}
			else {
				if(test_files.size()!=4) {
					BZ_CUDA::logger << "ERROR: force-decode takes four arguements: 1. source input file"\
					" 2. target input file  3. neural network file name 4. output file name\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}

				params.LM = false;
				params.source_file_name = test_files[0];
				params.target_file_name = test_files[1];
				params.input_weight_file = test_files[2];
				params.output_force_decode = test_files[3];

				//stuff for attention model alignments
				params.attent_params.tmp_alignment_file = params.unique_dir + "/alignments.txt";

				if(params.source_file_name == params.target_file_name) {
					BZ_CUDA::logger << "ERROR: do not use the same file for source and target data\n";
					boost::filesystem::path temp_path(params.unique_dir);
					boost::filesystem::remove_all(temp_path);
					exit (EXIT_FAILURE);
				}


				if(vm.count("multi-source")) {
					if(multi_source.size()!=2) {
						BZ_CUDA::logger << "ERROR only two arguements for the multi-source flag\n";
						exit (EXIT_FAILURE);
					}
					params.multi_src_params.multi_source = true;
					params.multi_src_params.test_file_name = multi_source[0];
					params.multi_src_params.source_model_name = multi_source[1];
					params.multi_src_params.int_file_name_test = params.unique_dir + params.multi_src_params.int_file_name_test;
				}


        if(!params.char_params.char_cnn) {
  				input_file_prep input_helper;
  				input_helper.integerize_file_nonLM(params.input_weight_file,params.source_file_name,
  					params.target_file_name,params.test_file_name,params.longest_sent,1,params.LSTM_size,
  					params.source_vocab_size,params.target_vocab_size,params.num_layers,params.attent_params.attention_model,
  					params.multi_src_params.multi_source,params.multi_src_params.test_file_name,params.multi_src_params.int_file_name_test,
  					params.multi_src_params.source_model_name);
        }
        else {
          params.test_file_name = params.char_params.word_dev_file;
        }

				params.minibatch_size=1;
			}

            std::ifstream tmp_if_stream(params.input_weight_file.c_str());
            std::string tmp_str;
            std::string tmp_word;
            std::getline(tmp_if_stream,tmp_str);
            std::istringstream my_ss(tmp_str,std::istringstream::in);
            std::vector<std::string> tmp_model_params;
            while(my_ss >> tmp_word) {
                tmp_model_params.push_back(tmp_word);
            }
            if(tmp_model_params.size() != 9) {
                BZ_CUDA::logger << "Error: the model file is not in the correct format for force-decode\n";
                exit (EXIT_FAILURE);
            }
            params.num_layers = std::stoi(tmp_model_params[0]);
            params.LSTM_size = std::stoi(tmp_model_params[1]);
            params.target_vocab_size = std::stoi(tmp_model_params[2]);
        	params.source_vocab_size = std::stoi(tmp_model_params[3]);
        	params.attent_params.attention_model = std::stoi(tmp_model_params[4]);
        	params.attent_params.feed_input = std::stoi(tmp_model_params[5]);
        	params.multi_src_params.multi_source = std::stoi(tmp_model_params[6]);
        	params.multi_src_params.lstm_combine = std::stoi(tmp_model_params[7]);
        	params.char_params.char_cnn = std::stoi(tmp_model_params[8]);

			params.train= false;
			params.decode=false;
			params.test = true;
			// params.minibatch_size=1;
			params.stochastic_generation = false;
			return;
		}

		if(vm.count("stoch-gen")) {
			if(!vm.count("sequence")) {
				BZ_CUDA::logger << "ERROR: you can only do stoch-gen on the sequence model\n";
				exit (EXIT_FAILURE);
			}

			if(stoicgen_files.size()!=2) {
				BZ_CUDA::logger << "ERROR: stoch-gen takes two inputs"\
				" 1. neural network file name 2. output file name\n";
				exit (EXIT_FAILURE);
			}

			boost::filesystem::path unique_path = boost::filesystem::unique_path();
      if(vm.count("tmp-dir-location")) {
        unique_path = boost::filesystem::path(params.tmp_location + unique_path.string());
      }
			BZ_CUDA::logger << "Temp directory being created named: " << unique_path.string() << "\n";
			boost::filesystem::create_directories(unique_path);
			params.unique_dir = unique_path.string();
      // if(vm.count("tmp-dir-location")) {
      //   params.unique_dir = params.tmp_location + params.unique_dir;
      // }
			params.sg_output_file_temp = params.unique_dir + "/sg.txt";

			params.input_weight_file = stoicgen_files[0];
			params.sg_output_file = stoicgen_files[1];

			std::ifstream weights_file;
			std::vector<std::string> info;
			std::string str;
			std::string word;
			weights_file.open(params.input_weight_file.c_str());
			weights_file.seekg(0, std::ios::beg);
			std::getline(weights_file, str); //info from first sentence
			std::istringstream iss(str, std::istringstream::in);
			while(iss >> word) {
				info.push_back(word);
			}
			weights_file.close();

			params.LSTM_size = std::stoi(info[1]);
			params.target_vocab_size = std::stoi(info[2]);

			params.LM = true;
			params.train= false;
			params.decode = false;
			params.test = false;
			params.minibatch_size = 1;
			params.stochastic_generation = true;
			return;
		}
	}
	catch(po::error& e) { 
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
    	//std::cerr << desc << std::endl;
    	exit (EXIT_FAILURE);
    }
}

void myexitfunc(void) {

}


int main(int argc, char **argv) {

	//Timing stuff
	std::chrono::time_point<std::chrono::system_clock> start_total,
	end_total, begin_minibatch,end_minibatch,begin_decoding,end_decoding,begin_epoch;
	std::chrono::duration<double> elapsed_seconds;

  start_total = std::chrono::system_clock::now();

    //Initializing the model
	global_params params; //Declare all of the global parameters

    //create tmp directory if it does not exist already
	// if( !(boost::filesystem::exists("tmp/"))) {
	//     std::cout << "Creating tmp directory for program\n";
	//     boost::filesystem::create_directory("tmp/");
	// }

  //atexit(); //this is used to clean up the end of the code

	//file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file); //Initialize the file information

	BZ_CUDA::curr_seed = static_cast<unsigned int>(std::time(0));
	BZ_CUDA::curr_seed = std::min((unsigned int)100000000,BZ_CUDA::curr_seed);//to prevent overflow

	//get the command line arguements
	command_line_parse(params,argc,argv);

  // if(params.HPC_output) {
  //   std::cout << "Opening logfile: " <<  params.HPC_output_file_name << "\n";
  //   HPC_output.open(params.HPC_output_file_name);
  // }

	//randomize the seed
	if(params.random_seed) {
    BZ_CUDA::gen.seed(static_cast<unsigned int>(params.random_seed_int));
	}
  else {
    BZ_CUDA::gen.seed(static_cast<unsigned int>(std::time(0)));
  }

	neuralMT_model<precision> model; //This is the model
	printIntroMessage(params);


	if(!params.decode) {
		model.initModel(params.LSTM_size,params.minibatch_size,params.source_vocab_size,params.target_vocab_size,
			params.longest_sent,params.debug,params.learning_rate,params.clip_gradient,params.norm_clip,
			params.input_weight_file,params.output_weight_file,params.softmax_scaled,params.train_perplexity,params.truncated_softmax,
			params.shortlist_size,params.sampled_size,params.LM,params.num_layers,params.gpu_indicies,params.dropout,
 			params.dropout_rate,params.attent_params,params);
	}

	if(params.load_model_train) {
		std::string temp_swap_weights = model.input_weight_file;
		model.input_weight_file = params.load_model_name;
		model.load_weights();
		model.input_weight_file = temp_swap_weights;
	}

	////////////////////////////////////Train the model//////////////////////////////////////
	if(params.train) {
		//info for averaging the speed
		int curr_batch_num_SPEED = 0;
		const int thres_batch_num_SPEED = params.screen_print_rate;//set this to whatever
		int total_words_batch_SPEED = 0;
		double total_batch_time_SPEED = 0;

		//File info for the training file
		file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size,params.train_total_words,params.truncated_softmax,
			params.shortlist_size,params.sampled_size,params.char_params,params.char_params.char_train_file); //Initialize the file information


		//model.initFileInfo(&file_info);
		params.half_way_count = params.train_total_words/2;
		if(params.google_learning_rate) {
			BZ_CUDA::logger << "Number of words at which to start halving the learning rate: " << params.half_way_count << "\n";
			// if(params.HPC_output) {
			// 	HPC_output << "Words at which to start halving the learning rate: " << params.half_way_count << "\n";
			// 	HPC_output.flush();
			// }
		}
		int current_epoch = 1;
		BZ_CUDA::logger << "Starting model training\n";
    BZ_CUDA::logger << "-----------------------------------"  << "\n";
		BZ_CUDA::logger << "Starting epoch 1\n";
    BZ_CUDA::logger << "-----------------------------------"  << "\n";
		// if(params.HPC_output) {
		// 		HPC_output << "Starting model training\n";
		// 		HPC_output << "Starting epoch 1\n";
		// 		HPC_output.flush();
		// }

	
		//stuff for learning rate schedule
		int total_words = 0;
		precision temp_learning_rate = params.learning_rate; //This is only for the google learning rate
		bool learning_rate_flag =true;//used for google learning rate for halving at every 0.5 epochs
		double old_perplexity = 0;
		model.train_perplexity = 0; //set the model perplexity to zero
    begin_epoch = std::chrono::system_clock::now();
		while(current_epoch <= params.num_epochs) {
			begin_minibatch = std::chrono::system_clock::now();
			bool success = file_info.read_minibatch();
			if(model.multi_source) {
				model.src_fh.read_minibatch();
			}
			end_minibatch = std::chrono::system_clock::now();
			elapsed_seconds = end_minibatch-begin_minibatch;
			//std::cout << "File I/O time: " << elapsed_seconds.count()/60.0 << " minutes\n";
			total_batch_time_SPEED+= elapsed_seconds.count();

			begin_minibatch = std::chrono::system_clock::now();

			//cudaProfilerStart();
			model.initFileInfo(&file_info);
			model.compute_gradients(file_info.minibatch_tokens_source_input,file_info.minibatch_tokens_source_output,
				file_info.minibatch_tokens_target_input,file_info.minibatch_tokens_target_output,
				file_info.h_input_vocab_indicies_source,file_info.h_output_vocab_indicies_source,
				file_info.h_input_vocab_indicies_target,file_info.h_output_vocab_indicies_target,
				file_info.current_source_length,file_info.current_target_length,
				file_info.h_input_vocab_indicies_source_Wgrad,file_info.h_input_vocab_indicies_target_Wgrad,
				file_info.len_source_Wgrad,file_info.len_target_Wgrad,file_info.h_sampled_indices,
				file_info.len_unique_words_trunc_softmax,file_info.h_batch_info,&file_info);

			//cudaProfilerStop();
			//return;
			// return 0;

			end_minibatch = std::chrono::system_clock::now();
			elapsed_seconds = end_minibatch-begin_minibatch;

			total_batch_time_SPEED+= elapsed_seconds.count();
			total_words_batch_SPEED+=file_info.words_in_minibatch;

			if(curr_batch_num_SPEED>=thres_batch_num_SPEED) {
				BZ_CUDA::logger << "Recent batch gradient L2 norm size (if using -w): " << BZ_CUDA::global_norm << "\n";
				BZ_CUDA::logger << "Time to compute gradients for previous " << params.screen_print_rate << " minibatches: "  << total_batch_time_SPEED/60.0 << " minutes\n";
				BZ_CUDA::logger << "Number of words in previous " << params.screen_print_rate << " minibatches: "  << total_words_batch_SPEED << "\n";
				BZ_CUDA::logger << "Throughput for previous " << params.screen_print_rate << " minibatches: " << (total_words_batch_SPEED)/(total_batch_time_SPEED) << " words per second\n";
				BZ_CUDA::logger << total_words << " words out of " << params.train_total_words << " epoch: " << current_epoch <<  "\n\n";
				// if(params.HPC_output) {
				// 	HPC_output << "Recent batch gradient L2 norm size: " << BZ_CUDA::global_norm << "\n";
				// 	HPC_output << "Batched Minibatch time: " << total_batch_time_SPEED/60.0 << " minutes\n";
				// 	HPC_output << "Batched Words in minibatch: " << total_words_batch_SPEED << "\n";
				// 	HPC_output << "Batched Throughput: " << (total_words_batch_SPEED)/(total_batch_time_SPEED) << " words per second\n";
				// 	HPC_output << total_words << " out of " << params.train_total_words << " epoch: " << current_epoch <<  "\n\n";
				// 	HPC_output.flush();
				// }
				total_words_batch_SPEED = 0;
				total_batch_time_SPEED = 0;
				curr_batch_num_SPEED = 0;

			}
			curr_batch_num_SPEED++;
			total_words += file_info.words_in_minibatch;

			//stuff for google learning rate
			if(params.google_learning_rate && current_epoch>=params.epoch_to_start_halving && total_words>=params.half_way_count &&
				learning_rate_flag) {
					temp_learning_rate = temp_learning_rate/2;
					BZ_CUDA::logger << "New Learning Rate: " << temp_learning_rate << "\n";
					model.update_learning_rate(temp_learning_rate);
					learning_rate_flag = false;
					// if(params.HPC_output) {
					// 	HPC_output << "New Learning Rate: " << temp_learning_rate << "\n";
					// 	HPC_output.flush();
					// }
			}

			//stuff for perplexity based learning schedule
			if(params.learning_rate_schedule && total_words>=params.half_way_count &&learning_rate_flag) {
				learning_rate_flag = false;
				double new_perplexity = model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
					params.source_vocab_size,params.target_vocab_size,false,params.test_total_words,params.HPC_output,false,"");
				BZ_CUDA::logger << "Old dev set Perplexity: " << old_perplexity << "\n";
				BZ_CUDA::logger << "New dev set Perplexity: " << new_perplexity << "\n";
				// if(params.HPC_output) {
				// 	HPC_output << "Old dev set Perplexity: " << old_perplexity << "\n";
				// 	HPC_output << "New dev set Perplexity: " << new_perplexity << "\n";
				// 	HPC_output.flush();
				// }
				if ( (new_perplexity + params.margin >= old_perplexity) && current_epoch!=1) {
					temp_learning_rate = temp_learning_rate*params.decrease_factor;
					model.update_learning_rate(temp_learning_rate);
					BZ_CUDA::logger << "New learning rate:" << temp_learning_rate <<"\n\n";
					// if(params.HPC_output) {
					// 	HPC_output << "New learning rate:" << temp_learning_rate <<"\n\n";
					// 	HPC_output.flush();
					// }
				}
				//perplexity is better so output the best model file
				if((params.best_model && params.best_model_perp > new_perplexity) || BZ_CUDA::dump_every_best) {
					//BZ_CUDA::logger << "Writing model file: "<< params.best_model_file_name <<"\n";
					model.dump_best_model(params.best_model_file_name,params.output_weight_file);
					// if(params.HPC_output) {
					// 		HPC_output << "Now outputting the new best model\n";
					// 		HPC_output.flush();
					// }
					params.best_model_perp = new_perplexity;
				}
			
				old_perplexity = new_perplexity;
			}

			if(!success) {
				current_epoch+=1;
				//stuff for google learning rate schedule
				if(params.google_learning_rate && current_epoch>=params.epoch_to_start_halving) {
					temp_learning_rate = temp_learning_rate/2;
					BZ_CUDA::logger << "New learning rate:" << temp_learning_rate <<"\n\n";
					model.update_learning_rate(temp_learning_rate);
					learning_rate_flag = true;
					// if(params.HPC_output) {
					// 	HPC_output << "New learning rate:" << temp_learning_rate <<"\n\n";
					// 	HPC_output.flush();
					// }
				}

				//stuff for stanford learning rate schedule
				if(params.stanford_learning_rate && current_epoch>=params.epoch_to_start_halving_full) {
					temp_learning_rate = temp_learning_rate/2;
					BZ_CUDA::logger << "New learning rate:" << temp_learning_rate <<"\n\n";
					model.update_learning_rate(temp_learning_rate);
					learning_rate_flag = true;
					// if(params.HPC_output) {
					// 	HPC_output << "New learning rate:" << temp_learning_rate <<"\n\n";
					// 	HPC_output.flush();
					// }
				}

				double new_perplexity;
				if(params.learning_rate_schedule) {
					new_perplexity = model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
						params.source_vocab_size,params.target_vocab_size,false,params.test_total_words,params.HPC_output,false,"");
				}
				//stuff for perplexity based learning schedule
				if(params.learning_rate_schedule) {
					BZ_CUDA::logger << "Old dev set Perplexity: " << old_perplexity << "\n";
					BZ_CUDA::logger << "New dev set Perplexity: " << new_perplexity << "\n";
					// if(params.HPC_output) {
					// 	HPC_output << "Old dev set Perplexity: " << old_perplexity << "\n";
					// 	HPC_output << "New dev set Perplexity: " << new_perplexity << "\n";
					// 	HPC_output.flush();
					// }
					if ( (new_perplexity + params.margin >= old_perplexity) && current_epoch!=1) {
						temp_learning_rate = temp_learning_rate*params.decrease_factor;
						model.update_learning_rate(temp_learning_rate);
						BZ_CUDA::logger << "New learning rate:" << temp_learning_rate <<"\n\n";
						// if(params.HPC_output) {
						// 	HPC_output << "New learning rate:" << temp_learning_rate <<"\n\n";
						// 	HPC_output.flush();
						// }
					}

					//perplexity is better so output the best model file
					if( (params.best_model && params.best_model_perp > new_perplexity) || BZ_CUDA::dump_every_best) {
						//BZ_CUDA::logger << "Now outputting the new best model\n";
						model.dump_best_model(params.best_model_file_name,params.output_weight_file);
						// if(params.HPC_output) {
						// 		HPC_output << "Now outputting the new best model\n";
						// 		HPC_output.flush();
						// }
						params.best_model_perp = new_perplexity;
					}

					learning_rate_flag = true;
					old_perplexity = new_perplexity;
				}

				if(params.train_perplexity) {
					model.train_perplexity = model.train_perplexity/std::log(2.0);
					BZ_CUDA::logger << "PData on train set: "  << model.train_perplexity << "\n";
					BZ_CUDA::logger << "Total target words: " << file_info.total_target_words << "\n";
					BZ_CUDA::logger << "Training set perplexity: " << std::pow(2,-1*model.train_perplexity/file_info.total_target_words) << "\n";
					// if(params.HPC_output) {
					// 	HPC_output << "Training set perplexity: " << std::pow(2,-1*model.train_perplexity/file_info.total_target_words) << "\n";
					// 	HPC_output.flush();
					// }
					model.train_perplexity = 0;
				}

				total_words=0;
				if(current_epoch <= params.num_epochs) {
                    elapsed_seconds = std::chrono::system_clock::now() - begin_epoch;
          BZ_CUDA::logger << "Previous Epoch time (minutes): " << (double)elapsed_seconds.count()/60.0 << "\n";
          begin_epoch = std::chrono::system_clock::now();
					BZ_CUDA::logger << "-----------------------------------"  << "\n";
					BZ_CUDA::logger << "Starting epoch " << current_epoch << "\n";
					BZ_CUDA::logger << "-----------------------------------"  << "\n";
					// if(params.HPC_output) {
					// 	HPC_output << "-----------------------------------"  << std::endl;
					// 	HPC_output << "Starting epoch " << current_epoch << std::endl;
					// 	HPC_output << "-----------------------------------"  << std::endl;
					// 	HPC_output.flush();
					// }
				}
			}
			devSynchAll();
		}	
		//Now that training is done, dump the weights
		devSynchAll();
		model.dump_weights();
	}


	/////////////////////////////////Get perplexity on test set////////////////////////////////
	if(params.test) {
		model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size,true,params.test_total_words,params.HPC_output,true,params.output_force_decode);
		//now unint alignments
		if(model.attent_params.dump_alignments) {
			input_file_prep input_helper;
			model.output_alignments.close();
			//input_helper.unint_alignments(params.input_weight_file,params.attent_params.tmp_alignment_file,params.attent_params.alignment_file);
		}
	}
  

	if(params.LM && params.stochastic_generation) {
		model.stoicastic_generation(params.sg_length,params.sg_output_file_temp,params.temperature);
		input_file_prep input_helper;
		input_helper.unint_file(params.input_weight_file,params.sg_output_file_temp,params.sg_output_file,true,false);
	}


	///////////////////////////////////////////decode the model////////////////////////////////////////////
	if(params.decode) {
		//std::cout << "-----------------Starting Decoding----------------\n";
		begin_decoding = std::chrono::system_clock::now();
		ensemble_factory<precision> ensemble_decode(params.model_names,params.num_hypotheses,params.beam_size, params.min_decoding_ratio,
			params.penalty, params.longest_sent,params.print_score,
			params.decoder_output_file,params.gpu_indicies,params.max_decoding_ratio,
			params.target_vocab_size,params);
        
        if (params.fsa_file != ""){
            fsa* fsa_model = new fsa(params.fsa_file);
            input_file_prep input_helper;
            input_helper.load_word_index_mapping(params.model_names[0],false,true);
            
            ensemble_decode.model_decoder->init_fsa(fsa_model, input_helper.tgt_mapping, params);
            // encourage list

            params.encourage_weight.clear();
            std::vector<std::string> ll = split(params.encourage_weight_str,',');
            for (std::string s: ll){
                float f = std::stof(s);
                params.encourage_weight.push_back(f);
            }

            ensemble_decode.model_decoder->init_encourage_lists(params.encourage_list, params.encourage_weight);
            
        }

		BZ_CUDA::logger << "-----------------Starting Decoding----------------\n";
		ensemble_decode.decode_file();

		end_decoding = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_decoding-begin_decoding;
		BZ_CUDA::logger << "Decoding time: " << elapsed_seconds.count()/60.0 << " minutes\n";

		//now unintegerize the file
		input_file_prep input_helper;
		//use model_names[0] since all models must have the same target vocab mapping and size
		input_helper.unint_file(params.model_names[0],params.decoder_output_file,params.decoder_final_file,false,true);
	}



	//remove the temp directory created
	if(params.unique_dir!="NULL") {
		boost::filesystem::path temp_path(params.unique_dir);
		//boost::filesystem::remove_all(temp_path);
	}

	//Compute the final runtime
	end_total = std::chrono::system_clock::now();
	elapsed_seconds = end_total-start_total;
  BZ_CUDA::logger << "\n\n\n";
  BZ_CUDA::logger << "Total Program Runtime: " << (double)elapsed_seconds.count()/60.0 << " minutes" << "\n";
}
