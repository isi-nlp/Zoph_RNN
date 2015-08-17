//STL includes
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <chrono>
#include <iomanip>

//Eigen includes
#include <Eigen/Dense>
#include <Eigen/Sparse>

//boost
#include "boost/program_options.hpp" 

//My own includes
#include "global_params.h"
#include "input_file_prep.h"
#include "BZ_CUDA_UTIL.h"
#include "base_layer.h"
#include "gpu_info_struct.h"
#include "custom_kernels.h"
#include "model.h"
#include "fileHelper.h"
#include "Eigen_Util.h"
#include "model.hpp"
#include "base_layer.hpp"
#include "LSTM.hpp"
#include "softmax.hpp"
#include "Input_To_Hidden_Layer.hpp"
#include "Hidden_To_Hidden_Layer.hpp"
#include "LSTM_HH.hpp"

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

	//basic format setup
	namespace po = boost::program_options; 
	po::options_description desc("Options");
	desc.add_options() 
  		("help,h", "Run to get help on how to use the program")
  		("train,t",po::value<std::vector<std::string> > (&train_files)->multitoken(),"Train a model with input data file(s) and a name for the neural network output file"\
  			". \nFORMAT (if sequence to sequence): <source file name> <target file name> <neural network output name> "\
  			" \nFORMAT (if sequence): <target file name> <neural network output name>")
  		("force-decode,f",po::value<std::vector<std::string> > (&test_files)->multitoken(), "Get per line probability of dataset plus the perplexity\n"\
  			"FORMAT: (if sequence to sequence): <source file name> <target file name> <trained neural network file name> <output file name>\n"\
  			"FORMAT: (if sequence): <target file name> <trained neural network file name> <output file name>")
  		("stoch-gen,g", po::value<std::vector<std::string> > (&stoicgen_files)->multitoken(),"Do random generation for a sequence model, such as a language model\n"\
  			"FORMAT: <neural network file name> <output file name>")
  		("stoch-gen-len",po::value<int>(&params.sg_length) ,"How many tokens to let stoch-gen run for")
  		("sequence,s", "Train model that learns a sequence,such as language modeling. Default model is sequence to sequence model")
  		("learning-rate,l",po::value<precision>(&params.learning_rate),"Set learning rate")
  		("longest-sent,L",po::value<int>(&params.longest_sent),"Set the maximum sentence length for training.")
  		("hiddenstate-size,H",po::value<int>(&params.LSTM_size),"Set hiddenstate size")
  		("truncated-softmax,T",po::value<int>(&params.LSTM_size),"Set hiddenstate size\n"\
  			"FORMAT: <shortlist size> <sampled size>")
  		("source-vocab,v",po::value<int>(&params.source_vocab_size),"Set source vocab size")
  		("target-vocab,V",po::value<int>(&params.target_vocab_size),"Set target vocab size")
  		("shuffle",po::value<bool>(&params.shuffle),"true if you want to shuffle the train data")
  		("parameter-range,p",po::value<std::vector<precision> > (&lower_upper_range)->multitoken(),"parameter initialization range")
  		("number-epochs,n",po::value<int>(&params.num_epochs),"Set number of epochs")
  		("clip-gradients,c",po::value<precision>(&params.norm_clip),"Set gradient clipping threshold")
  		("adaptive-halve-lr",po::value<std::vector<std::string>> (&adaptive_learning_rate)->multitoken(),"Halve the learning rate"\
  			" when the perplexity on your specified dev set decreases from the previous half epoch\n"\
  			"FORMAT:")
  		("fixed-halve-lr",po::value<int> (&params.epoch_to_start_halving),"Halve the learning rate"\
  			" after a certain epoch, every half epoch afterwards by a specific amount")
  		("minibatch-size,m",po::value<int>(&params.minibatch_size),"Set minibatch size")
  		("kbest,k",po::value<std::vector<std::string> > (&kbest_files)->multitoken(),"Get k best paths in sequence to sequence model\n"\
  			"FORMAT: <how many paths> <source file name> <neural network file name> <output file name>") 
  		("beam-size,b",po::value<int>(&params.beam_size),"Set beam size for kbest paths")
  		("penalty,p",po::value<precision>(&params.penalty),"Set penalty for kbest decoding. The value entered"\
  			" will be added to the log probability score per target word decoded. This can make the model favor longer sentences for decoding");

    po::variables_map vm; 

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		
		//see if the user specified the help flag
		if ( vm.count("help") ) {

			std::cout << "\n------------------------------\n";
			std::cout << "This is Barret Zoph's CPU/GPU RNN library\n"
            << "The flags for the command line interface are below\n" 
            << "" << "\n";

			std::cout << desc << "\n";
			exit (EXIT_FAILURE);
		}

		//error checks to be sure only once of these options is set
		if (vm.count("train") && vm.count("kbest")) {
			std::cout << "ERROR: you cannot train and get kbest at the same time\n";
			exit (EXIT_FAILURE);
		}
		if (vm.count("train") && vm.count("force-decode")) {
			std::cout << "ERROR: you cannot train and force-decode at the same time\n";
			exit (EXIT_FAILURE);
		}
		if (vm.count("force-decode") && vm.count("kbest")) {
			std::cout << "ERROR: you cannot force-decode and get kbest at the same time\n";
			exit (EXIT_FAILURE);
		}
		if (!(vm.count("train") || vm.count("force-decode") || vm.count("kbest")||vm.count("stoch-gen") )) {
			std::cout << "ERROR: you must either train,get kbest,stoch generate data or force-decode\n";
			exit (EXIT_FAILURE);
		}

		if(vm.count("train")) {

			//some basic error checks to parameters
			if(params.learning_rate<=0) {
				std::cout << "ERROR: you cannot have a learning rate <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.minibatch_size<=0) {
				std::cout << "ERROR: you cannot have a minibatch of size <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.LSTM_size<=0) {
				std::cout << "ERROR: you cannot have a hiddenstate of size <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.source_vocab_size<=0) {
				if(params.source_vocab_size!=-1) {
					std::cout << "ERROR: you cannot have a source_vocab_size <=0\n";
					exit (EXIT_FAILURE);
				}
			}
			if(params.target_vocab_size<=0) {
				if(params.target_vocab_size!=-1) {
					std::cout << "ERROR: you cannot have a target_vocab_size <=0\n";
					exit (EXIT_FAILURE);
				}
			}
			if(params.norm_clip<=0) {
				std::cout << "ERROR: you cannot have your norm clip <=0\n";
				exit (EXIT_FAILURE);
			}

			if(params.num_epochs<=0) {
				std::cout << "ERROR: you cannot have num_epochs <=0\n";
				exit (EXIT_FAILURE);
			}

			if(vm.count("truncated-softmax")) {
				params.shortlist_size = std::stoi(trunc_info[0]);
				params.sampled_size = std::stoi(trunc_info[1]);
				params.truncated_softmax = true;
			}


			//now create the necessary files
			if(vm.count("sequence")) {
				
				if(train_files.size()!=2) {
					std::cout << "ERROR: two arguements to be supplied to the train flag"\
					" 1. train data file name, 2. neural network output name\n";
					exit (EXIT_FAILURE);
				}

				params.LM = true;
				params.target_file_name = train_files[0];
				params.output_weight_file = train_files[1];

				input_file_prep input_helper;

				//this outputs the train.txt file along with the mappings and first line
				input_helper.prep_files_train_LM(params.minibatch_size,params.longest_sent,
					params.target_file_name,
					"tmp/train.txt",params.target_vocab_size,
					params.shuffle,params.output_weight_file,params.LSTM_size);
			}
			else {
				//then sequence to sequence model
				if(train_files.size()!=3) {
					std::cout << train_files.size() <<"\n";
					std::cout << "ERROR: three arguements to be supplied to the train flag for the sequence to sequence model\n"\
					" 1. source train data file name\n 2. target train data file name \n3. neural network output name\n";
					exit (EXIT_FAILURE);
				}

				params.LM = false;
				params.source_file_name = train_files[0];
				params.target_file_name = train_files[1];
				params.output_weight_file = train_files[2];

				input_file_prep input_helper;

				input_helper.prep_files_train_nonLM(params.minibatch_size,params.longest_sent,
					params.source_file_name,params.target_file_name,
					"tmp/train.txt",params.source_vocab_size,params.target_vocab_size,
					params.shuffle,params.output_weight_file,params.LSTM_size);

			}

			if(vm.count("parameter-range")) {

				if(lower_upper_range.size()!=2) {
					std::cout << "ERROR: you must have two inputs to parameter-range\n1.lower bound\n2. upper bound\n";
				}

				precision temp_lower = lower_upper_range[0];
				precision temp_upper = lower_upper_range[1];
				if(temp_lower >= temp_upper) {
					std::cout << "ERROR: the lower parameter range cannot be greater than the upper range\n"; 
					exit (EXIT_FAILURE);
				}
				params.lower_range = temp_lower;
				params.upper_range = temp_upper;
			}
				
			if(vm.count("fixed-halve-lr")) {
				params.google_learning_rate = true;
				if(params.epoch_to_start_halving<=0) {
					std::cout << "ERROR: cannot halve learning rate until 1st epoch \n";
					exit (EXIT_FAILURE);
				}
			}

			if(vm.count("adaptive-halve-lr")) {
				params.learning_rate_schedule = true;
				if(vm.count("sequence")) {
					if(adaptive_learning_rate.size()!=1) {
						std::cout << "ERROR: adaptive-halve-lr takes one arguement\n1.dev file name\n";
						exit (EXIT_FAILURE);
					}
					params.dev_target_file_name = adaptive_learning_rate[0];

					input_file_prep input_helper;

					input_helper.integerize_file_LM(params.output_weight_file,params.dev_target_file_name,"tmp/validation.txt",
						params.longest_sent,params.minibatch_size,true,params.LSTM_size,params.target_vocab_size,false,params.source_vocab_size); 

				}
				else {
					if(adaptive_learning_rate.size()!=2) {
						std::cout << "ERROR: adaptive-halve-lr takes two arguements\n1.source dev file name\n2.target dev file name\n";
						exit (EXIT_FAILURE);
					}
					params.dev_source_file_name = adaptive_learning_rate[0];
					params.dev_target_file_name = adaptive_learning_rate[1];

					input_file_prep input_helper;

					input_helper.integerize_file_nonLM(params.output_weight_file,params.dev_source_file_name,
						params.dev_target_file_name,"tmp/validation.txt",
						params.longest_sent,params.minibatch_size,params.LSTM_size,params.source_vocab_size,params.target_vocab_size);
				}
			}

			params.train= true;
			params.decode=false;
			params.test = false;
			params.stochastic_generation = false;
			return;
		}

		if(vm.count("kbest")) {
			if (kbest_files.size()!=4) {
				std::cout << "ERROR: 4 arguements must be entered for kbest, 1. number of best paths"\
				" 2 input file name "
				" 3. neural network file name (this is the output file you get after training the neural network)"\
				" 4. output file name\n";
				exit (EXIT_FAILURE);
			}

			params.num_hypotheses =std::stoi(kbest_files[0]);
			params.decode_tmp_file = kbest_files[1];
			params.input_weight_file = kbest_files[2];
			params.decoder_final_file = kbest_files[3];

			input_file_prep input_helper;

			input_helper.integerize_file_LM(params.input_weight_file,params.decode_tmp_file,"tmp/decoder_input.txt",
				params.longest_sent,1,false,params.LSTM_size,params.target_vocab_size,true,params.source_vocab_size);

			if(params.beam_size<=0) {
				std::cout << "ERROR: beam size cannot be <=0\n";
				exit (EXIT_FAILURE);
			}
			if(params.penalty<0) {
				std::cout << "ERROR: penalty cannot be less than zero\n";
				exit (EXIT_FAILURE);
			}
			params.train = false;
			params.decode = true;
			params.test = false;
			params.stochastic_generation = false;
			params.LM = false;
			return;
		}

		if(vm.count("force-decode")) {
			if(vm.count("sequence")) {
				if(test_files.size()!=3) {
					std::cout << "ERROR: force-decode takes three arguements 1.input file name (input sentences)"\
					"2. neural network file name 3.output file name \n";
					exit (EXIT_FAILURE);
				}

				params.target_file_name = test_files[0];
				params.input_weight_file = test_files[1];
				params.output_force_decode = test_files[2];
				params.LM = true;

				input_file_prep input_helper;

				input_helper.integerize_file_LM(params.input_weight_file,params.target_file_name,"tmp/validation.txt",
					params.longest_sent,1,false,params.LSTM_size,params.target_vocab_size,false,params.source_vocab_size);

			}
			else {
				if(test_files.size()!=4) {
					std::cout << "ERROR: force-decode takes four arguements: 1. source input file"\
					" 2. target input file  3. neural network file name 4. output file name\n";
					exit (EXIT_FAILURE);
				}

				params.LM = false;
				params.source_file_name = test_files[0];
				params.target_file_name = test_files[1];
				params.input_weight_file = test_files[2];
				params.output_force_decode = test_files[3];

				input_file_prep input_helper;

				input_helper.integerize_file_nonLM(params.input_weight_file,params.source_file_name,
					params.target_file_name,"tmp/validation.txt",params.longest_sent,1,params.LSTM_size,
					params.source_vocab_size,params.target_vocab_size);
			}
			params.train= false;
			params.decode=false;
			params.test = true;
			params.minibatch_size=1;
			params.stochastic_generation = false;
			return;
		}

		if(vm.count("stoch-gen")) {
			if(!vm.count("sequence")) {
				std::cout << "ERROR: you can only do stoch-gen on the sequence model\n";
				exit (EXIT_FAILURE);
			}

			if(stoicgen_files.size()!=2) {
				std::cout << "ERROR: stoch-gen takes two inputs"\
				" 1. neural network file name 2. output file name\n";
				exit (EXIT_FAILURE);
			}

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


int main(int argc, char **argv) {

	//Timing stuff
	std::chrono::time_point<std::chrono::system_clock> start_total,
	end_total, begin_minibatch,end_minibatch,begin_decoding,end_decoding;
	std::chrono::duration<double> elapsed_seconds;

    start_total = std::chrono::system_clock::now();

    //Initializing the model
	global_params params; //Declare all of the global parameters
	//file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file); //Initialize the file information

	//get the command line arguements
	command_line_parse(params,argc,argv);

	neuralMT_model<precision> model; //This is the model
	params.printIntroMessage();
	BZ_CUDA::lower = params.lower_range;
	BZ_CUDA::upper = params.upper_range;

	if(params.google_learning_rate && params.learning_rate_schedule) {
		std::cout << "ERROR: do not select both the fixed learning rate schedule and the perplexity based scheduler";
		std::cout << "I Guarantee this is not what you intended to do\n";
		exit (EXIT_FAILURE);
	}

	if(!params.decode) {
		model.initModel(params.LSTM_size,params.minibatch_size,params.source_vocab_size,params.target_vocab_size,
			params.longest_sent,params.debug,params.learning_rate,params.clip_gradient,params.norm_clip,
			params.input_weight_file,params.output_weight_file,params.softmax_scaled,params.train_perplexity,params.truncated_softmax,
			params.shortlist_size,params.sampled_size,params.LM);
	}

	std::ofstream HPC_output;
	if(params.HPC_output) {
		HPC_output.open("HPC_OUTPUT.txt");
	}

	////////////////////////////////////Train the model//////////////////////////////////////
	if(params.train) {
		//info for averaging the speed
		int curr_batch_num_SPEED = 0;
		const int thres_batch_num_SPEED = 5;//set this to whatever
		int total_words_batch_SPEED = 0;
		double total_batch_time_SPEED = 0;

		//File info for the training file
		file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size,params.train_total_words,params.truncated_softmax,
			params.shortlist_size,params.sampled_size); //Initialize the file information
		//model.initFileInfo(&file_info);
		params.half_way_count = params.train_total_words/2;
		if(params.google_learning_rate) {
			std::cout << "Words at which to start halving the learning rate: " << params.half_way_count << "\n";
			if(params.HPC_output) {
				HPC_output << "Words at which to start halving the learning rate: " << params.half_way_count << "\n";
				HPC_output.flush();
			}
		}
		int current_epoch = 1;
		std::cout << "Starting model training\n";
		std::cout << "Starting epoch 1\n";
		if(params.HPC_output) {
				HPC_output << "Starting model training\n";
				HPC_output << "Starting epoch 1\n";
				HPC_output.flush();
		}

	
		//stuff for learning rate schedule
		int total_words = 0;
		precision temp_learning_rate = params.learning_rate; //This is only for the google learning rate
		bool learning_rate_flag =true;//used for google learning rate for halving at every 0.5 epochs
		double old_perplexity = 0;
		model.train_perplexity = 0; //set the model perplexity to zero
		while(current_epoch <= params.num_epochs) {
			begin_minibatch = std::chrono::system_clock::now();
			bool success = file_info.read_minibatch();
			end_minibatch = std::chrono::system_clock::now();
			elapsed_seconds = end_minibatch-begin_minibatch;
			//std::cout << "File I/O time: " << elapsed_seconds.count()/60.0 << " minutes\n";
			total_batch_time_SPEED+= elapsed_seconds.count();

			begin_minibatch = std::chrono::system_clock::now();

			//cudaProfilerStart();

			model.compute_gradients(file_info.minibatch_tokens_source_input,file_info.minibatch_tokens_source_output,
				file_info.minibatch_tokens_target_input,file_info.minibatch_tokens_target_output,
				file_info.h_input_vocab_indicies_source,file_info.h_output_vocab_indicies_source,
				file_info.h_input_vocab_indicies_target,file_info.h_output_vocab_indicies_target,
				file_info.current_source_length,file_info.current_target_length,
				file_info.h_input_vocab_indicies_source_Wgrad,file_info.h_input_vocab_indicies_target_Wgrad,
				file_info.len_source_Wgrad,file_info.len_target_Wgrad,file_info.h_sampled_indices,
				file_info.len_unique_words_trunc_softmax);

			// cudaProfilerStop();
			// return 0;

			end_minibatch = std::chrono::system_clock::now();
			elapsed_seconds = end_minibatch-begin_minibatch;

			total_batch_time_SPEED+= elapsed_seconds.count();
			total_words_batch_SPEED+=file_info.words_in_minibatch;

			if(curr_batch_num_SPEED>=thres_batch_num_SPEED) {
				std::cout << "Batched Minibatch time: " << total_batch_time_SPEED/60.0 << " minutes\n";
				std::cout << "Batched Words in minibatch: " << total_words_batch_SPEED << "\n";
				std::cout << "Batched Throughput: " << (total_words_batch_SPEED)/(total_batch_time_SPEED) << " words per second\n";
				std::cout << total_words << " out of " << params.train_total_words << " epoch: " << current_epoch <<  "\n\n";
				if(params.HPC_output) {
					HPC_output << "Batched Minibatch time: " << total_batch_time_SPEED/60.0 << " minutes\n";
					HPC_output << "Batched Words in minibatch: " << total_words_batch_SPEED << "\n";
					HPC_output << "Batched Throughput: " << (total_words_batch_SPEED)/(total_batch_time_SPEED) << " words per second\n";
					HPC_output << total_words << " out of " << params.train_total_words << " epoch: " << current_epoch <<  "\n\n";
					HPC_output.flush();
				}
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
					std::cout << "HALVING LEARNING RATE: " << temp_learning_rate << "\n";
					model.update_learning_rate(temp_learning_rate);
					learning_rate_flag = false;
					if(params.HPC_output) {
						HPC_output << "HALVING LEARNING RATE: " << temp_learning_rate << "\n";
						HPC_output.flush();
					}
			}

			//stuff for perplexity based learning schedule
			if(params.learning_rate_schedule && total_words>=params.half_way_count &&learning_rate_flag) {
				learning_rate_flag = false;
				double new_perplexity = model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
					params.source_vocab_size,params.target_vocab_size,HPC_output,false,params.test_total_words,params.HPC_output,false,"");
				std::cout << "Old dev set Perplexity: " << old_perplexity << "\n";
				std::cout << "New dev set Perplexity: " << new_perplexity << "\n";
				if(params.HPC_output) {
					HPC_output << "Old dev set Perplexity: " << old_perplexity << "\n";
					HPC_output << "New dev set Perplexity: " << new_perplexity << "\n";
					HPC_output.flush();
				}
				if ( (new_perplexity + params.margin >= old_perplexity) && current_epoch!=1) {
					temp_learning_rate = temp_learning_rate*params.decrease_factor;
					model.update_learning_rate(temp_learning_rate);
					std::cout << "Halving learning rate!\n\n";
					if(params.HPC_output) {
						HPC_output << "Halving learning rate!\n\n";
						HPC_output.flush();
					}
				}
				old_perplexity = new_perplexity;
			}

			if(!success) {
				current_epoch+=1;
				//stuff for google learning rate schedule
				if(params.google_learning_rate && current_epoch>=params.epoch_to_start_halving) {
					temp_learning_rate = temp_learning_rate/2;
					std::cout << "HALVING LEARNING RATE: " << temp_learning_rate << "\n";
					model.update_learning_rate(temp_learning_rate);
					learning_rate_flag = true;
					if(params.HPC_output) {
						HPC_output << "HALVING LEARNING RATE: " << temp_learning_rate << "\n";
						HPC_output.flush();
					}
				}

				double new_perplexity;
				if(params.google_learning_rate || params.learning_rate_schedule) {
					new_perplexity = model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
						params.source_vocab_size,params.target_vocab_size,HPC_output,false,params.test_total_words,params.HPC_output,false,"");
				}
				//stuff for perplexity based learning schedule
				if(params.learning_rate_schedule) {
					std::cout << "Old dev set Perplexity: " << old_perplexity << "\n";
					std::cout << "New dev set Perplexity: " << new_perplexity << "\n";
					if(params.HPC_output) {
						HPC_output << "Old dev set Perplexity: " << old_perplexity << "\n";
						HPC_output << "New dev set Perplexity: " << new_perplexity << "\n";
						HPC_output.flush();
					}
					if ( (new_perplexity + params.margin >= old_perplexity) && current_epoch!=1) {
						std::cout << "Halving learning rate!\n\n";
						if(params.HPC_output) {
							HPC_output << "Halving learning rate!\n\n";
							HPC_output.flush();
						}
						temp_learning_rate = temp_learning_rate*params.decrease_factor;
						model.update_learning_rate(temp_learning_rate);
					}
					learning_rate_flag = true;
					old_perplexity = new_perplexity;
				}

				if(params.train_perplexity) {
					std::cout << "PData on train set:"  << model.train_perplexity << "\n";
					std::cout << "Total target words: " << file_info.total_target_words << "\n";
					std::cout << "Training set perplexity: " << std::pow(2,-1*model.train_perplexity/file_info.total_target_words) << "\n";
					if(params.HPC_output) {
						HPC_output << "Training set perplexity: " << std::pow(2,-1*model.train_perplexity/file_info.total_target_words) << "\n";
						HPC_output.flush();
					}
					model.train_perplexity = 0;
				}

				total_words=0;
				if(current_epoch <= params.num_epochs) {
					std::cout << "-----------------------------------"  << std::endl;
					std::cout << "Starting epoch " << current_epoch << std::endl;
					std::cout << "-----------------------------------"  << std::endl;
					if(params.HPC_output) {
						HPC_output << "Starting epoch " << current_epoch << std::endl;
						HPC_output.flush();
					}
				}
			}
			cudaDeviceSynchronize();
		}	
		//Now that training is done, dump the weights
		cudaDeviceSynchronize();
		model.dump_weights();
	}


	/////////////////////////////////Get perplexity on test set////////////////////////////////
	if(params.test) {
		model.get_perplexity(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size,HPC_output,true,params.test_total_words,params.HPC_output,true,params.output_force_decode);
	}

	if(params.LM && params.stochastic_generation) {
		model.stoicastic_generation(params.sg_length,params.sg_output_file_temp);
		input_file_prep input_helper;
		input_helper.unint_file(params.input_weight_file,"tmp/sg.txt",params.sg_output_file,true,false);
	}


	///////////////////////////////////////////decode the model////////////////////////////////////////////
	if(params.decode) {
		std::cout << "-----------------Starting Decoding----------------\n";
		begin_decoding = std::chrono::system_clock::now();
		model.beam_decoder(params.beam_size,params.decode_file_name,
			params.input_weight_file,params.decode_num_lines_in_file,params.source_vocab_size,
			params.target_vocab_size,params.longest_sent,params.LSTM_size,params.penalty,
			params.decoder_output_file,params.min_decoding_ratio,params.max_decoding_ratio,params.softmax_scaled,
			params.num_hypotheses,params.print_score);
		end_decoding = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_decoding-begin_decoding;
		std::cout << "Decoding time: " << elapsed_seconds.count()/60.0 << " minutes\n";

		//now unintegerize the file
		input_file_prep input_helper;
		input_helper.unint_file(params.input_weight_file,"tmp/decoder_output.txt",params.decoder_final_file,false,true);
	}

	//Compute the final runtime
	end_total = std::chrono::system_clock::now();
	elapsed_seconds = end_total-start_total;
    std::cout << "\n\n\n";
    std::cout << "Total Program Runtime: " << elapsed_seconds.count()/60.0 << " minutes" << std::endl;
}
