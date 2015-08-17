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

//Boost stuff
//#include <boost/algorithm/string.hpp>

//My own includes
#include "global_params.h"
//#include "BZ_CUDA_UTIL.h"
#include "model.h"
#include "fileHelper.h"
#include "Eigen_Util.h"
#include "model.hpp"
#include "LSTM.hpp"
#include "softmax.hpp"
#include "Input_To_Hidden_Layer.hpp"
#include "Hidden_To_Hidden_Layer.hpp"
#include "LSTM_HH.hpp"


int main(int argc, char **argv) {

	//Timing stuff
	std::chrono::time_point<std::chrono::system_clock> start_total,
	end_total, begin_minibatch,end_minibatch,begin_decoding,end_decoding;

    start_total = std::chrono::system_clock::now();

    std::ofstream perp("perplexity.txt");

    //Initializing the model
	global_params params; //Declare all of the global parameters
	//file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file); //Initialize the file information
	neuralMT_model<precision> model; //This is the model
	params.printIntroMessage();
	model.initModel(params.LSTM_size,params.minibatch_size,params.source_vocab_size,params.target_vocab_size,
		params.longest_sent,params.debug,params.learning_rate,params.clip_gradient,params.norm_clip,
		params.input_weight_file,params.output_weight_file);

	////////////////////////////////////Train the model//////////////////////////////////////
	if(params.train) {
		//File info for the training file
		file_helper file_info(params.train_file_name,params.minibatch_size,params.train_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size); //Initialize the file information
		model.initFileInfo(&file_info);
		int current_epoch = 1;
		std::cout << "Starting model training\n";
		std::cout << "Starting epoch 1\n";
		int total_words = 0;
		while(current_epoch <= params.num_epochs) {
			bool success = file_info.read_minibatch();
			//std::cout << "log[P(data)]: " << model.getError(file_info.minibatch_tokens_input,file_info.minibatch_tokens_output) << "\n";
			begin_minibatch = std::chrono::system_clock::now();

			model.compute_gradients(file_info.minibatch_tokens_source_input,file_info.minibatch_tokens_source_output,
				file_info.minibatch_tokens_target_input,file_info.minibatch_tokens_target_output,
				file_info.h_input_vocab_indicies_source,file_info.h_output_vocab_indicies_source,
				file_info.h_input_vocab_indicies_target,file_info.h_output_vocab_indicies_target,
				file_info.current_source_length,file_info.current_target_length,
				file_info.h_input_vocab_indicies_target_Wgrad,file_info.h_input_vocab_indicies_target_Wgrad);

			end_minibatch = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end_minibatch-begin_minibatch;
			std::cout << "Minibatch time: " << elapsed_seconds.count()/60.0 << " minutes\n";
			std::cout << "Words in minibatch: " << file_info.words_in_minibatch << "\n";
			std::cout << "Throughput: " << (file_info.words_in_minibatch)/(elapsed_seconds.count()) << " words per second\n";
			total_words += file_info.words_in_minibatch;
			std::cout << total_words << " out of " << params.train_total_words << " epoch: " << current_epoch <<  "\n\n";
			if(!success) {
				current_epoch+=1;
				total_words=0;
				if(current_epoch <= params.num_epochs) {
					std::cout << "Starting epoch " << current_epoch << std::endl; 
				}
			}
		}

		//Now that training is done, dump the weights
		model.dump_weights();
	}


	/////////////////////////////////Get perplexity on test set////////////////////////////////
	if(params.test) {
		model.load_weights();
		//File input for the testing file
		file_helper file_info(params.test_file_name,params.minibatch_size,params.test_num_lines_in_file,params.longest_sent,
			params.source_vocab_size,params.target_vocab_size); //Initialize the file information
		model.initFileInfo(&file_info);

		int current_epoch = 1;
		std::cout << "Starting training with model" << std::endl;
		int total_words = 0; //For perplexity
		precision P_data = 0;
		while(current_epoch <= 1) {
			bool success = file_info.read_minibatch();
			P_data += model.getError();
			total_words += file_info.words_in_minibatch;
			std::cout << total_words << " out of " << params.test_total_words << " testing data: " << "\n\n";

			if(!success) {
				current_epoch+=1;
			}
		}
		P_data = P_data/std::log(2.0); //Change to base 2 log
		precision perplexity = std::pow(2,-1*P_data/total_words);
		std::cout << "Perplexity: " << perplexity << std::endl;
		perp << perplexity;
	}


	///////////////////////////////////////////decode the model////////////////////////////////////////////
	if(params.decode) {
		begin_decoding = std::chrono::system_clock::now();
		model.beam_decoder(params.beam_size,params.decode_file_name,
			params.input_weight_file,params.decode_num_lines_in_file,params.source_vocab_size,
			params.target_vocab_size,params.longest_sent,params.LSTM_size,params.penalty,
			params.decoder_output_file,params.min_decoding_ratio,params.max_decoding_ratio);
		end_decoding = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end_decoding-begin_decoding;
		std::cout << "Decoding time: " << elapsed_seconds.count()/60.0 << " minutes\n";
	}

	//Compute the final runtime
	end_total = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end_total-start_total;
    std::cout << "\n\n\n";
    std::cout << "Runtime: " << elapsed_seconds.count()/60.0 << " minutes" << std::endl;
}
