//file helper for other language

//Load in the training examples from the file

#ifndef FILE_INPUT_SOURCE
#define FILE_INPUT_SOURCE

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unordered_map>

//templated for float or doubles
struct file_helper_source {
	std::string file_name; //Name of input file
	int minibatch_size; //Size of minibatches
	std::ifstream input_file; //Input file stream
	int current_line_in_file = 1;
	int nums_lines_in_file;
	
	//Used for computing the maximum sentence length of previous minibatch
	int words_in_minibatch;

	//num rows is the length of minibatch, num columns is len of longest sentence
	//unused positions are padded with -1, since that is not a valid token
	Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_source_input;
	Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_source_output;
	Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_target_input;
	Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> minibatch_tokens_target_output; 

	//-----------------------------------------GPU Parameters---------------------------------------------
	//This is for storing the vocab indicies on the GPU
	int max_sent_len; //max sentence length
	int current_source_length;
	int source_vocab_size;

	int *h_input_vocab_indicies_source;
	int *h_output_vocab_indicies_source;

	int *h_input_vocab_indicies_source_temp;
	int *h_output_vocab_indicies_source_temp;

	//These are the special vocab indicies for the W gradient updates
	int *h_input_vocab_indicies_source_Wgrad;

	bool *bitmap_source; //This is for preprocessing the input vocab for quick updates on the W gradient

	//length for the special W gradient stuff
	int len_source_Wgrad;

	//for the attention model
	int *h_batch_info;

	bool free_flag = false;

	~file_helper_source() {

		if(free_flag) {
			delete [] bitmap_source;
			free(h_input_vocab_indicies_source);
			free(h_output_vocab_indicies_source);
			free(h_input_vocab_indicies_source_temp);
			free(h_output_vocab_indicies_source_temp);
			free(h_input_vocab_indicies_source_Wgrad);
			free(h_batch_info);
			input_file.close();
		}
	}


	//can change to memset for speed if needed
	void zero_bitmaps() {

		for(int i=0; i<source_vocab_size; i++) {
			bitmap_source[i] = false;
		}
	}

	//This returns the length of the special sequence for the W grad
	void preprocess_input_Wgrad() {

		//zero out bitmaps at beginning
		zero_bitmaps();

		//For source
		for(int i=0; i<minibatch_size*current_source_length; i++) {

			if(h_input_vocab_indicies_source[i]==-1) {
				h_input_vocab_indicies_source_Wgrad[i] = -1;
			}
			else if(bitmap_source[h_input_vocab_indicies_source[i]]==false) {
				bitmap_source[h_input_vocab_indicies_source[i]]=true;
				h_input_vocab_indicies_source_Wgrad[i] = h_input_vocab_indicies_source[i];
			}
			else  {
				h_input_vocab_indicies_source_Wgrad[i] = -1;
			}
		}
		

		//source
		//Now go and put all -1's at far right and number in far left
		len_source_Wgrad = -1;
		int left_index = 0;
		int right_index = minibatch_size*current_source_length-1;
		while(left_index < right_index) {
			if(h_input_vocab_indicies_source_Wgrad[left_index]==-1) {
				if(h_input_vocab_indicies_source_Wgrad[right_index]!=-1) {
					int temp_swap = h_input_vocab_indicies_source_Wgrad[left_index];
					h_input_vocab_indicies_source_Wgrad[left_index] = h_input_vocab_indicies_source_Wgrad[right_index];
					h_input_vocab_indicies_source_Wgrad[right_index] = temp_swap;
					left_index++;
					right_index--;
					continue;
				}
				else {
					right_index--;
					continue;
				}
			}
			left_index++;
		}
		if(h_input_vocab_indicies_source_Wgrad[left_index]!=-1) {
			left_index++;
		}
		len_source_Wgrad = left_index;
	}

	//Constructor
	void init_file_helper_source(std::string fn,int ms,int max_sent_len,int source_vocab_size)
	{
		file_name = fn;
		minibatch_size = ms;
		input_file.open(file_name.c_str(),std::ifstream::in); //Open the stream to the file
		this->source_vocab_size = source_vocab_size;

		get_file_stats_source(nums_lines_in_file,input_file);

		free_flag = true;
		//GPU allocation
		this->max_sent_len = max_sent_len;
		h_input_vocab_indicies_source = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_source = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_source_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_source_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_source_Wgrad = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		bitmap_source = new bool[source_vocab_size*sizeof(bool)];
		h_batch_info = (int *)malloc(2*minibatch_size * sizeof(int));
	}

	//Read in the next minibatch from the file
	//returns bool, true is same epoch, false if now need to start new epoch
	void read_minibatch() {

		//std::cout << "IN READ MINIBATCH\n";

		//int max_sent_len_source = 0;
		words_in_minibatch=0; //For throughput calculation

		//For gpu file input
		int current_temp_source_input_index = 0;
		int current_temp_source_output_index = 0;

		//std::cout << "Begin minibatch(Now printing input that was in the file)\n";
		//Now load in the minibatch
		for(int i=0; i<minibatch_size; i++) {
			if(current_line_in_file > nums_lines_in_file) {
				input_file.clear();
				input_file.seekg(0, std::ios::beg);
				current_line_in_file = 1;
			}

			std::string temp_input_source;
			std::string temp_output_source;
			std::getline(input_file, temp_input_source);
			std::getline(input_file, temp_output_source);

			///////////////////////////////////Process the source////////////////////////////////////
			std::istringstream iss_input_source(temp_input_source, std::istringstream::in);
			std::istringstream iss_output_source(temp_output_source, std::istringstream::in);
			std::string word; //The temp word

			int input_source_length = 0;
			while( iss_input_source >> word ) {
				//std::cout << word << " ";
				h_input_vocab_indicies_source_temp[current_temp_source_input_index] = std::stoi(word);
				input_source_length+=1;
				current_temp_source_input_index+=1;
			}
			//std::cout << "\n";
			int output_source_length = 0;
			while( iss_output_source >> word ) {
				//std::cout << word << " ";
				h_output_vocab_indicies_source_temp[current_temp_source_output_index] = std::stoi(word);
				output_source_length+=1;
				current_temp_source_output_index+=1;
			}

			words_in_minibatch += input_source_length;
			//max_sent_len_source = input_source_length;

			///////////////////////////////////Process the target////////////////////////////////////

			current_source_length = input_source_length;

			//Now increase current line in file because we have seen two more sentences
			current_line_in_file+=2;
		}
		if(current_line_in_file>nums_lines_in_file) {
			current_line_in_file = 1;
			input_file.clear();
			input_file.seekg(0, std::ios::beg);
		}

		//reset for GPU
		words_in_minibatch = 0;

		//get vocab indicies in correct memory layout on the host
		//std::cout << "-------------------source input check--------------------\n";
		for(int i=0; i<minibatch_size; i++) {
			int STATS_source_len = 0;
			for(int j=0; j<current_source_length; j++) {

				//stuff for getting the individual source lengths in the minibatch
				if(h_input_vocab_indicies_source_temp[j + current_source_length*i]!=-1) {
					STATS_source_len+=1;
				}
				h_input_vocab_indicies_source[i + j*minibatch_size] = h_input_vocab_indicies_source_temp[j + current_source_length*i];
				h_output_vocab_indicies_source[i + j*minibatch_size] = h_output_vocab_indicies_source_temp[j + current_source_length*i];
				if(h_input_vocab_indicies_source[i + j*minibatch_size]!=-1) {
					words_in_minibatch+=1;
				}
			}
			h_batch_info[i] = STATS_source_len;
			h_batch_info[i+minibatch_size] = current_source_length - STATS_source_len;
		}

		//Now preprocess the data on the host before sending it to the gpu
		preprocess_input_Wgrad();
	} 
};

#endif
