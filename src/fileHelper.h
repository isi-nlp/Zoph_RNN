//Load in the training examples from the file

#ifndef FILE_INPUT
#define FILE_INPUT

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unordered_map>

//templated for float or doubles
struct file_helper {
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
	int current_target_length;
	int source_vocab_size;
	int target_vocab_size;

	int *h_input_vocab_indicies_source;
	int *h_output_vocab_indicies_source;

	int *h_input_vocab_indicies_target;
	int *h_output_vocab_indicies_target;

	int *h_input_vocab_indicies_source_temp;
	int *h_output_vocab_indicies_source_temp;
	int *h_input_vocab_indicies_target_temp;
	int *h_output_vocab_indicies_target_temp;

	//These are the special vocab indicies for the W gradient updates
	int *h_input_vocab_indicies_source_Wgrad;
	int *h_input_vocab_indicies_target_Wgrad;

	bool *bitmap_source; //This is for preprocessing the input vocab for quick updates on the W gradient
	bool *bitmap_target; //This is for preprocessing the input vocab for quick updates on the W gradient

	//length for the special W gradient stuff
	int len_source_Wgrad;
	int len_target_Wgrad;

	//for perplexity
	int total_target_words;

	bool truncated_softmax;
	int shortlist_size;
	int sampled_size;
	int len_unique_words_trunc_softmax; //use the sample rate for words above this index
	int *h_sampled_indices; //size of sampled size, for truncated softmax
	std::unordered_map<int,int> resevoir_mapping; //stores mapping for word in vocab to row number in output distribution/weight matrices

	~file_helper() {
		delete [] bitmap_source;
		delete [] bitmap_target;

		free(h_input_vocab_indicies_source);
		free(h_output_vocab_indicies_source);

		free(h_input_vocab_indicies_target);
		free(h_output_vocab_indicies_target);

		free(h_input_vocab_indicies_source_temp);
		free(h_output_vocab_indicies_source_temp);
		free(h_input_vocab_indicies_target_temp);
		free(h_output_vocab_indicies_target_temp);

		free(h_input_vocab_indicies_source_Wgrad);
		free(h_input_vocab_indicies_target_Wgrad);

		input_file.close();
	}


	//can change to memset for speed if needed
	void zero_bitmaps() {

		for(int i=0; i<source_vocab_size; i++) {
			bitmap_source[i] = false;
		}

		for(int i=0; i<target_vocab_size; i++) {
			bitmap_target[i] = false;
		}
	}

	void preprocess_output_truncated_softmax() {
		zero_bitmaps();
		resevoir_mapping.clear();

		std::cout << "Shortlist size: " << shortlist_size << "\n";


		int curr_index = 0;
		for(int i=0; i < minibatch_size*current_target_length; i++) {

			if(bitmap_target[h_output_vocab_indicies_target[i]]==false && h_output_vocab_indicies_target[i] >= shortlist_size) {
				bitmap_target[h_output_vocab_indicies_target[i]]=true;
				h_sampled_indices[curr_index] = h_output_vocab_indicies_target[i];
				curr_index+=1;
			}
		}
		len_unique_words_trunc_softmax =curr_index;
		std::cout << "len_unique_words_trunc_softmax: " << len_unique_words_trunc_softmax << "\n";
		std::cout << "len of W grad target: " << len_target_Wgrad << "\n";

		if(curr_index > sampled_size) {
			std::cout << "ERROR: the sample size of the truncated softmax is too small\n";
			std::cout << "More unique words in the minibatch that there are sampled slots\n";
			exit (EXIT_FAILURE);
		}

		curr_index = 0;
		int num_to_sample = sampled_size - len_unique_words_trunc_softmax;
		boost::uniform_real<> distribution(0,1);
		for(int i=shortlist_size; i<target_vocab_size; i++) {
			if(bitmap_target[i]==false) {
				//fill the resevoir
				if(curr_index < num_to_sample) {
					h_sampled_indices[len_unique_words_trunc_softmax+curr_index] = i;
					curr_index++;
				}
				else {
					int rand_num = (int)(curr_index*distribution(BZ_CUDA::gen));
					
					if (rand_num <num_to_sample) {
						h_sampled_indices[len_unique_words_trunc_softmax+rand_num] = i;
					}
					curr_index++;
				}
			}
			if(len_unique_words_trunc_softmax+curr_index >= sampled_size) {
				break;
			}
		}

		//get the mappings
		std::cout << "The samples:\n";
		for(int i=0; i<sampled_size; i++) {
			std::cout << h_sampled_indices[i] << " ";
			resevoir_mapping[h_sampled_indices[i]] = i;
		}
		std::cout << "\n";

		for(int i=0; i< minibatch_size*current_target_length; i++) {

			if(h_output_vocab_indicies_target[i]>=shortlist_size && h_output_vocab_indicies_target[i]!=-1) {
				h_output_vocab_indicies_target[i] = resevoir_mapping.at(h_output_vocab_indicies_target[i]);
			}
		}

	}

	//This returns the length of the special sequence for the W grad
	void preprocess_input_Wgrad() {

		//zero out bitmaps at beginning
		zero_bitmaps();
		// std::cout << "-------------CURRENT SOURCE W DEBUG---------------\n";
		// std::cout << current_source_length << "\n";
		// std::cout << "-------------CURRENT TARGET W DEBUG---------------\n";
		// std::cout << current_target_length << "\n";

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

		for(int i=0; i < minibatch_size*current_target_length; i++) {

			if(h_input_vocab_indicies_target[i]==-1) {
				h_input_vocab_indicies_target_Wgrad[i] = -1;
			}
			else if(bitmap_target[h_input_vocab_indicies_target[i]]==false) {
				bitmap_target[h_input_vocab_indicies_target[i]]=true;
				h_input_vocab_indicies_target_Wgrad[i] = h_input_vocab_indicies_target[i];
			}
			else  {
				h_input_vocab_indicies_target_Wgrad[i] = -1;
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

		//target
		//Now go and put all -1's at far right and number in far left
		len_target_Wgrad = -1;
		left_index = 0;
		right_index = minibatch_size*current_target_length-1;
		while(left_index < right_index) {
			if(h_input_vocab_indicies_target_Wgrad[left_index]==-1) {
				if(h_input_vocab_indicies_target_Wgrad[right_index]!=-1) {
					int temp_swap = h_input_vocab_indicies_target_Wgrad[left_index];
					h_input_vocab_indicies_target_Wgrad[left_index] = h_input_vocab_indicies_target_Wgrad[right_index];
					h_input_vocab_indicies_target_Wgrad[right_index] = temp_swap;
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
		if(h_input_vocab_indicies_target_Wgrad[left_index]!=-1) {
			left_index++;
		}
		len_target_Wgrad = left_index;
		//std::cout << "YO1 " << len_target_Wgrad << " " << len_source_Wgrad << "\n";

		//debug the source and target W_grad special indices
		// std::cout << "------------------DEBUG FOR SOURCE then TARGET W_GRAD INDICIES-----------------------\n";
		// for(int i=0; i< len_source_Wgrad + 2; i++) {
		// 	std::cout << h_input_vocab_indicies_source_Wgrad[i] << " ";
		// }
		// std::cout << "\n\n";

		// for(int i=0; i< len_target_Wgrad + 2; i++) {
		// 	std::cout << h_input_vocab_indicies_target_Wgrad[i] << " ";
		// }
		// std::cout << "\n\n";

	}

	//Constructor
	file_helper(std::string fn,int ms,int &nlif,int max_sent_len,int source_vocab_size,int target_vocab_size,int &total_words,bool truncated_softmax,int shortlist_size,
		int sampled_size)
	{
		file_name = fn;
		minibatch_size = ms;
		input_file.open(file_name.c_str(),std::ifstream::in); //Open the stream to the file
		this->source_vocab_size = source_vocab_size;
		this->target_vocab_size = target_vocab_size;

		get_file_stats(nlif,total_words,input_file,total_target_words);
		nums_lines_in_file = nlif;
		//std::cout << "NUMBER OF LINES IN FILE: " << nlif << "\n";
		//std::cout << "NUMBER OF WORDS IN FILE: " << total_words << "\n";

		//GPU allocation
		this->max_sent_len = max_sent_len;
		h_input_vocab_indicies_source = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_source = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_target = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_target = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_source_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_source_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_input_vocab_indicies_target_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_output_vocab_indicies_target_temp = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		h_input_vocab_indicies_source_Wgrad = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));
		h_input_vocab_indicies_target_Wgrad = (int *)malloc(minibatch_size * max_sent_len * sizeof(int));

		if(source_vocab_size!=-1) {
			bitmap_source = new bool[source_vocab_size*sizeof(bool)];
		}
		else {
			bitmap_source = new bool[2*sizeof(bool)];
		}
		bitmap_target = new bool[target_vocab_size*sizeof(bool)];

		this->truncated_softmax = truncated_softmax;
		this->sampled_size = sampled_size;
		this->shortlist_size = shortlist_size;
		h_sampled_indices = (int *)malloc(sampled_size * sizeof(int));

	}

	//Read in the next minibatch from the file
	//returns bool, true is same epoch, false if now need to start new epoch
	bool read_minibatch() {

		#ifdef CPU_DEBUG
		std::vector<std::vector<int>*> temp_minibatch_tokens_source_input; //Each vector is the tokens for each sentence
		std::vector<std::vector<int>*> temp_minibatch_tokens_source_output; //Each vector is the tokens for each sentence
		std::vector<std::vector<int>*> temp_minibatch_tokens_target_input; //Each vector is the tokens for each sentence
		std::vector<std::vector<int>*> temp_minibatch_tokens_target_output; //Each vector is the tokens for each sentence
		#endif
		int max_sent_len_source = 0;
		int max_sent_len_target = 0;
		bool sameEpoch = true;
		words_in_minibatch=0; //For throughput calculation

		//For gpu file input
		int current_temp_source_input_index = 0;
		int current_temp_source_output_index = 0;
		int current_temp_target_input_index = 0;
		int current_temp_target_output_index = 0;

		//std::cout << "Begin minibatch(Now printing input that was in the file)\n";
		//Now load in the minibatch
		for(int i=0; i<minibatch_size; i++) {
			if(current_line_in_file > nums_lines_in_file) {
				input_file.clear();
				input_file.seekg(0, std::ios::beg);
				current_line_in_file = 1;
				sameEpoch = false;
				break;
			}

			std::string temp_input_source;
			std::string temp_output_source;
			std::getline(input_file, temp_input_source);
			std::getline(input_file, temp_output_source);
			#ifdef CPU_DEBUG
			std::vector<int>* temp_input_sentence_source = new std::vector<int>;
			std::vector<int>* temp_output_sentence_source = new std::vector<int>;
			#endif

			std::string temp_input_target;
			std::string temp_output_target;
			std::getline(input_file, temp_input_target);
			std::getline(input_file, temp_output_target);
			#ifdef CPU_DEBUG
			std::vector<int>* temp_input_sentence_target = new std::vector<int>;
			std::vector<int>* temp_output_sentence_target = new std::vector<int>;
			#endif

			///////////////////////////////////Process the source////////////////////////////////////
			std::istringstream iss_input_source(temp_input_source, std::istringstream::in);
			std::istringstream iss_output_source(temp_output_source, std::istringstream::in);
			std::string word; //The temp word

			int input_source_length = 0;
			while( iss_input_source >> word ) {
				//std::cout << word << " ";
				#ifdef CPU_DEBUG
				temp_input_sentence_source->push_back(std::stoi(word));
				#endif
				h_input_vocab_indicies_source_temp[current_temp_source_input_index] = std::stoi(word);
				input_source_length+=1;
				current_temp_source_input_index+=1;
			}
			//std::cout << "\n";
			int output_source_length = 0;
			while( iss_output_source >> word ) {
				//std::cout << word << " ";
				#ifdef CPU_DEBUG
				temp_output_sentence_source->push_back(std::stoi(word));
				#endif
				h_output_vocab_indicies_source_temp[current_temp_source_output_index] = std::stoi(word);
				output_source_length+=1;
				current_temp_source_output_index+=1;
			}
			//std::cout << "\n";
			//CHANGED
			//words_in_minibatch+=temp_input_sentence_source->size();
			//max_sent_len_source = temp_input_sentence_source->size();
			words_in_minibatch+=input_source_length;
			max_sent_len_source = input_source_length;


			#ifdef CPU_DEBUG
			temp_minibatch_tokens_source_input.push_back(temp_input_sentence_source);
			temp_minibatch_tokens_source_output.push_back(temp_output_sentence_source);
			#endif

			///////////////////////////////////Process the target////////////////////////////////////
			std::istringstream iss_input_target(temp_input_target, std::istringstream::in);
			std::istringstream iss_output_target(temp_output_target, std::istringstream::in);

			int input_target_length = 0;
			while( iss_input_target >> word ) {
				//std::cout << word << " ";
				#ifdef CPU_DEBUG
				temp_input_sentence_target->push_back(std::stoi(word));
				#endif
				h_input_vocab_indicies_target_temp[current_temp_target_input_index] = std::stoi(word);
				current_temp_target_input_index+=1;
				input_target_length+=1;
			}
			//std::cout << "\n";
			int output_target_length = 0;
			while( iss_output_target >> word ) {
				//std::cout << word << " ";
				#ifdef CPU_DEBUG
				temp_output_sentence_target->push_back(std::stoi(word));
				#endif
				h_output_vocab_indicies_target_temp[current_temp_target_output_index] = std::stoi(word);
				current_temp_target_output_index+=1;
				output_target_length+=1;
			}
			//std::cout << "\n";

			current_source_length = input_source_length;
			current_target_length = input_target_length;
			//std::cout << "Current input source length: " << input_source_length << "\n";
			//std::cout << "Current input target length: " << input_target_length << "\n";

			//CHANGED
			//words_in_minibatch += temp_input_sentence_target->size();
			//max_sent_len_target = temp_input_sentence_target->size();
			words_in_minibatch += input_target_length; 
			max_sent_len_target = input_target_length;

			#ifdef CPU_DEBUG
			temp_minibatch_tokens_target_input.push_back(temp_input_sentence_target);
			temp_minibatch_tokens_target_output.push_back(temp_output_sentence_target);
			#endif

			//Now increase current line in file because we have seen two more sentences
			current_line_in_file+=4;
		}
		if(current_line_in_file>nums_lines_in_file) {
			current_line_in_file = 1;
			input_file.clear();
			input_file.seekg(0, std::ios::beg);
			sameEpoch = false;
		}

		//reset for GPU
		words_in_minibatch = 0;

		//Now fill in the minibatch_tokens_input and minibatch_tokens_output
		#ifdef CPU_DEBUG
		minibatch_tokens_source_input.resize(minibatch_size,max_sent_len_source);
		minibatch_tokens_source_output.resize(minibatch_size,max_sent_len_source);
		minibatch_tokens_target_input.resize(minibatch_size,max_sent_len_target);
		minibatch_tokens_target_output.resize(minibatch_size,max_sent_len_target);
		#endif

		#ifdef CPU_DEBUG
		for(int i=0; i<temp_minibatch_tokens_source_input.size(); i++) {
			for(int j=0; j< temp_minibatch_tokens_source_input[i]->size(); j++) {
				minibatch_tokens_source_input(i,j) = temp_minibatch_tokens_source_input[i]->at(j);
				minibatch_tokens_source_output(i,j) = temp_minibatch_tokens_source_output[i]->at(j);
			}
			//Now deallocate the vectors
			delete temp_minibatch_tokens_source_input[i];
			delete temp_minibatch_tokens_source_output[i];
		}
		#endif

		//get vocab indicies in correct memory layout on the host
		//std::cout << "-------------------source input check--------------------\n";
		for(int i=0; i<minibatch_size; i++) {
			for(int j=0; j<current_source_length; j++) {
				h_input_vocab_indicies_source[i + j*minibatch_size] = h_input_vocab_indicies_source_temp[j + current_source_length*i];
				h_output_vocab_indicies_source[i + j*minibatch_size] = h_output_vocab_indicies_source_temp[j + current_source_length*i];
				if(h_input_vocab_indicies_source[i + j*minibatch_size]!=-1) {
					words_in_minibatch+=1;
				}
				//std::cout << h_input_vocab_indicies_source[i + j*minibatch_size] << "   " << minibatch_tokens_source_input(i,j) << "\n";
				//std::cout << h_output_vocab_indicies_source[i + j*minibatch_size] << "   " << minibatch_tokens_source_output(i,j) << "\n";
			}
		}
		//std::cout << "\n\n";

		#ifdef CPU_DEBUG
		for(int i=0; i<temp_minibatch_tokens_target_input.size(); i++) {
			for(int j=0; j< temp_minibatch_tokens_target_input[i]->size(); j++) {
				minibatch_tokens_target_input(i,j) = temp_minibatch_tokens_target_input[i]->at(j);
				// if(minibatch_tokens_target_input(i,j)!=-1) {
				// 	words_in_minibatch+=1;
				// }
				minibatch_tokens_target_output(i,j) = temp_minibatch_tokens_target_output[i]->at(j);
			}
			//Now deallocate the vectors
			delete temp_minibatch_tokens_target_input[i];
			delete temp_minibatch_tokens_target_output[i];
		}
		#endif

		//std::cout << "-------------------target input check--------------------\n";
		for(int i=0; i<minibatch_size; i++) {
			for(int j=0; j<current_target_length; j++) {
				h_input_vocab_indicies_target[i + j*minibatch_size] = h_input_vocab_indicies_target_temp[j + current_target_length*i];
				h_output_vocab_indicies_target[i + j*minibatch_size] = h_output_vocab_indicies_target_temp[j + current_target_length*i];
				if(h_output_vocab_indicies_target[i + j*minibatch_size]!=-1) {
					words_in_minibatch+=1;
				}
				//std::cout << h_input_vocab_indicies_target[i + j*minibatch_size] << "   " << minibatch_tokens_target_input(i,j) << "\n";
				//std::cout << h_output_vocab_indicies_target[i + j*minibatch_size] << "   " << minibatch_tokens_target_output(i,j) << "\n";
			}
		}

		//std::cout << "\n\n";

		//Now preprocess the data on the host before sending it to the gpu
		preprocess_input_Wgrad();

		if(truncated_softmax) {
			preprocess_output_truncated_softmax();
		}
		return sameEpoch;
	}
};

#endif