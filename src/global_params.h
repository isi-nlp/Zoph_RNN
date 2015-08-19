//Global parameter file that needs to be specified for learning

typedef float precision;
//#define CPU_DEBUG //This is if you want to have the eigen code running
//#define CPU_MODE
#define GPU_MODE

struct global_params {	

	//General settings
	static const bool debug = false;
	bool train = true; //If you want to train the model
	bool test = false; //If you want to test the model
	bool decode = false;
	bool HPC_output = false; //flushes the output to a file, so it can be read as the program executes
	bool train_perplexity = true; //print out the train perplexity every epoch (or half epoch if you have a learning rate schedule)
	bool LM = false; //if true it is only a sequence model, not sequence to sequence, sequence to sequence is the default
	bool shuffle = true; //shuffle the training data

	bool stochastic_generation = false; //This is for Language modeling only
	int sg_length=1000; //how many tokens to generate
	std::string sg_output_file = "sg.txt";
	std::string sg_output_file_temp = "tmp/sg.txt";


	//Model training info
	int minibatch_size = 128; //Size of the minibatch
	int num_epochs = 10; //Number passes through the dataset
	precision learning_rate = 0.7; //The learning rate for SGD

	//stuff for the google learning rate
	//This halves the learning rate every 0.5 epochs after some inital epoch
	bool google_learning_rate = false;
	int epoch_to_start_halving = 6; //After what epoch do you halve the learnign rate
	int half_way_count = -1; //What is the total number of words that mark half an epoch

	//stuff for normal halving of the learning rate where every half epoch the validation set is looked at 
	//and if it didn't improve, or did worse, the learning rate is halved.
	//NOTE do not have on google learning rate and the normal learning rate schedule
	bool learning_rate_schedule = false;
	precision decrease_factor = 0.5;
	double margin = 0.0; 
	std::string dev_source_file_name;
	std::string dev_target_file_name;


	//parameter initialization
	//Currently uniform initialization
	precision lower_range = -0.08;
	precision upper_range = 0.08;

	//note this is only for GPU, not for CPU testing
	//always use this as thrust cannot use streams until I download new version, fix this for performance when using
	const bool softmax_scaled = true; //This is for if you want to scale all values in the outputdist to avoid overflow when exping floats


	//the truncated softmax
	//top_fixed + sampled = target vocabulary
	bool truncated_softmax =false;
	int shortlist_size = 10000;
	int sampled_size = 5000;

	//gradient update parameters
	bool clip_gradient = true;
	precision norm_clip = 5.0; //Renormalize the gradients so they fit in normball this size


	//Model size info
	//vocab size of -1 defaults to the size of the train file specified
	int source_vocab_size = -1;
	int target_vocab_size = -1; //Size in input vocabulary, ranging from 0-input_vocab_size, where 0 is start symbol
	int LSTM_size = 1000; //LSTM cell size, by definition it is the same as the word embedding layer
	const int num_hidden_layers = 1; //This is the number of stacked LSTM's in the model


	////////////////////Decoder settings//////////////////
	int beam_size = 12;
	precision penalty = 0;
	int num_hypotheses = 1;//This prints out the k best paths from the beam decoder for the input
	const precision min_decoding_ratio = 0.5; //target translation divided by source sentence length must be greater than min_decoding_ratio
	const precision max_decoding_ratio = 1.5;
	bool print_score = false; //Whether to print the score of the hypotheses or not
	std::string decode_tmp_file; //used for tmp stuff
	std::string decode_file_name = "tmp/decoder_input.txt";
	std::string decoder_output_file = "tmp/decoder_output.txt";
	std::string decoder_final_file;
	int decode_num_lines_in_file = -1;//This is learned

	/////////////////////////////I/O file info/////////////////////////////
	std::string source_file_name; //for training, kbest, force decode and sg
	std::string target_file_name; //for training, kbest, force decode and sg
	std::string output_force_decode;

	int longest_sent = 100; //Note this doubles when doing translation

	std::string train_file_name = "tmp/train.txt";//Input file where source is first line then target is second line
	int train_num_lines_in_file = -1; //This is learned
	int train_total_words = -1; //This is learned

	std::string test_file_name = "tmp/validation.txt";//Input file where source is first line then target is second line
	int test_num_lines_in_file = -1; //This is learned
	int test_total_words = -1; //This is learned

	//For weights
	std::string input_weight_file = "model.nn";
	std::string output_weight_file = "model.nn";

	void printIntroMessage() {

		if(train) {
			std::cout << "\n\n------------------------Train Info------------------------\n";
			std::cout << "Minibatch Size: " << minibatch_size << std::endl;
			std::cout << "Number of Epochs: " << num_epochs << std::endl;
			std::cout << "Learning Rate: " << learning_rate << std::endl;
			if(clip_gradient) {
				std::cout << "Gradient Clipping Threshold (Norm Ball): " << norm_clip << std::endl;
			}
			std::cout << "Parameter initialization range (uniform): " << lower_range << " " << upper_range << "\n";
			std::cout << "----------------------------------------------------------\n\n";
			if(truncated_softmax) {
				std::cout << "-------------------Truncated softmax info----------------------\n";
				std::cout << "Shortlist Size: " << shortlist_size << std::endl;
				std::cout << "Sampled Size: " << sampled_size << std::endl;
				std::cout << "---------------------------------------------------------------\n\n";
			}
		}
		std::cout << "------------------------Model Info------------------------\n";
		if(LM) {
			std::cout << "Sequence model\n";
		}
		else {
			std::cout << "Sequence to sequence model\n";
		}
		std::cout << "Source Vocab Size: " << source_vocab_size << std::endl;
		std::cout << "Target Vocab Size: " << target_vocab_size << std::endl;
		std::cout << "Number of Hidden Units: " << LSTM_size << std::endl;
		std::cout << "Number of Hidden Layers: " << num_hidden_layers << std::endl;
		std::cout << "---------------------------------------------------------------\n\n";
		if(decode) {
			std::cout << "------------------------Decode Info------------------------\n";
			std::cout << "Beam size for kbest: " << beam_size << "\n";
			std::cout << "Number of paths for kbest: " << num_hypotheses << "\n";
			std::cout << "------------------------------------------------------------\n\n";
		}
		if(stochastic_generation) {
			std::cout << "------------------------Stoch Generation Info------------------------\n";
			std::cout << "Number of tokens for stoch generation: " << sg_length << "\n";
			std::cout << "------------------------------------------------------------\n\n";
		}

		//std::cout << "Number of Lines in Training File: " << train_num_lines_in_file << std::endl;
		std::cout << "\n\n";
	}

};


