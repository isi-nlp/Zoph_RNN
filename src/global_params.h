//Global parameter file that needs to be specified for learning

typedef float precision;
#define NDEBUG
//#define REMOVE_STREAMS //this gets rid of all stream parallelism
//#define NAN_DEBUG
//#define REMOVE_STREAMS_FEED_INPUT

struct attention_params {
	bool attention_model = false;
	int D = 10;
	bool feed_input = false;
	bool dump_alignments = false;
	std::string tmp_alignment_file = "NULL";
	std::string alignment_file = "alignments.txt";
};

struct bi_directional_params {
	bool bi_dir = false;
	bool bi_dir_comb = false;
	bool share_embeddings = false;
};

struct multi_source_params {
	bool multi_source = false;
	bool multi_attention = false;
	bool multi_attention_v2 = false;
	bool add_ht = false; //add the hidden states instead of sending them through a neural network
	bool lstm_combine = false;
	std::string file_name = "NULL"; //this is for the training data for the additional file
	std::string int_file_name = "/multi_source.txt";//the integerized file name in the booth path
	std::string source_model_name = "NULL";//specified by user
	std::string int_file_name_test = "/validation_multi.txt";
	std::string test_file_name = "NULL";//specified by the user

	std::string ensemble_train_file_name = "NULL";
};

struct char_cnn_params {

	bool char_cnn = false;
	int longest_word; //learn this from char mapping file
	int filter_size;
	int num_unique_chars_source; //learn this from char mapping file
	int num_unique_chars_target; //learn this from char mapping file
	int char_emb_size;

	int num_highway_layers;

	std::string char_mapping_file = "char_mapping.nn";
	std::string word_mapping_file = "word_mapping.nn";
	std::string char_train_file = "train_char.txt.brz";
	std::string word_train_file = "train_word.txt.brz";
	std::string char_dev_file = "dev_char.txt.brz";
	std::string word_dev_file = "dev_word.txt.brz";
	std::string char_test_file = "test_char.txt.brz";
	std::string word_test_file = "test_word.txt.brz";
};


struct global_params {	


	//for file system cleanup
	std::string unique_dir= "NULL";


	//for restarting model training
	std::string load_model_name = "NULL";
	bool load_model_train=false;


	//for training a model with the same indicies as another models for ensembles
	std::string ensemble_train_file_name = "NULL";
	bool ensemble_train = false;


	//for dropout
	bool dropout = false;
	precision dropout_rate = 1.0; //probability of a node being kept

	//for random seed
	bool random_seed = false;
	int random_seed_int = -1;

	std::string tmp_location=""; //location where tmp directory will be created

	//for charCNN
	char_cnn_params char_params;

	//for the attention model
	attention_params attent_params;
	// bool clip_cell = false;
	// precision cell_clip_threshold = 50;

	//for individual gradient clipping
	bool individual_grad_clip = false;
	precision ind_norm_clip_thres = 0.1;

	//for gradient clipping whole matrices
	bool clip_gradient = true;
	precision norm_clip = 5.0; //Renormalize the gradients so they fit in normball this size, this is also used for total threshold


	//for loss functions
	bool softmax = true;


	//nce
	bool NCE = false;
	int num_negative_samples = 500;
	bool share_samples = true;


	//UNK replace
	bool unk_replace = false;
	int unk_aligned_width = 7;


	//bidirectional encoder
	bi_directional_params bi_dir_params;

	//multi source
	multi_source_params multi_src_params;

	//General settings
	static const bool debug = false;
	bool train = true; //If you want to train the model
	bool test = false; //If you want to test the model
	bool decode = false;
	bool train_perplexity = true; //print out the train perplexity every epoch (or half epoch if you have a learning rate schedule)
	bool LM = false; //if true it is only a sequence model, not sequence to sequence, sequence to sequence is the default
	bool shuffle = true; //shuffle the training data

	bool stochastic_generation = false; //This is for Language modeling only
	int sg_length=10; //how many tokens to generate
	std::string sg_output_file = "sg.txt";
	std::string sg_output_file_temp = "NULL";
	double temperature=1;


	bool HPC_output = false; //flushes the output to a file, so it can be read as the program executes
	std::string HPC_output_file_name = "logfile.txt";

	//Model training info
	int minibatch_size = 8; //Size of the minibatch
	int num_epochs = 10; //Number passes through the dataset
	precision learning_rate = 0.5; //The learning rate for SGD

	//stuff for the google learning rate
	//This halves the learning rate every 0.5 epochs after some inital epoch
	bool google_learning_rate = false;
	int epoch_to_start_halving = 6; //After what epoch do you halve the learnign rate
	int half_way_count = -1; //What is the total number of words that mark half an epoch

	bool stanford_learning_rate = false;
	precision stanford_decrease_factor = 0.5;
	int epoch_to_start_halving_full = 6;


	//stuff for normal halving of the learning rate where every half epoch the validation set is looked at 
	//and if it didn't improve, or did worse, the learning rate is halved.
	//NOTE do not have on google learning rate and the normal learning rate schedule
	bool learning_rate_schedule = false;
	precision decrease_factor = 0.5;
	double margin = 0.0; 
	std::string dev_source_file_name;
	std::string dev_target_file_name;


	//note this is only for GPU, not for CPU testing
	//always use this as thrust cannot use streams until I download new version, fix this for performance when using
	const bool softmax_scaled = true; //This is for if you want to scale all values in the outputdist to avoid overflow when exping floats


	//the truncated softmax
	//top_fixed + sampled = target vocabulary
	bool truncated_softmax =false;
	int shortlist_size = 10000;
	int sampled_size = 5000;


	//Model size info
	//vocab size of -1 defaults to the size of the train file specified
	int source_vocab_size = -1;
	int target_vocab_size = -1; //Size in input vocabulary, ranging from 0-input_vocab_size, where 0 is start symbol
	int LSTM_size = 100; //LSTM cell size, by definition it is the same as the word embedding layer
	int num_layers = 1; //This is the number of stacked LSTM's in the model
	std::vector<int> gpu_indicies;//for training with multiple gpu's


	////////////////////Decoder settings//////////////////
	int beam_size = 12;
	precision penalty = 0;
	int num_hypotheses = 1;//This prints out the k best paths from the beam decoder for the input
	precision min_decoding_ratio = 0.5; //target translation divided by source sentence length must be greater than min_decoding_ratio
	precision max_decoding_ratio = 1.5;
	bool print_score = false; //Whether to print the score of the hypotheses or not
	//std::string decode_tmp_file; //used for tmp stuff

	std::vector<std::string> decode_user_files;//source file being decoded
	std::vector<std::string> decode_user_files_additional;//source file being decoded
	std::vector<std::string> decode_temp_files;//one for each model being decoded
	std::vector<std::string> decode_temp_files_additional;//one for each model being decoded
	std::string decoder_output_file = "NULL"; //decoder output in temp before integerization
	std::vector<std::string> model_names; //for kbest ensembles
	std::vector<std::string> model_names_multi_src;//NULL value represents not using one


	std::string decoder_final_file; //what to output the final outputs to for decoding
	int decode_num_lines_in_file = -1;//This is learned


	//this is the file for outputting the hidden,cell states, etc.
	//format
	//1. xt=a,embedding
	//2. forget gate
	//3. input gate
	//4. c_t
	//5. output
	//6. h_t
	//7. probabilities
	bool dump_LSTM=false;
	std::string LSTM_dump_file;


	//for printing stuff to the screen
	int screen_print_rate=5;

	//for saving the best model for training
	std::string best_model_file_name;
	bool best_model=false;
	double best_model_perp = DBL_MAX;

	/////////////////////////////I/O file info/////////////////////////////
	std::string source_file_name; //for training, kbest, force decode and sg
	std::string target_file_name; //for training, kbest, force decode and sg
	std::string output_force_decode;

	int longest_sent = 100; //Note this doubles when doing translation, it is really 4 less than it is

	std::string train_file_name = "NULL";//Input file where source is first line then target is second line
	int train_num_lines_in_file = -1; //This is learned
	int train_total_words = -1; //This is learned

	std::string test_file_name = "NULL";//Input file where source is first line then target is second line
	int test_num_lines_in_file = -1; //This is learned
	int test_total_words = -1; //This is learned

	//For weights
	std::string input_weight_file = "model.nn";
	std::string output_weight_file = "model.nn";

    //For fsa
    std::string fsa_file = "";
    float fsa_weight = 0.0;
    bool print_beam = false;
    bool fsa_log = true;
    bool interactive = false;
    bool interactive_line = false;
    precision repeat_penalty = 0;
    precision adjacent_repeat_penalty = 0;
    float wordlen_weight = 0;
    float alliteration_weight = 0;

    
    // for encourage list
    std::vector<std::string> encourage_list;
    std::vector<float> encourage_weight;
    std::string encourage_weight_str = "";
    

    // for LSH decoding
    // 0: no LSH; 1 Winner-takes-all 
    int LSH_type = 0;
    // for WTA
    int WTA_K = 8;
    int WTA_units_per_band = 2; // log2(WTA_K) * WTA_units_per_band <= 32 (unsigned int)
    int WTA_W = 100; // number of bands;
    int show_debug_info = 0;
    int WTA_m = 10;
    int WTA_threshold = 1;
    int WTA_topn = 0;

    // for target vocab set shrink
    // 0 full softmax
    // 1 top 10k
    // 2 with alignment
    int target_vocab_policy = 0;
    // if 1
    int top_vocab_size = 10;
    // if 2
    std::string alignment_file = "";
    int target_vocab_cap = 1;
    // for lagecy-model
    bool legacy_model = false;
    
    
};


