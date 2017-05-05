//Model file that contains the parameters for the model

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <Eigen/Dense>
#include "file_helper_decoder.h"
#include "fileHelper_source.h"
#include "decoder.h"
#include "LSTM.h"
#include "Eigen_Util.h"
//#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include "softmax.h"
#include <math.h>
#include <limits>
#include "Input_To_Hidden_Layer.h"
#include "Hidden_To_Hidden_Layer.h"
#include "memory_util.h"

template<typename dType>
class Input_To_Hidden_Layer;

template<typename dType>
class Hidden_To_Hidden_Layer;

struct file_helper;

namespace debug_flag {
	bool flag = false;
}


template<typename dType>
class neuralMT_model {
public:
	/////////////////////////////////Current minibatch info for the model///////////////////////////////////

	//loss layer for the model
	//softmax_layer<dType> *softmax;
	base_loss_layer<dType> *softmax;

	//First layer of model, the input to hidden layer
	Input_To_Hidden_Layer<dType> input_layer_source;
	Input_To_Hidden_Layer<dType> input_layer_target;

	//Hidden layers of model
	std::vector<Hidden_To_Hidden_Layer<dType>> source_hidden_layers;
	std::vector<Hidden_To_Hidden_Layer<dType>> target_hidden_layers;

	//extra source encoder for bi-directional stuff. In bidirectional case, the indicies are in the forward direction
	Input_To_Hidden_Layer<dType> input_layer_source_bi;
	std::vector<Hidden_To_Hidden_Layer<dType>> source_hidden_layers_bi;

	bi_encoder<dType> bi_dir_source; //the bidirectional wrapper

	encoder_multi_source<dType> multi_source_layer; //for multiple source languages


	/////////////////////////////////Other random stuff//////////////////////////////////////////////////

	file_helper *file_info;

	softmax_layer_gpu_info s_layer_info;

	std::ifstream input;
	std::ofstream output;

	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic> zero_error; //passed in from imaginary softmax for source side

	std::string input_weight_file;
	std::string output_weight_file; 

	bool debug;

	bool train_perplexity_mode;
	double train_perplexity=0;

	bool truncated_softmax;

	bool LM;// true if language model only, aka no source side

	bool train = false; //this is for makign sure dropout is not used at test time
	bool grad_check_flag = false;
	//for the attention model
	int source_length = -1;

	//for birdirectional layer
	bool bi_dir = false;
	bool multi_source = false;

	//attention model
	attention_params attent_params;
	std::ofstream output_alignments;

	//for visualizing the RNN
	bool dump_LSTM;
	std::ofstream LSTM_stream_dump;

	//for decoding multilayer models, on index for each layer
	bool decode = false;
	std::vector<prev_source_state<dType>> previous_source_states;
	std::vector<prev_source_state<dType>> previous_source_states_bi; //for bi_directional encoder
	std::vector<prev_target_state<dType>> previous_target_states;
	std::vector<dType*> top_source_states; //for attention model in decoder
	std::vector<dType*> top_source_states_v2; //for attention model in decoder
	attention_layer<dType> decoder_att_layer; //for decoding only

	bool multi_attention = false;
	bool multi_attention_v2 = false;

	file_helper_source src_fh; //for training
	file_helper_source *src_fh_test; //for training
	std::string multisource_file; //for training and testing, it is the path to the correct file

	bool char_cnn = false;
	char_cnn_params char_params;

    Timer timer;
    
	///////////////////////////////////Methods for the class//////////////////////////////////////////////

	neuralMT_model() {};

	//Called at beginning of program once to initialize the weights
	void initModel(int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
		int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
		std::string input_weight_file,std::string output_weight_file,bool scaled,bool train_perplexity,
		bool truncated_softmax,int shortlist_size,int sampled_size,bool LM,int num_layers,std::vector<int> gpu_indicies,
		bool dropout,dType dropout_rate,struct attention_params attent_params,global_params &params);

	//For the decoder
	void initModel_decoding(int LSTM_size,int beam_size,int source_vocab_size,int target_vocab_size,
		int num_layers,std::string input_weight_file,int gpu_num,global_params &params,
		bool attention_model,bool feed_input,bool multi_source,bool combine_LSTM,bool char_cnn);

	//This initializes the streams,event and cuBLAS handlers, along with setting the GPU's for the layers
	void init_GPUs();

	//Dumps all the GPU info
	void print_GPU_Info();

	//initialize prev states for decoding
	void init_prev_states(int num_layers, int LSTM_size,int minibatch_size, int device_number,bool multi_source);

	//Gets called one minibatch is formulated into a matrix
	//This matrix is then passed in and forward/back prop is done, then gradients are updated
	template<typename Derived>
	void compute_gradients(const Eigen::MatrixBase<Derived> &source_input_minibatch_const,
		const Eigen::MatrixBase<Derived> &source_output_minibatch_const,const Eigen::MatrixBase<Derived> &target_input_minibatch_const,
		const Eigen::MatrixBase<Derived> &target_output_minibatch_const,int *h_input_vocab_indicies_source,
		int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_output_vocab_indicies_target,
		int current_source_length,int current_target_length,int *h_output_vocab_indicies_source_Wgrad,
		int *h_input_vocab_indicies_target_Wgrad,int len_source_Wgrad,int len_target_Wgrad,int *h_sampled_indices,
		int len_unique_words_trunc_softmax,int *h_batch_info,file_helper *temp_fh);

	//Sets all gradient matrices to zero, called after a minibatch updates the gradients
	void clear_gradients();

	//Called after you get gradients for the current minibatch
	void updateParameters();

	void check_all_gradients(dType epsilon);

	//Get the sum of all errors in the minibatch
	double getError(bool GPU);


	//Runs gradient check on a parameter vector or matrix
	template<typename Derived,typename Derived3>
	void check_gradient(dType epsilon,const Eigen::MatrixBase<Derived3> &parameter_const,const Eigen::MatrixBase<Derived> &grad);

	//Called after each minibatch, once the gradients are calculated
	void update_weights();

	void update_weights_OLD(); //per matrix clipping

	//Output the weights to a file
	void dump_weights();

	void dump_best_model(std::string best_model_name,std::string const_model);

	//Read in Weights from file
	void load_weights();

	void update_learning_rate(dType new_learning_rate);

	//gets the perplexity of a file
	double get_perplexity(std::string test_file_name,int minibatch_size,int &test_num_lines_in_file, int longest_sent,
		int source_vocab_size,int target_vocab_size,bool load_weights_val,int &test_total_words,
		bool HPC_output_flag,bool force_decode,std::string fd_filename);

	//Maps the file info pointer to the model
	void initFileInfo(struct file_helper *file_info);

	void stoicastic_generation(int length,std::string output_file_name,double temperature);

	void forward_prop_source(int *d_input_vocab_indicies_source,int *d_input_vocab_indicies_source_bi,int *d_ones,
		int source_length,int source_length_bi,int LSTM_size,int *d_char_cnn_indicies);

	void forward_prop_target(int curr_index,int *d_current_indicies,int *d_ones,int LSTM_size, int beam_size,
		int *d_char_cnn_indicies);

	template<typename Derived>
	void swap_decoding_states(const Eigen::MatrixBase<Derived> &indicies,int index,dType *d_temp_swap_vals);

	void target_copy_prev_states(int LSTM_size, int beam_size);

	void dump_alignments(int target_length,int minibatch_size,int *h_input_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_input_vocab_indicies_source_2);
    
    // for fsa line
    void get_chts(std::vector<Eigen::Matrix<dType, Eigen::Dynamic,1>> &chts, int beam_index, int beam_size);
    
    void set_chts(const std::vector<Eigen::Matrix<dType, Eigen::Dynamic,1>>& chts, int beam_size);
};

#endif
