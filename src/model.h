//Model file that contains the parameters for the model

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <Eigen/Dense>
#include "file_helper_decoder.h"
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

template<typename dType>
class Input_To_Hidden_Layer;

struct file_helper;

template<typename dType>
class neuralMT_model {
public:
	/////////////////////////////////Current minibatch info for the model///////////////////////////////////

	//Softmax layer for the model
	softmax_layer<dType> softmax;

	//First layer of model, the input to hidden layer
	Input_To_Hidden_Layer<dType> input_layer_source;
	Input_To_Hidden_Layer<dType> input_layer_target;

	input_layer_gpu_info ih_layer_info;
	softmax_layer_gpu_info s_layer_info;

	//Hidden layer of model
	//Hidden_To_Hidden_Layer hidden_layer;

	/////////////////////////////////Other random stuff//////////////////////////////////////////////////

	file_helper *file_info;

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

	///////////////////////////////////Methods for the class//////////////////////////////////////////////

	//Called at beginning of program once to initialize the weights
	void initModel(int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
		int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
		std::string input_weight_file,std::string output_weight_file,bool scaled,bool train_perplexity,
		bool truncated_softmax,int shortlist_size,int sampled_size,bool LM);

	//For the decoder
	void initModel_decoding(int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
 		int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
 		std::string input_weight_file,std::string output_weight_file,bool scaled);

	//This initializes the streams,event and cuBLAS handlers, along with setting the GPU's for the layers
	void init_GPUs();

	//Dumps all the GPU info
	void print_GPU_Info();

	//Gets called one minibatch is formulated into a matrix
	//This matrix is then passed in and forward/back prop is done, then gradients are updated
	template<typename Derived>
	void compute_gradients(const Eigen::MatrixBase<Derived> &source_input_minibatch_const,
		const Eigen::MatrixBase<Derived> &source_output_minibatch_const,const Eigen::MatrixBase<Derived> &target_input_minibatch_const,
		const Eigen::MatrixBase<Derived> &target_output_minibatch_const,int *h_input_vocab_indicies_source,
		int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_output_vocab_indicies_target,
		int current_source_length,int current_target_length,int *h_output_vocab_indicies_source_Wgrad,
		int *h_input_vocab_indicies_target_Wgrad,int len_source_Wgrad,int len_target_Wgrad,int *h_sampled_indices,
		int len_unique_words_trunc_softmax);

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

	//Output the weights to a file
	void dump_weights();

	//Read in Weights from file
	void load_weights();

	void update_learning_rate(dType new_learning_rate);

	//gets the perplexity of a file
	double get_perplexity(std::string test_file_name,int minibatch_size,int &test_num_lines_in_file, int longest_sent,
		int source_vocab_size,int target_vocab_size,std::ofstream &HPC_output,bool load_weights_val,int &test_total_words,
		bool HPC_output_flag,bool force_decode,std::string fd_filename);

	//Maps the file info pointer to the model
	void initFileInfo(struct file_helper *file_info);

	//This is the beam decoder
	void beam_decoder(int beam_size,std::string input_file_name,
		std::string input_weight_file_name,int num_lines_in_file,int source_vocab_size,int target_vocab_size,
		int longest_sent,int LSTM_size,dType penalty,std::string decoder_output_file,dType min_decoding_ratio,
		dType max_decoding_ratio,bool scaled,int num_hypotheses,bool print_score);

	template<typename Derived>
	void decoder_forward_prop_source(const Eigen::MatrixBase<Derived> &source_vocab_indices,int *d_input_vocab_indicies_source,int *d_ones);

	void decoder_forward_prop_target(struct file_helper_decoder *fh,struct decoder<dType> *d,int *d_ones,
		int curr_index,dType *h_outputdist);

	template<typename Derived>
	void copy_dist_to_eigen(dType *h_outputdist,const Eigen::MatrixBase<Derived> &outputdist_const);


	void stoicastic_generation(int length,std::string output_file_name);

};

#endif