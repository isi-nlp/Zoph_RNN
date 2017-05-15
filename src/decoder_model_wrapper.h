#ifndef DECODER_MODEL_WRAPPER_H
#define DECODER_MODEL_WRAPPER_H

#include "file_helper_decoder.h"

template<typename dType>
class neuralMT_model;

//the entire model will lie on one GPU, but different models in the ensemble can lie on different GPU's
template<typename dType>
class decoder_model_wrapper {
public:
	int gpu_num;
	int *d_ones; //vector of all ones, used for forward prop in beam search, on GPU
	dType *h_outputdist;
	dType *d_temp_swap_vals;
	int *d_input_vocab_indicies_source;
	int *d_current_indicies;
    
    int *h_current_indices; // every model should have this vector for model ensemble; 
    
    

	neuralMT_model<dType> *model; //This is the model

	file_helper_decoder *fileh; //for file input, so each file can get read in seperately
	file_helper_decoder *fileh_multi_src; //reads in additional multi-source file

	int source_length; //current length of the source sentence being decoded
	int beam_size;
	int source_vocab_size;
	int target_vocab_size;
	int num_layers;
	int LSTM_size;
	bool attention_model;
	bool feed_input;
	bool combine_LSTM;
	int num_lines_in_file = -1;
	int longest_sent;

	bool multi_source = false;
	int source_length_bi; //current length of the source sentence being decoded
	int *d_input_vocab_indicies_source_bi;

	bool char_cnn = false;
	int *d_char_vocab_indicies_source;
	int longest_word;
	std::unordered_map<int,std::vector<int>> word_to_char_map; //for word index, what is the character sequence, this is read from a file
	int *h_new_char_indicies;
	int *d_new_char_indicies;

	std::string main_weight_file;
	std::string multi_src_weight_file;
	std::string main_integerized_file;
	std::string multi_src_integerized_file;

	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> outputdist;
	std::vector<int> viterbi_alignments_ind; //individual viterbi alignments before voting
	std::vector<dType> viterbi_alignments_scores; //individual viterbi scores

    // for shrink the target set vocab;
    dType *d_D_shrink;
    dType *d_softmax_original_D; // a pointer to refer the d_D in original softmax;
    dType *d_b_shrink;
    dType *d_softmax_original_b;
    int new_output_vocab_size = 0;
    int *h_new_vocab_index;
    int *d_new_vocab_index;
    // for policy 1
    bool show_shrink_debug = false;
    bool policy_1_done = false;
    // for policy 2
    int *h_alignments; // [cap+1, source_vocab_size]
    int *d_alignments;
    int cap = 0;
    
    // for LSH
    int nnz = 0;
    int target_vocab_policy = 0;

    
    global_params * p_params;
    
    
	decoder_model_wrapper() {};
	decoder_model_wrapper(int gpu_num,int beam_size,
		std::string main_weight_file,std::string multi_src_weight_file,std::string main_integerized_file,
		std::string multi_src_integerized_file,int longest_sent,global_params &params);
	void extract_model_info(std::string weights_file_name); //get how many layers, hiddenstate size, vocab sizes, etc
	void memcpy_vocab_indicies();
    void prepare_target_vocab_set();
    
    void before_target_vocab_shrink();
    void after_target_vocab_shrink();
    
	void forward_prop_source();
	void forward_prop_target(int curr_index,int *h_current_indicies);


	template<typename Derived>
	void swap_decoding_states(const Eigen::MatrixBase<Derived> &indicies,int index);

	//copy h_outputdist to eigen
	template<typename Derived>
	void copy_dist_to_eigen(dType *h_outputdist,const Eigen::MatrixBase<Derived> &outputdist_const);

    template<typename Derived>
    void copy_dist_to_eigen(dType *h_outputdist,const Eigen::MatrixBase<Derived> &outputdist_const, int nnz);

    
	void target_copy_prev_states();
};


#endif
