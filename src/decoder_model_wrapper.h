#ifndef DECODER_MODEL_WRAPPER_H
#define DECODER_MODEL_WRAPPER_H

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

	neuralMT_model<dType> *model; //This is the model

	int source_length; //current length of the source sentence being decoded
	int beam_size;
	int source_vocab_size;
	int target_vocab_size;
	int num_layers;
	int LSTM_size;
	std::string weights_file_name;
	std::string input_file_name;

	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> outputdist;

	decoder_model_wrapper() {};
	decoder_model_wrapper(int gpu_num,int beam_size,std::string weights_file_name,int longest_sent,bool dump_LSTM,std::string LSTM_stream_dump_name,
		global_params &params);
	void extract_model_info(std::string weights_file_name); //get how many layers, hiddenstate size, vocab sizes, etc
	void memcpy_vocab_indicies(int *h_input_vocab_indicies_source,int sentence_length);
	void forward_prop_source();
	void forward_prop_target(int curr_index,int *h_current_indicies);


	template<typename Derived>
	void swap_decoding_states(const Eigen::MatrixBase<Derived> &indicies,int index);

	//copy h_outputdist to eigen
	template<typename Derived>
	void copy_dist_to_eigen(dType *h_outputdist,const Eigen::MatrixBase<Derived> &outputdist_const);

	void target_copy_prev_states();
};


#endif