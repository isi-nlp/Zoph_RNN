//Bidirectional source encoder for MT
#ifndef BI_ENCODER_H
#define BI_ENCODER_H

#include "gpu_info_struct.h"

template<typename dType>
class neuralMT_model;

enum model_type_t {SEND_REV,COMBINE};

template<typename dType>
class bi_encoder {
public:

	//notes
	//The final hidden layer indicies are in reversed order still
	int num_layers;
	int LSTM_size;
	int minibatch_size;
	int longest_sent;
	int longest_sent_minibatch = -1; //this must be sent per minibatch in training
	dType norm_clip;
	dType learning_rate;

	model_type_t model_type = SEND_REV;

	dType *d_top_param_rev; //for transforming the top indicies
	dType *d_top_param_nonrev; //for transforming the top indicies
	dType *d_top_bias; //for transforming the top indicies
	dType *d_top_param_rev_grad;
	dType *d_top_param_nonrev_grad;
	dType *d_top_bias_grad; 

	dType *d_ones_minibatch;
	dType *d_temp_result;
 	dType *d_result;

 	std::vector<dType*> d_temp_result_vec;
 	std::vector<dType*> d_result_vec;

 	std::vector<int> gpu_indicies;

	thrust::device_ptr<dType> thrust_d_top_param_rev_grad;
	thrust::device_ptr<dType> thrust_d_top_param_nonrev_grad;
	thrust::device_ptr<dType> thrust_d_top_bias_grad;

	std::vector<dType*> d_ht_rev_total; //size (LSTM size x minibatch size)
	std::vector<dType*> d_ht_nonrev_total; //size (LSTM size x minibatch size)
	std::vector<dType*> d_ht_rev_total_errors; //size (LSTM size x minibatch size)
	std::vector<dType*> d_ht_nonrev_total_errors; //size (LSTM size x minibatch size)
	std::vector<dType*> d_final_mats;
	std::vector<dType*> d_final_errors;

	std::vector<dType*> d_horiz_param_rev; //for transforming the top indicies
	std::vector<dType*> d_horiz_param_nonrev; //for transforming the top indicies
	std::vector<dType*> d_horiz_bias; //for transforming the top indicies
	std::vector<dType*> d_horiz_param_rev_grad;
	std::vector<dType*> d_horiz_param_nonrev_grad;
	std::vector<dType*> d_horiz_bias_grad;
	std::vector<dType*> d_hs_final_target; //these are the final states being sent to the decoder
	std::vector<dType*> d_ct_final_target;
	std::vector<dType*> d_horiz_param_rev_ct; 
	std::vector<dType*> d_horiz_param_nonrev_ct; 
	std::vector<dType*> d_horiz_param_rev_ct_grad; 
	std::vector<dType*> d_horiz_param_nonrev_ct_grad; 
	std::vector<dType*> d_horiz_bias_ct_grad;
	std::vector<dType*> d_ct_start_target; //these are the cell states from the final nonrev encoder
	std::vector<dType*> d_ct_rev_error_horiz;
	std::vector<dType*> d_ct_nonrev_error_horiz;

	std::vector<dType*> d_hs_start_target; //these are the hidden states from the final nonrev encoder
	std::vector<dType*> d_hs_rev_error_horiz;
	std::vector<dType*> d_hs_nonrev_error_horiz;
	std::vector<int> final_index_hs; //these are the indicies for the last hiddenstate on the nonrev sice

	dType *d_temp_error_1;
	dType *d_temp_error_2;

	bi_layer_info layer_info;

	int *h_source_indicies; //this is used to pass to the forward direction indicies (or another)
	int *h_source_indicies_mask; //for reversed direction mask
	int *d_source_indicies_mask;

	neuralMT_model<precision> *model;

	bi_encoder();

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);

	void reverse_indicies(int *h_vocab_indices,int len); //reverse the indicies and store them in h_source_indicies

	void init_layer(global_params &params,int device_number,neuralMT_model<dType> *model,std::vector<int> &gpu_indicies);

	void forward_prop();

	void back_prop();

	void clear_gradients();

	void check_all_gradients(dType epsilon);

	void update_weights();

	void calculate_global_norm();

	void dump_weights(std::ofstream &output);

	void load_weights(std::ifstream &input);

	void update_global_params();

};


#endif
