//Bidirectional source encoder for MT
#ifndef ENCODER_MULTI_SOURCE_H
#define ENCODER_MULTI_SOURCE_H

#include "gpu_info_struct.h"
#include "tree_LSTM.h"

template<typename dType>
class neuralMT_model;

template<typename dType>
class encoder_multi_source {
public:

	//notes
	//The final hidden layer indicies are in reversed order still
	int num_layers;
	int LSTM_size;
	int minibatch_size;
	dType norm_clip;
	dType learning_rate;
	int longest_sent_minibatch_s1 = -1;
	int longest_sent_minibatch_s2 = -1;

	bool decode = false; //are we decoding? 


	//for norm clipping
 	std::vector<dType*> d_temp_result_vec;
 	std::vector<dType*> d_result_vec;

 	std::vector<int> gpu_indicies;

	std::vector<dType*> d_ht_s1_total; //size (LSTM size x minibatch size)
	std::vector<dType*> d_ht_s2_total; //size (LSTM size x minibatch size)
	std::vector<dType*> d_ht_s1_total_errors; //size (LSTM size x minibatch size)
	std::vector<dType*> d_ht_s2_total_errors; //size (LSTM size x minibatch size)
	std::vector<dType*> d_final_mats;
	std::vector<dType*> d_final_errors;

	std::vector<dType*> d_horiz_param_s1; //for transforming the top indicies
	std::vector<dType*> d_horiz_param_s2; //for transforming the top indicies
	std::vector<dType*> d_horiz_bias; //for transforming the top indicies
	std::vector<dType*> d_horiz_param_s1_grad;
	std::vector<dType*> d_horiz_param_s2_grad;
	std::vector<dType*> d_horiz_bias_grad;
	std::vector<dType*> d_hs_final_target; //these are the final states being sent to the decoder
	std::vector<dType*> d_ct_final_target; //these are the final states being sent to the decoder
	std::vector<dType*> d_horiz_param_s1_ct; 
	std::vector<dType*> d_horiz_param_s2_ct; 
	std::vector<dType*> d_horiz_param_s1_ct_grad; 
	std::vector<dType*> d_horiz_param_s2_ct_grad; 
	std::vector<dType*> d_horiz_bias_ct_grad;

	std::vector<dType*> d_ct_s1_error_horiz;
	std::vector<dType*> d_ct_s2_error_horiz;
	std::vector<dType*> d_hs_s1_error_horiz;
	std::vector<dType*> d_hs_s2_error_horiz;

	neuralMT_model<precision> *model;

	//stuff for combining using a tree LSTM
	bool lstm_combine=false; //combine using a tree LSTM variant
	std::vector<tree_LSTM<dType>*> lstm_combiner_layers;


	encoder_multi_source();

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols,int gpu_index);

	void init_layer(global_params &params,neuralMT_model<dType> *model,std::vector<int> &gpu_indicies);

	void init_layer_decoder(neuralMT_model<dType> *model,int gpu_num,bool lstm_combine,int LSTM_size,int num_layers);

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
