#ifndef ATTENTION_LAYER_H
#define ATTENTION_LAYER_H


template<typename dType>
class neuralMT_model;

template<typename dType>
class attention_node;

#include "gpu_info_struct.h"

template<typename dType>
class attention_layer {
public:

	cublasHandle_t handle;
	int device_number;
	int LSTM_size;
	int minibatch_size;
	bool clip_gradients;
	dType norm_clip;
	bool feed_input = false;
	int longest_sent;
	bool transfer_done = false; //if true then take the copied matrix
	bool multi_attention_v2 = false;
	
	dType *d_W_a; //for the score function
	dType *d_v_p;
	dType *d_W_p;
	dType *d_W_c_p1;
	dType *d_W_c_p2;
	dType *d_output_bias;

	//for multi-attention_v2
	dType *d_W_c_p3_v2;
	dType *d_W_a_v2; //for the score function
	dType *d_v_p_v2;
	dType *d_W_p_v2;
	dType *d_W_c_p3_grad_v2;
	dType *d_W_a_grad_v2; //for the score function
	dType *d_v_p_grad_v2;
	dType *d_W_p_grad_v2;

	dType **d_total_hs_mat_v2;
	dType **d_total_hs_error_v2;
	dType *d_ERRnTOt_as_v2;
	dType *d_ERRnTOt_pt_v2;
	dType *d_ERRnTOt_ct_v2;
	dType *d_temp_1_v2; //LSTM by minibatch size
	dType *d_temp_Wa_grad_v2;
	dType *d_h_t_Wa_factor_v2; //for W_a gradient
	dType *d_h_t_sum_v2; //for summing weighted h_t's
	dType *d_h_s_sum_v2; //for summing weighted h_s
	int *d_batch_info_v2; // length of minibatches, then offsets

	thrust::device_ptr<dType> thrust_d_W_a_grad_v2;
	thrust::device_ptr<dType> thrust_d_v_p_grad_v2;
	thrust::device_ptr<dType> thrust_d_W_p_grad_v2;
	thrust::device_ptr<dType> thrust_d_W_c_p3_grad_v2;

	int *d_viterbi_alignments; //for decoding unk replacement

	dType *d_W_a_grad;
	dType *d_v_p_grad;
	dType *d_W_p_grad;
	dType *d_W_c_p1_grad;
	dType *d_W_c_p2_grad;
	dType *d_output_bias_grad;

	thrust::device_ptr<dType> thrust_d_W_a_grad;
	thrust::device_ptr<dType> thrust_d_v_p_grad;
	thrust::device_ptr<dType> thrust_d_W_p_grad;
	thrust::device_ptr<dType> thrust_d_W_c_p1_grad;
	thrust::device_ptr<dType> thrust_d_W_c_p2_grad;
	thrust::device_ptr<dType> thrust_d_output_bias_grad;
	dType *d_result; //for gradient clipping
	dType *d_temp_result; // for gradient clipping

	dType *d_ERRnTOt_ht_p1;
	dType *d_ERRnTOt_tan_htild;
	dType *d_ERRnTOt_ct;
	dType **d_total_hs_mat;
	dType **d_total_hs_error;

	dType *d_ones_minibatch;

	dType *d_ERRnTOt_as;
	dType *d_ERRnTOt_pt;

	dType *d_temp_1; //LSTM by minibatch size
	dType *d_temp_Wa_grad;
	dType *d_h_t_Wa_factor; //for W_a gradient
	dType *d_h_t_sum; //for summing weighted h_t's
	dType *d_h_s_sum; //for summing weighted h_s

	dType *d_ERRnTOt_htild_below; //from the input layer

	attention_layer_gpu_info layer_info; //stores the gpu info for the attention model
	curandGenerator_t rand_gen;


	int *d_batch_info; // length of minibatches, then offsets

	int *d_ones_minibatch_int;

	std::vector<attention_node<dType>> nodes;
	neuralMT_model<dType> *model;

	attention_layer() {};

	attention_layer(int LSTM_size,int minibatch_size, int device_number, int D, int longest_sent,cublasHandle_t &handle,neuralMT_model<dType> *model,
		bool feed_input,bool clip_gradients,dType norm_clip,bool dropout,dType dropout_rate,global_params &params,bool bi_side);

	void check_gradients(dType epsilon);

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);

	void clear_gradients();

	void prep_minibatch_info(int *h_batch_info);

	void prep_minibatch_info_v2(int *h_batch_info_v2);

	void dump_weights(std::ofstream &output);

	void load_weights(std::ifstream &input);

	void clip_gradients_func();

	void scale_gradients();

	void update_params();

	void norm_p1();

	void norm_p2();

	void clip_indiv();

	void init_att_decoder(int LSTM_size,int beam_size, int device_number, int D, int longest_sent,cublasHandle_t &handle,neuralMT_model<dType> *model,
		bool feed_input,std::vector<dType*> &top_source_states,bool multi_attention_v2,std::vector<dType*> &top_source_states_v2);

};


#endif