//for combining two attention layers

#ifndef ATT_COMBINER_H
#define ATT_COMBINER_H

#include "attention_combiner_node.h"

template<typename dType>
class attention_combiner_layer {
public:

	int LSTM_size;
	int minibatch_size;
	int longest_sent;
	int device_number;
	dType norm_clip;

	bool transfer_done = false;

	bool add_ht = false;

	dType *d_M_1;
	dType *d_M_2;
	dType *d_b_d;

	dType *d_M_1_grad;
	dType *d_M_2_grad;
	dType *d_b_d_grad;

	dType *d_result;
	dType *d_temp_result;

	//events and streams
	cublasHandle_t handle;
	cudaEvent_t start_forward;
	cudaEvent_t start_backward;
	cudaEvent_t forward_prop_done;
	cudaEvent_t backward_prop_done;
	cudaEvent_t error_htild_below;
	cudaStream_t s0;

	std::vector<attention_combiner_node<dType>*> nodes;

	neuralMT_model<dType> *model;

	thrust::device_ptr<dType> thrust_d_M_1_grad;
	thrust::device_ptr<dType> thrust_d_M_2_grad;
	thrust::device_ptr<dType> thrust_d_b_d_grad;


	attention_combiner_layer(global_params &params,int device_number,neuralMT_model<dType> *model);

	void clear_gradients();

	void check_gradients(dType epsilon);

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);

	void dump_weights(std::ofstream &output);

	void load_weights(std::ifstream &input);

	void clip_gradients_func();

	void scale_gradients();

	void update_params();

	void norm_p1();

	void norm_p2();

	void clip_indiv();

};

#endif