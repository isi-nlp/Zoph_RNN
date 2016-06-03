#ifndef HIGHWAY_NETWORK_H
#define HIGHWAY_NETWORK_H

#include "highway_node.h"

//highway network layer
template<typename dType>
class highway_network_layer {
public:

	cublasHandle_t handle;
	cudaStream_t s0;
	int device_number;
	int state_size;
	int minibatch_size;
	int longest_sent;
	dType norm_clip;


	#define RELU_NONLIN_BZ //use RELU, if not use tanh

	//params
	dType *d_W_h; //for ReLU gate
	dType *d_W_t; //for sigmoid gate
	dType *d_b_h;
	dType *d_b_t; //initialize to -2

	//gradients
	dType *d_W_h_grad; //for ReLU gate
	dType *d_W_t_grad; //for sigmoid gate
	dType *d_b_h_grad;
	dType *d_b_t_grad;

	thrust::device_ptr<dType> thrust_d_W_h_grad;
	thrust::device_ptr<dType> thrust_d_W_t_grad;
	thrust::device_ptr<dType> thrust_d_b_h_grad;
	thrust::device_ptr<dType> thrust_d_b_t_grad;

	dType *d_result;
	dType *d_temp_result;

	//forward prop values
	//these are stored in nodes

	dType *d_temp;
	dType *d_ones_minibatch;

	neuralMT_model<dType> *model;

	//back prop values
	dType *d_Err_t; //gate value
	dType *d_Err_y; //input
	dType *d_Err_g; //
	//dType *d_Err_z;//output error being passed in

	std::vector<highway_node<dType>*> nodes;

	void init(int state_size,int minibatch_size,int longest_sent,int device_number,
		cublasHandle_t &handle,cudaStream_t &s0,neuralMT_model<dType> *model,dType norm_clip);
	void forward(int index,dType *d_y_temp); 
	void backward(int index,dType *d_Err_z_temp);
	void clear_gradients();
	void check_gradients(dType epsilon);
	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
	void norm_p1();
	void norm_p2();
	void scale_gradients();
	void update_params();
	void clip_gradients_func();
	void dump_weights(std::ofstream &output);
	void load_weights(std::ifstream &input);
};





#endif
