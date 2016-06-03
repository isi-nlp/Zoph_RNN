//tree LSTM for 2 children
#ifndef TREE_LSTM_H
#define TREE_LSTM_H


template<typename dType>
class encoder_multi_source;

template<typename dType>
class tree_LSTM {
public:

	int device_number;
	cudaStream_t s0;
	cublasHandle_t handle;
	int LSTM_size;
	int minibatch_size;
	bool clip_gradients = false;
	dType norm_clip;

	//hidden and cell states
	dType *d_child_ht_1;
	dType *d_child_ht_2;
	dType *d_child_ct_1;
	dType *d_child_ct_2;

	dType *d_ones_minibatch;

	//parameters
	//biases
	dType *d_b_i;
	dType *d_b_f; //initialize to one
	dType *d_b_o;
	dType *d_b_c;
	//for hidden states
	dType *d_M_i_1;
	dType *d_M_f_1;
	dType *d_M_o_1;
	dType *d_M_c_1;
	dType *d_M_i_2;
	dType *d_M_f_2;
	dType *d_M_o_2;
	dType *d_M_c_2;

	//biases
	dType *d_b_i_grad;
	dType *d_b_f_grad; //initialize to one
	dType *d_b_o_grad;
	dType *d_b_c_grad;
	//for hidden states
	dType *d_M_i_1_grad;
	dType *d_M_f_1_grad;
	dType *d_M_o_1_grad;
	dType *d_M_c_1_grad;
	dType *d_M_i_2_grad;
	dType *d_M_f_2_grad;
	dType *d_M_o_2_grad;
	dType *d_M_c_2_grad;


	//forward prop values
	dType *d_i_t;
	dType *d_f_t_1;
	dType *d_f_t_2;
	dType *d_c_prime_t_tanh;
	dType *d_o_t;
	dType *d_c_t;
	dType *d_h_t;

	//temp stuff
	dType *d_temp1;
	dType *d_temp2;
	dType *d_temp3;
	dType *d_temp4;
	dType *d_temp5;
	dType *d_temp6;
	dType *d_temp7;
	dType *d_temp8;

	//backprop errors
	dType *d_d_ERRnTOt_ht; //future h_t error stored here
	dType *d_d_ERRnTOtp1_ct; //future c_t error stored here
	dType *d_d_ERRt_ct; //cell error with tree LSTM stored here
	dType *d_d_ERRnTOt_ct; //sum of the two cell errors
	dType *d_d_ERRnTOt_it;
	dType *d_d_ERRnTOt_ot;
	dType *d_d_ERRnTOt_ft_1;
	dType *d_d_ERRnTOt_ft_2;
	dType *d_d_ERRnTOt_tanhcpt;

	//for children hidden states
	dType *d_d_ERRnTOt_h1;
	dType *d_d_ERRnTOt_h2;
	dType *d_d_ERRnTOt_c1;
	dType *d_d_ERRnTOt_c2;

	dType *d_temp_result;
	dType *d_result;

	encoder_multi_source<dType> *model;

	//for training
	tree_LSTM(global_params &params,int device_number,encoder_multi_source<dType> *model);

	//for decoding
	tree_LSTM(int LSTM_size,int device_number,encoder_multi_source<dType> *model);

	void forward();

	void backward();

	void clear_gradients();

	void check_all_gradients(dType epsilon);

	void update_weights();

	void calculate_global_norm();

	void dump_weights(std::ofstream &output);

	void load_weights(std::ifstream &input);

	void update_global_params();

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols,int gpu_index);
};


#endif