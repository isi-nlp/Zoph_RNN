 //The LSTM file that contains all the info for the LSTM that is needed for forward and backward propagation for gradient calculations

#ifndef LSTM_IH_H
#define LSTM_IH_H

#include <Eigen/Dense>
#include "Eigen_Util.h"

#include "model.h"

//Forward declaration
template<typename dType>
class neuralMT_model;

template<typename dType>
class Input_To_Hidden_Layer;

template<typename dType>
class LSTM_IH_Node {
public:
	//Pointer to the model struct, so it can access all of the weight matrices
	Input_To_Hidden_Layer<precision> *model;

	//--------------------------------------------------GPU parameters------------------------------------
	int minibatch_size;
	int LSTM_size;
	int index;
	bool dropout;
	dType dropout_rate;
	dType *d_dropout_mask;
	bool attention_model = false; //this will only be true for the upper layer on the target side of the LSTM
	bool feed_input = false;


	//host pointers
	dType *h_o_t;
	dType *h_c_t;
	dType *h_d_ERRt_ht;
	int *h_input_vocab_indices_01;
	int *h_input_vocab_indices;
	dType *h_f_t;
	dType *h_c_t_prev;
	dType *h_c_prime_t_tanh;
	dType *h_i_t;
	dType *h_h_t_prev;

	dType *h_sparse_lookup;

	dType *h_h_t;

	//device pointers
	dType *d_d_ERRnTOtp1_ht;
	dType *d_d_ERRnTOtp1_ct;
	dType *d_d_ERRt_ht;
	dType *d_o_t;
	dType *d_c_t;
	int *d_input_vocab_indices_01;
	int *d_input_vocab_indices;
	dType *d_f_t;
	dType *d_c_t_prev;
	dType *d_c_prime_t_tanh;
	dType *d_i_t;

	dType *d_h_t_prev;
	dType *d_sparse_lookup;
	dType *d_h_t;
	dType *d_zeros; //points to a zero matrix that can be used for d_ERRt_ht in backprop
	dType *d_ERRnTOt_h_tild;
	dType *d_ERRnTOt_h_tild_cpy;
	dType *d_h_tild;


	//Constructor
	LSTM_IH_Node(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m,int index,dType *d_zero_ptr,bool dropout,
		dType dropout_rate);

	void init_LSTM_GPU(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m);


	void update_vectors_forward_GPU(int *d_input_vocab_indices,int *d_input_vocab_indices_01,
		dType *d_h_t_prev,dType *d_c_t_prev);

	//Compute the forward values for the LSTM node
	//This is after the node has recieved the previous hidden and cell state values
	void forward_prop();
	void forward_prop_GPU();

	void back_prop_GPU();

	//Update the gradient matrices
	void compute_gradients_GPU();

	void backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct);//,dType *d_d_ERRt_ht);

	void update_vectors_forward_decoder(int *d_input_vocab_indices,int *d_input_vocab_indices_01);

	void dump_LSTM(std::ofstream &LSTM_dump_stream,std::string intro);

	void send_h_t_above();

	void attention_extra();

};

#endif