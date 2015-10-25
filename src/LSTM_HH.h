 //The LSTM file that contains all the info for the LSTM that is needed for forward and backward propagation for gradient calculations

#ifndef LSTM_HH_H
#define LSTM_HH_H

#include <Eigen/Dense>
#include "Eigen_Util.h"

#include "model.h"

//Forward declaration
template<typename dType>
class neuralMT_model;

template<typename dType>
class Hidden_To_Hidden_Layer;

template<typename dType>
class LSTM_HH_Node {
public:

	//--------------------------------------------------GPU parameters------------------------------------
	int minibatch_size;
	int LSTM_size;
	int index; //what node is this
	bool attention_model = false; //this will only be true for the upper layer on the target side of the LSTM

	bool dropout;
	dType dropout_rate;
	dType *d_dropout_mask;

	//Pointer to the model struct, so it can access all of the weight matrices
	Hidden_To_Hidden_Layer<precision> *model;

	//host pointers
	dType *h_d_ERRt_ht;
	dType *h_o_t;
	dType *h_c_t;
	int *h_input_vocab_indices_01;
	dType *h_f_t;
	dType *h_c_t_prev;
	dType *h_c_prime_t_tanh;
	dType *h_i_t;
	dType *h_h_t_prev;
	dType *h_h_t;

	//device pointers
	dType *d_d_ERRnTOtp1_ht;
	dType *d_d_ERRnTOtp1_ct;
	dType *d_d_ERRt_ht;
	dType *d_o_t;
	dType *d_c_t;
	int *d_input_vocab_indices_01;
	dType *d_f_t;
	dType *d_c_t_prev;
	dType *d_c_prime_t_tanh;
	dType *d_i_t;
	dType *d_h_t_prev;
	dType *d_h_t;
	

	dType *h_h_t_below;
	dType *d_h_t_below;

	dType *d_zeros;

	dType *d_ERRnTOt_h_tild;
	dType *d_h_tild;


	//Constructor
	LSTM_HH_Node(int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m,int index,dType *d_zeros, bool dropout, 
		dType dropout_rate);

	void init_LSTM_GPU(int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m);

	void update_vectors_forward_GPU(int *d_input_vocab_indices_01,
		dType *d_h_t_prev,dType *d_c_t_prev);

	//Compute the forward values for the LSTM node
	//This is after the node has recieved the previous hidden and cell state values
	void forward_prop();
	void forward_prop_GPU();

	void forward_prop_sync(cudaStream_t &my_s);

	void back_prop_GPU(int index);

	//Update the gradient matrices
	void compute_gradients_GPU();

	void backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct);

	void update_vectors_forward_decoder(int *d_input_vocab_indices_01);

	void dump_LSTM(std::ofstream &LSTM_dump_stream,std::string intro);

	void send_h_t_above();
};

#endif