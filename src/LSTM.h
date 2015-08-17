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
	//Stored after forward propagation
	//These have dimension (hidden state size)x(size of minibatch)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> i_t;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> f_t;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> c_t;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> c_prime_t_tanh;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> o_t;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> h_t;

	//Needed for forward propagation
	//These have dimension (hidden state size)x(size of minibatch)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> h_t_prev; //h_(t-1)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> c_t_prev; //c_(t-1)

	//Temp matrix for computing embedding layer weight matrix multiplied by one-hot matrix
	//Dim (hidden state size)x(minibatch size)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> temp_mat;

	//This is the length of the minibatch
	//Each index ranges from 0 to (input vocab size-1), in order to select matrix column from embedding layer
	//This is for x_t, since it is a one-hot vector
	Eigen::Matrix<int,Eigen::Dynamic,1> vocab_indices_input;

	//This is the length of the minibatch
	//Each index ranges from 0 to (input vocab size-1), in order to select matrix column from embedding layer
	//This is for softmax
	Eigen::Matrix<int,Eigen::Dynamic,1> vocab_indices_output;

	//Pointer to the model struct, so it can access all of the weight matrices
	Input_To_Hidden_Layer<precision> *model;


	//These store the derivatives of errors for back propagation

	//This is the derivative of the error from time n to time t with respect to h_t
	//Has size (minibatch size)x(hidden state size)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ht;

	//This is the derivative of the error from time n to time t with respect to o_t
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ot;

	//This is the derivative of the error from time n to time t with respect to c_t
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ct;

	//This is the derivative of the error at time t with respect to c_t
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRt_ct;

	//This is the derivative of the error from time n to time t with respect to f_t
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ft;

	//This is the derivative of the error from time n to time t with respect to tanhc'_t
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_tanhcpt;

	//This is the derivative of the error from time n to time t with respect to i_t
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_it;

	//This is the derivative of the error from time n to t with respect to h_(t-1)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_htM1;

	//This is the derivative of the error from time n to t with respect to c_(t-1)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ctM1;

	//Used for precomputing, for solving for the W matrix
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> Z_i;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> Z_f;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> Z_o;
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> Z_c;


	//--------------------------------------------------GPU parameters------------------------------------
	int minibatch_size;
	int LSTM_size;

	//host pointers
	//dType *h_d_ERRnTOtp1_ht;
	//dType *h_d_ERRnTOtp1_ct;
	//dType *h_d_ERRt_ht;
	//dType *h_d_ERRnTOt_ht;
	dType *h_o_t;
	dType *h_c_t;
	//dType *h_d_ERRt_ct;
	//dType *h_d_ERRnTOt_ct;
	int *h_input_vocab_indices_01;
	int *h_input_vocab_indices;
	//dType *h_d_ERRnTOt_ot;
	dType *h_f_t;
	dType *h_c_t_prev;
	//dType *h_d_ERRnTOt_ft;
	dType *h_c_prime_t_tanh;
	dType *h_i_t;
	//dType *h_d_ERRnTOt_tanhcpt;
	//dType *h_d_ERRnTOt_it;

	//dType *h_d_ERRnTOt_htM1;
	//dType *h_d_ERRnTOt_ctM1;

	dType *h_h_t_prev;

	dType *h_sparse_lookup;

	dType *h_h_t;

	//device pointers
	dType *d_d_ERRnTOtp1_ht;
	dType *d_d_ERRnTOtp1_ct;
	dType *d_d_ERRt_ht;
	//dType *d_d_ERRnTOt_ht;
	dType *d_o_t;
	dType *d_c_t;
	//dType *d_d_ERRt_ct;
	//dType *d_d_ERRnTOt_ct;
	int *d_input_vocab_indices_01;
	int *d_input_vocab_indices;
	//dType *d_d_ERRnTOt_ot;
	dType *d_f_t;
	dType *d_c_t_prev;
	//dType *d_d_ERRnTOt_ft;
	dType *d_c_prime_t_tanh;
	dType *d_i_t;
	//dType *d_d_ERRnTOt_tanhcpt;
	//dType *d_d_ERRnTOt_it;

	//dType *d_d_ERRnTOt_htM1;
	//dType *d_d_ERRnTOt_ctM1;

	dType *d_h_t_prev;

	dType *d_sparse_lookup;

	dType *d_h_t;


	//Constructor
	LSTM_IH_Node(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m);

	void init_LSTM_CPU(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m);
	void init_LSTM_GPU(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m);

	//Update the hidden state and cell state vectors
	template<typename Derived,typename Derived2>
	void update_vectors_forward(const Eigen::MatrixBase<Derived> &h_prev,
		const Eigen::MatrixBase<Derived> &c_prev,
		const Eigen::MatrixBase<Derived2> &vocab,
		int index,int *d_input_vocab_indices,int *d_input_vocab_indices_01,
		dType *d_h_t_prev,dType *d_c_t_prev);

	template<typename Derived,typename Derived2>
	void update_vectors_forward_CPU(const Eigen::MatrixBase<Derived> &h_prev,
		const Eigen::MatrixBase<Derived> &c_prev,
		const Eigen::MatrixBase<Derived2> &vocab,int index);

	void update_vectors_forward_GPU(int *d_input_vocab_indices,int *d_input_vocab_indices_01,
		dType *d_h_t_prev,dType *d_c_t_prev);

	void update_vectors_forward_DEBUG();

	//For input embedding layer
	//Pass in the weight matrix for the embedding layer 
	//Need to multithread later
	template<typename Derived>
	void compute_temp_mat(const Eigen::MatrixBase<Derived> &W_mat);

	//Compute the forward values for the LSTM node
	//This is after the node has recieved the previous hidden and cell state values
	void forward_prop();
	void forward_prop_CPU();
	void forward_prop_GPU();


	//Update the output vocab indicies
	template<typename Derived>
	void update_vectors_backward(const Eigen::MatrixBase<Derived> &vocab,int index);

	//Does back propagation for the nodes
	//Pass in the error from positions n to t+1 with respect to h_t
	template<typename Derived>
	void back_prop(const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ht,const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ct,
		const Eigen::MatrixBase<Derived> &d_ERRt_ht);

	template<typename Derived>
	void back_prop_CPU(const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ht,const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ct,
	const Eigen::MatrixBase<Derived> &d_ERRt_ht);

	void back_prop_GPU();

	//Called for the gradient update for input embedding layer
	template<typename Derived, typename Derived2>
	void sparseGradUpdate(const Eigen::MatrixBase<Derived> &grad_const, const Eigen::MatrixBase<Derived2> &d_Err);

	//Update the gradient matrices
	void compute_gradients_CPU();
	void compute_gradients_GPU();

	//Compute the gradient for the W matrix, has seperate function because it is messy
	void compute_W_gradient_CPU();

	void backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct,dType *d_d_ERRt_ht);

	//get the error matrix from the softmax, which could lie on a different GPU
	void get_d_ERRt_ht_ONE(dType *d_d_ERRt_ht_softmax);
	void get_d_ERRt_ht_CPU(dType *d_d_ERRt_ht_softmax);
	void get_d_ERRt_ht_DMA(dType *d_d_ERRt_ht_softmax);

	template<typename Derived>
	void update_vectors_forward_decoder(const Eigen::MatrixBase<Derived> &vocab,int index,
		int *d_input_vocab_indices,int *d_input_vocab_indices_01);

};

#endif