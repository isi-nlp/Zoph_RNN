 //The LSTM file that contains all the info for the LSTM that is needed for forward and backward propagation for gradient calculations

#ifndef LSTM_HH_H
#define LSTM_HH_H

#include <Eigen/Dense>
#include "Eigen_Util.h"

#include "model.h"
#include "Hidden_To_Hidden_Layer.h"

//Forward declaration
template<typename dType>
struct neuralMT_model;

struct Hidden_To_Hidden_Layer;

struct LSTM_HH_Node {
	//Stored after forward propagation
	//These have dimension (hidden state size)x(size of minibatch)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> i_t;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> f_t;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> c_t;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> c_prime_t_tanh;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> o_t;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_t;

	//Needed for forward propagation
	//These have dimension (hidden state size)x(size of minibatch)
	//There are being passed from the left node
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_t_prev; //h_(t-1)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> c_t_prev; //c_(t-1)

	//These are being passed from below
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_t_prev_b; //h_(t-1)


	//This is the length of the minibatch
	//Each index ranges from 0 to (input vocab size-1), in order to select matrix column from embedding layer
	//This is for softmax
	Eigen::Matrix<int,Eigen::Dynamic,1> vocab_indices_output;

	//Pointer to the model struct, so it can access all of the weight matrices
	Hidden_To_Hidden_Layer *model;

	//This is the derivative of the error from time n to time t with respect to h_t
	//Has size (minibatch size)x(hidden state size)
	//This gets passed the the LSTM block below it
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_htM1_b;


	//These store the derivatives of errors for back propagation

	//This is the derivative of the error from time n to time t with respect to h_t
	//Has size (minibatch size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ht;

	//This is the derivative of the error from time n to time t with respect to o_t
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ot;

	//This is the derivative of the error from time n to time t with respect to c_t
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ct;

	//This is the derivative of the error at time t with respect to c_t
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRt_ct;

	//This is the derivative of the error from time n to time t with respect to f_t
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ft;

	//This is the derivative of the error from time n to time t with respect to tanhc'_t
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_tanhcpt;

	//This is the derivative of the error from time n to time t with respect to i_t
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_it;

	//This is the derivative of the error from time n to t with respect to h_(t-1)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_htM1;

	//This is the derivative of the error from time n to t with respect to c_(t-1)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> d_ERRnTOt_ctM1;

	//Used for precomputing, for solving for the W matrix
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Z_i;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Z_f;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Z_o;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Z_c;


	//Constructor
	LSTM_HH_Node(int LSTM_size,int minibatch_size,int vocab_size,struct Hidden_To_Hidden_Layer *m);

	//Update the hidden state and cell state vectors
	template<typename Derived>
	void update_vectors_forward(const Eigen::MatrixBase<Derived> &h_prev,
		const Eigen::MatrixBase<Derived> &c_prev,const Eigen::MatrixBase<Derived> &h_prev_b);

	//For input embedding layer
	//Pass in the weight matrix for the embedding layer 
	//Need to multithread later
	template<typename Derived>
	void compute_temp_mat(const Eigen::MatrixBase<Derived> &W_mat);

	//Compute the forward values for the LSTM node
	//This is after the node has recieved the previous hidden and cell state values
	void forward_prop();


	//Update the output vocab indicies
	template<typename Derived>
	void update_vectors_backward(const Eigen::MatrixBase<Derived> &vocab,int index);

	//Does back propagation for the nodes
	//Pass in the error from positions n to t+1 with respect to h_t
	template<typename Derived>
	void back_prop(const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ht,const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ct,const Eigen::MatrixBase<Derived> &d_ERRt_ht);


	//Called for the gradient update for input embedding layer
	template<typename Derived, typename Derived2>
	void sparseGradUpdate(const Eigen::MatrixBase<Derived> &grad_const, const Eigen::MatrixBase<Derived2> &d_Err);

	//Update the gradient matrices
	void update_gradients();

	//Compute the gradient for the W matrix, has seperate function because it is messy
	void compute_W_gradient();
};

#endif