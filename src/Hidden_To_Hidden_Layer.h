//LSTM layer that connects input to hidden
#ifndef LSTM_HIDDEN_TO_HIDDEN_H
#define LSTM_HIDDEN_TO_HIDDEN_H

#include "LSTM_HH.h"

template<typename dType>
struct neuralMT_model;

struct Hidden_To_Hidden_Layer {

	//Parameters for the model
	//The parameters need to connect input to input gate

	//previous hidden state vector to input gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hi_b;

	//previous hidden state vector to input gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hi;

	//bias for the input gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_i;


	//The parameters needed to connect input to forget gate

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hf_b;

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hf;

	//bias for the forget gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_f;


	//The parameters needed to connect input to cell state

	//previous hidden state to cell gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hc_b;

	//previous hidden state to cell gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hc;

	//bias for cell gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_c;



	//Parameters needed to connect input to output gate

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_ho_b;

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_ho;

	//bias for the forget gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_o;

	/////////////////////////////////Stores the gradients for the models/////////////////////////////////

	//The parameters need to connect input to input gate

	//previous hidden state vector to input gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_hi_b_grad;

	//previous hidden state vector to input gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_hi_grad;

	//bias for the input gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_i_grad;


	//The parameters needed to connect input to forget gate

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_hf_b_grad;

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_hf_grad;


	//bias for the forget gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_f_grad;


	//The parameters needed to connect input to cell state

	//previous hidden state to cell gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_hc_b_grad;

	//previous hidden state to cell gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_hc_grad;

	//bias for cell gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_c_grad;


	//Parameters needed to connect input to output gate

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_ho_b_grad;

	//previous hidden state to forget gate
	//Dimension (hidden state size)x(hidden state size)
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_ho_grad;


	//bias for the forget gate
	//Dimension (hidden state size)x(1)
	Eigen::Matrix<double, Eigen::Dynamic, 1> b_o_grad;



	/////////////////////////////////Current minibatch info for the model///////////////////////////////////
	std::vector<LSTM_HH_Node> nodes; //Stores all the LSTM nodes for forward and backward propagation

	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> init_hidden_vector; //Initial hidden state vector
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> init_cell_vector; //Initial cell vector for LSTM

	//initial backprop errors to pass
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> init_d_ERRnTOtp1_ht; 
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> init_d_ERRnTOtp1_ct;

	////////////////////////////////////////////Other parameters////////////////////////////////////////////
	boost::random::mt19937 gen; //Random number generator for initializing weights

	struct neuralMT_model<precision> *model;

	//True if want debugging printout,false otherwise
	bool debug;
	int minibatch_size;
	double learning_rate;
	bool clip_gradients;
	double norm_clip; //For gradient clipping


	///////////////////////////////////////////Function Declarations///////////////////////////////

	//Constructor
	void init_Hidden_To_Hidden_Layer(int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,double learning_rate,bool clip_gradients,double norm_clip,
 		struct neuralMT_model<precision> *model);

	//Clear the previous gradients
	void clear_gradients();

	//Update the weights of the model
	void update_weights();

	template<typename Derived2>
	void check_all_gradients(double epsilon,
		const Eigen::MatrixBase<Derived2> &input_minibatch_const,
		const Eigen::MatrixBase<Derived2> &output_minibatch_const);
	
	void dump_weights(std::ofstream &output);

	void load_weights(std::ifstream &input);

	template<typename Derived>
	void initMatrix(const Eigen::MatrixBase<Derived> &input_const);

	template<typename Derived,typename Derived2,typename Derived3>
	void check_gradient(double epsilon,
		const Eigen::MatrixBase<Derived3> &parameter_const,const Eigen::MatrixBase<Derived> &grad,
		const Eigen::MatrixBase<Derived2> &input_minibatch_const,
		const Eigen::MatrixBase<Derived2> &output_minibatch_const);

};


#endif