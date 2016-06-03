//LSTM layer that connects input to hidden
#ifndef LSTM_INPUT_TO_HIDDEN_H
#define LSTM_INPUT_TO_HIDDEN_H

template<typename dType>
class neuralMT_model;

#include "transfer_layer.h"

template<typename dType>
class Input_To_Hidden_Layer {
public:
	//Parameters for the model
	//The parameters need to connect input to input gate
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> W;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_i;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_f;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_o;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_c;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hi;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_i;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hf;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_f;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_hc;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_c;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> W_ho;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_o;

	/////////////////////////////////Stores the gradients for the models/////////////////////////////////
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> W_hi_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_i_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> W_hf_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_f_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> W_hc_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_c_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> W_ho_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_o_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> W_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_i_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_f_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_o_grad;
//	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> M_c_grad;

	/////////////////////////////////Current minibatch info for the model///////////////////////////////////
	std::vector<LSTM_IH_Node<dType>> nodes; //Stores all the LSTM nodes for forward and backward propagation
//	Eigen::Matrix<dType,Eigen::Dynamic,Eigen::Dynamic> init_hidden_vector; //Initial hidden state vector
//	Eigen::Matrix<dType,Eigen::Dynamic,Eigen::Dynamic> init_cell_vector; //Initial cell vector for LSTM
//	Eigen::Matrix<dType,Eigen::Dynamic,Eigen::Dynamic> init_d_ERRnTOtp1_ht; 
//	Eigen::Matrix<dType,Eigen::Dynamic,Eigen::Dynamic> init_d_ERRnTOtp1_ct;




	//---------------------------------------------GPU parameters---------------------------------------------

	layer_gpu_info ih_layer_info;
	
	//host pointers
	dType *h_temp1;
	dType *h_temp2;
	dType *h_temp3;
	dType *h_temp4;

	dType *h_W_ho;
	dType *h_W_hf;
	dType *h_W_hi;
	dType *h_W_hc;

	dType *h_W_hi_grad;
	dType *h_W_hf_grad;
	dType *h_W_hc_grad;
	dType *h_W_ho_grad;


	dType *h_M_i_grad;
	dType *h_M_f_grad;
	dType *h_M_o_grad;
	dType *h_M_c_grad;

	dType *h_W;

	dType *h_b_i_grad;
	dType *h_b_f_grad;
	dType *h_b_c_grad;
	dType *h_b_o_grad;

	dType *h_ones_minibatch;

	dType *h_M_i;
	dType *h_M_f;
	dType *h_M_o;
	dType *h_M_c;

	dType *h_W_grad;

	dType *h_b_i;
	dType *h_b_f;
	dType *h_b_c;
	dType *h_b_o;

	dType *h_temp5;
	dType *h_temp6;

	dType *h_temp7;
	dType *h_temp8;

	//Convert this into 0/1's and to one with no -1's as indicies
	int *h_input_vocab_indicies;
	int *d_input_vocab_indicies; 
	int current_length; //This is the current length of this target or source sequence
	int w_grad_len; //This is special length for the W_grad special preprocessing for vocab indicies

	 //contains the entire input sequence, use pointer arithmetic to pass correct segments to LSTM cells
	int *h_input_vocab_indices_full; //only for debugging
	int *h_input_vocab_indices_01_full; //only for debugging
	int *h_input_vocab_indicies_Wgrad;
	int *d_input_vocab_indices_full;
	int *d_input_vocab_indices_01_full;
	int *d_input_vocab_indicies_Wgrad;

	//for setting inital cell and hidden state values
	dType *h_init_hidden_vector;
	dType *h_init_cell_vector;
	dType *d_init_hidden_vector;
	dType *d_init_cell_vector;

	dType *h_init_d_ERRnTOtp1_ht;
	dType *h_init_d_ERRnTOtp1_ct;
	dType *d_init_d_ERRnTOtp1_ht;
	dType *d_init_d_ERRnTOtp1_ct;

	//pass this in for backprop gpu prep from source size (all zero error matrix)
	dType *d_zeros;

	//stuff for norm clipping
	dType *d_result;
	dType *d_temp_result;


	//device pointers
	dType *d_temp1;
	dType *d_temp2;
	dType *d_temp3;
	dType *d_temp4;

	dType *d_W_ho;
	dType *d_W_hf;
	dType *d_W_hi;
	dType *d_W_hc;

	dType *d_W_hi_grad;
	dType *d_W_hf_grad;
	dType *d_W_hc_grad;
	dType *d_W_ho_grad;

	dType *d_M_i_grad;
	dType *d_M_f_grad;
	dType *d_M_o_grad;
	dType *d_M_c_grad;

	dType *d_W;

	dType *d_b_i_grad;
	dType *d_b_f_grad;
	dType *d_b_c_grad;
	dType *d_b_o_grad;

	dType *d_ones_minibatch;

	dType *d_M_i;
	dType *d_M_f;
	dType *d_M_o;
	dType *d_M_c;

	//dType *d_W_grad;


	dType *d_small_W_grad;
	thrust::device_ptr<dType> thrust_d_small_W_grad;
	int *d_reverse_unique_indicies;


	dType *d_b_i;
	dType *d_b_f;
	dType *d_b_c;
	dType *d_b_o;

	dType *d_temp5;
	dType *d_temp6;

	dType *d_temp7;
	dType *d_temp8;

	dType *d_temp9;
	dType *d_temp10;
	dType *d_temp11;
	dType *d_temp12;

	//these are for the feed input connections
	dType *d_Q_i;
	dType *d_Q_f;
	dType *d_Q_o;
	dType *d_Q_c;
	dType *d_Q_i_grad;
	dType *d_Q_f_grad;
	dType *d_Q_o_grad;
	dType *d_Q_c_grad;


	//new for saving space in the LSTM
	dType *h_d_ERRnTOt_ht;
	dType *h_d_ERRt_ct;
	dType *h_d_ERRnTOt_ct;
	dType *h_d_ERRnTOt_ot;
	dType *h_d_ERRnTOt_ft;
	dType *h_d_ERRnTOt_tanhcpt;
	dType *h_d_ERRnTOt_it;
	dType *h_d_ERRnTOt_htM1;
	dType *h_d_ERRnTOt_ctM1;

	dType *d_d_ERRnTOt_ht;
	dType *d_d_ERRt_ct;
	dType *d_d_ERRnTOt_ct;
	dType *d_d_ERRnTOt_ot;
	dType *d_d_ERRnTOt_ft;
	dType *d_d_ERRnTOt_tanhcpt;
	dType *d_d_ERRnTOt_it;
	dType *d_d_ERRnTOt_htM1;
	dType *d_d_ERRnTOt_ctM1;

	dType *d_conv_char_error;

	//thrust device pointers to doing parameter updates nicely (not input word embeddings though)
	thrust::device_ptr<dType> thrust_d_W_ho_grad; 
	thrust::device_ptr<dType> thrust_d_W_hf_grad;
	thrust::device_ptr<dType> thrust_d_W_hi_grad; 
	thrust::device_ptr<dType> thrust_d_W_hc_grad;

	thrust::device_ptr<dType> thrust_d_M_i_grad;
	thrust::device_ptr<dType> thrust_d_M_f_grad;
	thrust::device_ptr<dType> thrust_d_M_o_grad;
	thrust::device_ptr<dType> thrust_d_M_c_grad;

	thrust::device_ptr<dType> thrust_d_Q_i_grad;
	thrust::device_ptr<dType> thrust_d_Q_f_grad;
	thrust::device_ptr<dType> thrust_d_Q_o_grad;
	thrust::device_ptr<dType> thrust_d_Q_c_grad;

	//remove then put in custom reduction kernel
	thrust::device_ptr<dType> thrust_d_W_grad;

	thrust::device_ptr<dType> thrust_d_b_i_grad;
	thrust::device_ptr<dType> thrust_d_b_f_grad;
	thrust::device_ptr<dType> thrust_d_b_c_grad;
	thrust::device_ptr<dType> thrust_d_b_o_grad;


	//Decoder stuff
	Eigen::Matrix<dType,Eigen::Dynamic,Eigen::Dynamic> temp_swap_vals; //used for changing hidden and cell state columns

	////////////////////////////////////////////Other parameters////////////////////////////////////////////
	boost::random::mt19937 gen; //Random number generator for initializing weights

	neuralMT_model<precision> *model;

	//True if want debugging printout,false otherwise
	bool debug;
	int minibatch_size;
	dType learning_rate;
	bool clip_gradients;
	dType norm_clip; //For gradient clipping
	int LSTM_size;
	int longest_sent;
	int input_vocab_size;
	attention_layer<dType> *attent_layer=NULL;
	bool feed_input = false;

	bool multi_source_attention = false;
	attention_layer<dType> *attent_layer_bi=NULL; //for multi source stuff
	attention_combiner_layer<dType> *att_comb_layer=NULL;
	conv_char_layer<dType> *char_cnn_layer = NULL;
	bool char_cnn = false;

	bool bi_dir = false;
	bool nonrev_bi_dir = false; //This will only be true if using combine bi-dir and this is the nonrev encoder
	bool share_embeddings = false;
	bool combine_embeddings = false; //this is true for the nonrev encoder in the bi-directional model
	
	//for dropout
	bool dropout;
	dType dropout_rate;
	curandGenerator_t rand_gen;

	//for gpu to gpu transfers
	upper_transfer_layer<dType> upper_layer;

	///////////////////////////////////////////Function Declarations///////////////////////////////
	Input_To_Hidden_Layer() {};

	void check_gradient_GPU_SPARSE(dType epsilon,dType *d_mat,dType *d_grad,int LSTM_size,int *h_unique_indicies,int curr_num_unique);

	//Constructor
	void init_Input_To_Hidden_Layer(int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,struct neuralMT_model<precision> *model,int seed,
 		bool dropout,dType dropout_rate,bool is_bi_dir,bool share_embeddings,dType *d_embedding_ptr,bool combine_embeddings,
 		global_params &params, bool source);

	void init_Input_To_Hidden_Layer_GPU(int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,struct neuralMT_model<precision> *model,int seed,
 		bool share_embeddings,dType *d_embedding_ptr,bool combine_embeddings,global_params &params,bool source);

	//Clear the previous gradients
	void clear_gradients(bool init);
	void clear_gradients_GPU(bool init);

	//Update the weights of the model
	void update_weights();
	void update_weights_GPU();

	void calculate_global_norm();
	void update_global_params();

	void check_all_gradients(dType epsilon);
	void check_all_gradients_GPU(dType epsilon);
	
	void dump_weights(std::ofstream &output);
	void dump_weights_GPU(std::ofstream &output);

	void load_weights(std::ifstream &input);
	void load_weights_GPU(std::ifstream &input);

	void prep_char_cnn(int *h_vocab_indicies_full,int curr_sent_len,int *h_unique_chars_minibatch,int num_unique_chars_minibatch);

	template<typename Derived,typename Derived3>
	void check_gradient(dType epsilon,const Eigen::MatrixBase<Derived3> &parameter_const,const Eigen::MatrixBase<Derived> &grad);

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
	
	//convert to 0/1's and to indicies where there are no -1's
	void prep_GPU_vocab_indices(int *h_input_vocab_indicies,int *h_input_vocab_indicies_Wgrad,int current_length,int len_W);

	//swap the states during the decoding process
	//index specifies which node to swap at
	template<typename Derived>
	void swap_states_decoding(const Eigen::MatrixBase<Derived> &indicies,int index,dType *d_temp_swap_vals);

	//This transfers the single column source vector and replicates it for the decoding
	template<typename Derived>
	void transfer_decoding_states(const Eigen::MatrixBase<Derived> &s_h_t,const Eigen::MatrixBase<Derived> &s_c_t);

	void transfer_decoding_states_GPU(dType *d_h_t,dType *d_c_t);

	void init_attention(int device_number,int D,bool feed_input,neuralMT_model<dType> *model,global_params &params);

	void zero_attent_error();

	void init_feed_input(Hidden_To_Hidden_Layer<dType> *hidden_layer,bool multi_attention);

	void scale_gradients();

	void update_params();

	void decoder_init_feed_input();

	void load_weights_decoder_feed_input(std::ifstream &input);

	void load_weights_charCNN(std::ifstream &input);
};


#endif
