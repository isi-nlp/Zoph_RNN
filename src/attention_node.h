#ifndef ATTENTION_NODE_H
#define ATTENTION_NODE_H


template<typename dType>
class attention_layer;


template<typename dType>
class attention_node {
public:

	attention_layer<dType> *attent_layer;
	int LSTM_size;
	int minibatch_size;
	int device_number;
	int D; //alignment width
	bool dropout;
	dType dropout_rate;
	dType *d_dropout_mask;

	bool multi_attention = false;
	bool multi_attention_v2 = false;

	dType *d_tanh_1;
	dType *d_sigma_1;
	dType *d_p_t; // size (1,minibatch size)
	dType *d_alignments; // size (minibatch size x 2D + 1)
	dType *d_h_t=NULL;
	dType *d_c_t; // size (LSTM size, minibatch size)
	dType *d_exped_scored; //multiply these witha binary mask, so if alignments go off edge then just set to zero
	dType *d_final_temp_1; //for W_c_p1*c_t
	dType *d_final_temp_2; //for W_c_p2*h_t, also reuse to add the bias and tanh
	int *d_lower_upper;
	int *d_indicies;
	dType sigma_sq;
	dType *d_h_t_att;

	//for attention_model_v2
	dType *d_tanh_1_v2;
	dType *d_sigma_1_v2;
	dType *d_p_t_v2; // size (1,minibatch size)
	dType *d_alignments_v2; // size (minibatch size x 2D + 1)
	dType *d_c_t_v2; // size (LSTM size, minibatch size)
	dType *d_exped_scored_v2; //multiply these witha binary mask, so if alignments go off edge then just set to zero
	int *d_lower_upper_v2;
	int *d_indicies_v2;
	dType *d_h_t_Wa_cache_v2; //precompute h_t multiplied by W_a
	dType *d_hs_mat_v2;
	dType *d_cached_exp_v2;


	int **d_indicies_mask; //points to the LSTM node for this info for zeroing out forward and back prop

	dType *d_cached_exp; //stores the exp(- (s-pt)^2 ...) , size is 2*D+1 by minibatch size

	dType *d_h_t_Wa_cache; //precompute h_t multiplied by W_a

	dType *d_hs_mat;
	dType *d_d_ERRt_ht_tild; //this is the error passed back from the softmax
	dType *d_d_ERRt_ht_input; //if feed input, then this error will be added in place to d_d_ERRt_ht_p
	dType *d_ERR_above; //what the LSTM get passed from the above layer or softmax

	dType *d_lower_htild; //send htild to this location
	dType *d_ERRtTOn_htild_below; //this is from the previous lower LSTM for feed input

	bool feed_input = false; //get rid of most parallelism :( 

	int index;

	attention_node(int LSTM_size,int minibatch_size,int device_number,int D,bool feed_input,attention_layer<dType> *attent_layer,int index,
		bool dropout,dType dropout_rate,bool multi_attention,bool multi_attention_v2);

	void forward_prop();

	void back_prop();

	void feed_input_init(dType *d_ptr_htild);

	void debug_func();

	void debug_checker();

};




#endif