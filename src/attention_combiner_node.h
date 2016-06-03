#ifndef ATT_COMBINER_NODE_H
#define ATT_COMBINER_NODE_H

template<typename dType>
class attention_combiner_layer;

template<typename dType>
class attention_combiner_node {
public:

	int LSTM_size;
	int minibatch_size;
	int index;

	dType *d_ht_1;
	dType *d_ht_2;
	dType *d_ht_final;

	dType *d_h_tild; //location of feed input to copy

	bool add_ht = false;

	dType *d_ERR_ht_1; //final error getting added before being sent to LSTM
	dType *d_ERR_ht_2; //final error getting added before being sent to LSTM
	dType *d_ERR_ht_top_loss; //error coming from loss, also contains total after first add
	dType *d_ERR_ht_top_feed; //error coming feed_input, gets copied into from lowest LSTM

	int **d_indicies_mask; //for zeroing out errors

	dType *d_ones_minibatch;

	attention_combiner_layer<dType> *model;

	attention_combiner_node(global_params &params,attention_combiner_layer<dType> *model,int index);

	void forward();
	void backward();
};





#endif
