
#ifndef NCE_NODE_H
#define NCE_NODE_H

template<typename dType>
struct NCE_Node {

	//each node stores the unnormalized probabilities, plus the h_t
	dType *d_h_t;
	dType *d_p_true;
	dType *d_temp_embeddings;
	dType *d_d_ERRt_ht;
	dType *d_dropout_mask;
	int index;

	NCE_Node(int LSTM_size,int minibatch_size,int num_negative_samples,int index,bool dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_embeddings, (num_negative_samples + minibatch_size)*LSTM_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_p_true, (num_negative_samples + minibatch_size)*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_d_ERRt_ht, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		this->index = index;
		if(dropout) {
			CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		}
	}
};


#endif