#ifndef SOFTMAX_NODE_H
#define SOFTMAX_NODE_H

//for multigpu training
template<typename dType>
struct softmax_node {

	//each node stores the unnormalized probabilities, plus the h_t
	dType *d_outputdist;
	dType *d_h_t;
	dType *d_d_ERRt_ht;
	dType *d_dropout_mask;
	int index;

	softmax_node(int LSTM_size,int minibatch_size,int output_vocab_size,int index,bool dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist, output_vocab_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_d_ERRt_ht, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		this->index = index;
		if(dropout) {
			CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		}
	}
};

#endif