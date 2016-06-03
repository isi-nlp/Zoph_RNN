#ifndef CHARCNN_NODE_H
#define CHARCNN_NODE_H

//charCNN node

template<typename dType>
struct charCNN_node {

	dType *d_output_conv;
	dType *d_output_pooling;
	dType *d_C;
	cudnnTensorDescriptor_t tensor_output_conv; //output from character convolution before max pooling
	cudnnTensorDescriptor_t tensor_output_pooling; //output from max pooling
	cudnnTensorDescriptor_t tensor_C; //character embeddings tensor descriptor

	charCNN_node(const cudnnTensorFormat_t &cudnn_tensor_format,cudnnDataType_t &cudnn_dtype,
		int minibatch_size,int num_filters,int longest_word,int filter_size,int char_emb_size) 
	{

		dType *h_temp;
		full_matrix_setup(&h_temp,&d_output_conv,num_filters*(longest_word - filter_size + 1),minibatch_size); //this is actually a tensor
		full_matrix_setup(&h_temp,&d_output_pooling,num_filters,minibatch_size); //this is actually a tensor
		full_matrix_setup(&h_temp,&d_C,char_emb_size,longest_word*minibatch_size); //this is actually a tensor


		//allocation for tensor for output_conv
		checkCUDNN(cudnnCreateTensorDescriptor(&tensor_output_conv));
		checkCUDNN(cudnnSetTensor4dDescriptor( tensor_output_conv,
	    	cudnn_tensor_format,
	        cudnn_dtype,
	        minibatch_size,  //n
	        num_filters,  //c
			longest_word - filter_size + 1,  //h
			1 ));  //w


		//allocation for tensor for output of pooling
		checkCUDNN(cudnnCreateTensorDescriptor(&tensor_output_pooling));
		checkCUDNN(cudnnSetTensor4dDescriptor( tensor_output_pooling,
	    	cudnn_tensor_format,
	        cudnn_dtype,
	        minibatch_size,  //n
	        num_filters,  //c
			1,  //h
			1 ));  //w

		//allocation for tensor C
		checkCUDNN(cudnnCreateTensorDescriptor(&tensor_C));
		checkCUDNN(cudnnSetTensor4dDescriptor( tensor_C,
	    	cudnn_tensor_format,
	        cudnn_dtype,
	        minibatch_size,  //n
	        1,  //c
			longest_word,  //h
			char_emb_size ));  //w
	}
};






#endif