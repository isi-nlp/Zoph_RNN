#ifndef CONV_CHAR_H
#define CONV_CHAR_H

template<typename dType>
class neuralMT_model;

#include "highway_network.h"
#include "charCNN_node.h"

/*
	-the conv operation and all highway layers are all on the same GPU and share the same cublas handle and stream
	
*/
template<typename dType>
class conv_char_layer {
public:

	//model info
	int longest_word;
	int char_emb_size;
	//int word_emb_size = 100;
	int minibatch_size;
	int num_unique_chars;
	int filter_size;
	int num_filters;
	int longest_sent;
	dType norm_clip;

	int num_highway_networks;
	std::vector<highway_network_layer<dType>*> highway_layers;


	//streams and events
	cublasHandle_t handle;
	cudaStream_t s0; //everything goes on this stream
	cudaEvent_t forward_prop_start;
	cudaEvent_t forward_prop_done;
	cudaEvent_t back_prop_start;

	//gpu info
	int device_number = 0;
	cudnnHandle_t cudnnHandle;
	cudnnDataType_t cudnn_dtype; //datatype for cudnn
	const cudnnTensorFormat_t cudnn_tensor_format = CUDNN_TENSOR_NCHW; //type of tensor

	neuralMT_model<dType> *model;

	//params
	dType *d_Q; //character embeddings
	dType *d_C; //character embeddings for specific word memory
	dType *d_H; //storage for filter
	// dType *d_output_conv;
	// dType *d_output_pooling;
	dType *d_b; //bias for convolution

	dType *d_output_conv_err;
	dType *d_output_pooling_err;
	dType *d_H_grad; //storage for filter gradient
	dType *d_C_err; //error for character embeddings
	dType *d_Q_grad; //character embeddings
	dType *d_b_grad; //gradient of bias for convolution

	thrust::device_ptr<dType> thrust_d_H_grad;
	thrust::device_ptr<dType> thrust_d_Q_grad;
	thrust::device_ptr<dType> thrust_d_b_grad;

	dType *d_result;
	dType *d_temp_result;

	highway_network_layer<dType>* top_highway_layer;

	int curr_sent_len; //what is the current longest sentence for this minibatch
	int *d_vocab_indicies_full;
	int *d_vocab_indicies;
	int num_unique_chars_minibatch;
	int *d_unique_chars_minibatch;

	//cudnnTensorDescriptor_t tensor_C; //character embeddings tensor descriptor
	cudnnTensorDescriptor_t tensor_b; //bias
	// cudnnTensorDescriptor_t tensor_output_conv; //output from character convolution before max pooling
	// cudnnTensorDescriptor_t tensor_output_pooling; //output from max pooling
	cudnnFilterDescriptor_t filter_H; //cudnn filter for going over character embeddings

	cudnnTensorDescriptor_t tensor_output_conv_err; //output from character convolution before max pooling
	cudnnTensorDescriptor_t tensor_output_pooling_err; //output from max pooling
	cudnnFilterDescriptor_t filter_H_grad; //grad for filters
	cudnnTensorDescriptor_t tensor_C_grad; //grad for filters
	cudnnTensorDescriptor_t tensor_b_grad; //bias

	dType *d_workspace_conv_forward; //for cudnn algorithms
	size_t workspace_conv_forward_size;

	dType *d_workspace_conv_backward_data; //for cudnn algorithms
	size_t workspace_conv_backward_data_size;
	dType *d_workspace_conv_backward_filter; //for cudnn algorithms
	size_t workspace_conv_backward_filter_size;

	cudnnConvolutionFwdAlgo_t cudnn_conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	cudnnConvolutionBwdDataAlgo_t cudnn_conv_back_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	cudnnConvolutionBwdFilterAlgo_t cudnn_conv_back_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	cudnnPoolingDescriptor_t cudnn_poolingDesc; //for max pooling

	cudnnConvolutionDescriptor_t cudnn_conv_info;

	std::vector<charCNN_node<dType> *> nodes;

	bool decode_source = false;
	bool decode_target = false;
	int curr_decode_step = 0;

	void init(global_params &params,int device_number,cudaEvent_t &forward_prop_start,
		neuralMT_model<dType> *model,int num_unique_chars);
	void fill_char_embeddings();
	void forward(int index);
	void backward(int index);
	void clear_gradients();
	void check_gradients(dType epsilon);
	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
	void prep_vocab_indicies(int *h_vocab_indicies_full,int curr_sent_len,
		int *h_unique_chars_minibatch,int num_unique_chars_minibatch);
	void norm_p1();
	void norm_p2();
	void scale_gradients();
	void update_params();
	void clip_gradients_func();
	void dump_weights(std::ofstream &output);
	void load_weights(std::ifstream &input);
};


#endif
