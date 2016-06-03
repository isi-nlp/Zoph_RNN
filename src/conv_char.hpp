template<typename dType>
void conv_char_layer<dType>::prep_vocab_indicies(int *h_vocab_indicies_full,int curr_sent_len,
	int *h_unique_chars_minibatch,int num_unique_chars_minibatch) 
{
	devSynchAll();
	CUDA_GET_LAST_ERROR("POST INDICES SETUP GPU CNN PRIOR");
	cudaSetDevice(device_number);
	this->curr_sent_len = curr_sent_len;
	this->num_unique_chars_minibatch = num_unique_chars_minibatch;

	curr_decode_step = 0; //for decoding
	// std::cout << "***************   In charCNN prep vocab indicies   ***************\n";
	// // std::cout << "curr_sent_len: "  << curr_sent_len << "\n";
	// // std::cout << "num_unique_chars_minibatch: "  << num_unique_chars_minibatch << "\n";
	// std::cout << "longest_word: " << longest_word << "\n";
	// std::cout << "minibatch_size: " << minibatch_size << "\n";

	// for(int i=0; i<num_unique_chars_minibatch; i++) {
	// 	std::cout << h_unique_chars_minibatch[i] << " ";
	// }
	//std::cout << "---------------------- ENTERING FOR MINIBATCH ------------------------\n";
	// int word_counter = 0;
	// for(int i=0; i<longest_word*minibatch_size*curr_sent_len; i++) {
	// 	if(word_counter%longest_word==0) {
	// 		std::cout << "\n\n";
	// 	}
	// 	std::cout << h_vocab_indicies_full[i] << " ";
	// 	word_counter++;
	// }
	//std::cout << "\nTotal words in minibatch: " << word_counter << "\n";
	//std::cout << "current longest sent: " << curr_sent_len << "\n";
	// std::cout << "\n";
	cudaMemcpy(d_vocab_indicies_full,h_vocab_indicies_full,longest_word*minibatch_size*curr_sent_len*sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_unique_chars_minibatch,h_unique_chars_minibatch,num_unique_chars_minibatch*sizeof(int), cudaMemcpyHostToDevice);
	devSynchAll();
	CUDA_GET_LAST_ERROR("POST INDICES SETUP GPU CNN POST");
}

template<typename dType>
void conv_char_layer<dType>::clear_gradients() {

	cudaSetDevice(device_number);

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->clear_gradients();
	}

	cudaMemset(d_H_grad,0,char_emb_size*num_filters*filter_size*sizeof(dType));
	cudaMemset(d_Q_grad,0,char_emb_size*num_unique_chars*sizeof(dType));
	cudaMemset(d_b_grad,0,num_filters*sizeof(dType));
}

template<typename dType>
void conv_char_layer<dType>::norm_p1() {

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->norm_p1();
	}
	
	norm_clip_GPU_v2_p1(thrust_d_Q_grad,d_Q_grad,norm_clip,char_emb_size*num_unique_chars,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_H_grad,d_H_grad,norm_clip,char_emb_size*num_filters*filter_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_grad,d_b_grad,norm_clip,num_filters,d_temp_result,d_result);
}


template<typename dType>
void conv_char_layer<dType>::norm_p2() {

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->norm_p2();
	}
	
	norm_clip_GPU_v2_p2(thrust_d_Q_grad,d_Q_grad,norm_clip,char_emb_size*num_unique_chars,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_H_grad,d_H_grad,norm_clip,char_emb_size*num_filters*filter_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_grad,d_b_grad,norm_clip,num_filters,d_temp_result,d_result);
}

template<typename dType>
void conv_char_layer<dType>::scale_gradients() {

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->scale_gradients();
	}
	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_Q_grad,thrust_d_Q_grad + char_emb_size*num_unique_chars,unary_op);
	thrust::for_each(thrust_d_H_grad,thrust_d_H_grad + char_emb_size*num_filters*filter_size,unary_op);
	thrust::for_each(thrust_d_b_grad,thrust_d_b_grad + num_filters,unary_op);
}


template<typename dType>
void conv_char_layer<dType>::update_params() {

	if( (deniz::source_side && deniz::train_source_RNN) || (!deniz::source_side && deniz::train_target_RNN) ) {
		for(int i=0; i<highway_layers.size(); i++) {
			highway_layers[i]->update_params();
		}
	}

	if( (deniz::source_side && deniz::train_source_input_embedding) || (!deniz::source_side && deniz::train_target_input_embedding) ) {
		gradient_update_mats<<<256,256,0,s0>>>(d_Q,d_Q_grad,model->input_layer_target.learning_rate,char_emb_size*num_unique_chars);
	}

	if( (deniz::source_side && deniz::train_source_RNN) || (!deniz::source_side && deniz::train_target_RNN) ) {
		gradient_update_mats<<<256,256,0,s0>>>(d_H,d_H_grad,model->input_layer_target.learning_rate,char_emb_size*num_filters*filter_size);
		gradient_update_mats<<<256,256,0,s0>>>(d_b,d_b_grad,model->input_layer_target.learning_rate,num_filters);
	}

	devSynchAll();
}


template<typename dType>
void conv_char_layer<dType>::clip_gradients_func() {

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->clip_gradients_func();
	}
	
	norm_clip_GPU_v2(thrust_d_Q_grad,d_Q_grad,norm_clip,char_emb_size*num_unique_chars,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_H_grad,d_H_grad,norm_clip,char_emb_size*num_filters*filter_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_b_grad,d_b_grad,norm_clip,num_filters,d_temp_result,d_result);
}

template<typename dType>
void conv_char_layer<dType>::dump_weights(std::ofstream &output) {

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->dump_weights(output);
	}

	write_matrix_GPU(d_Q,char_emb_size,num_unique_chars,output);
	write_matrix_GPU(d_H,char_emb_size,num_filters*filter_size,output);
	write_matrix_GPU(d_b,num_filters,1,output);
}

template<typename dType>
void conv_char_layer<dType>::load_weights(std::ifstream &input) {

	for(int i=0; i<highway_layers.size(); i++) {
		highway_layers[i]->load_weights(input);
	}

	// std::cout << "LOADING WEIGHTS in CONV CHAR\n";
	// std::cout << "char_emb_size: " << char_emb_size << "\n";
	// std::cout << "num_unique_chars: " << num_unique_chars << "\n";
	// std::cout << "num_filters: " << num_filters << "\n";

	read_matrix_GPU(d_Q,char_emb_size,num_unique_chars,input);
	read_matrix_GPU(d_H,char_emb_size,num_filters*filter_size,input);
	read_matrix_GPU(d_b,num_filters,1,input);
}





template<typename dType>
void conv_char_layer<dType>::init(global_params &params,int device_number,cudaEvent_t &forward_prop_start,
	neuralMT_model<dType> *model,int num_unique_chars) {


	this->device_number = device_number;
	this->minibatch_size = params.minibatch_size;
	this->longest_word = params.char_params.longest_word;
	this->char_emb_size = params.char_params.char_emb_size;
	this->num_unique_chars = num_unique_chars;
	this->filter_size = params.char_params.filter_size;
	this->num_filters = params.LSTM_size;
	this->longest_sent = params.longest_sent;
	this->num_highway_networks = params.char_params.num_highway_layers;
	this->forward_prop_start = forward_prop_start;
	this->model = model;
	this->norm_clip = params.norm_clip;

	// std::cout << "INIT: longest_word " << longest_word << "\n";
	// std::cout << "INIT: minibatch_size " << minibatch_size << "\n";
	// std::cout << "INIT: longest_sent " << longest_sent << "\n";
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies_full, longest_word*minibatch_size*longest_sent*sizeof(int)),"GPU memory allocation failed\n");


	if(decode_source) {
		longest_sent = 1;
		minibatch_size = 1;
	}

	if(decode_target) {
		longest_sent = 1;
		minibatch_size = params.beam_size;
	}

	cudaSetDevice(device_number);

	CUBLAS_ERROR_WRAPPER(cublasCreate(&handle),"CUBLAS handler initialization failed\n");
	cudaStreamCreate(&s0);
	cudaEventCreate(&forward_prop_done);
	cudaEventCreate(&back_prop_start);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");

	//create cuDNN handle
	checkCUDNN(cudnnCreate(&cudnnHandle));

	//set datatype
	if(sizeof(dType) == sizeof(float)) {
		cudnn_dtype = CUDNN_DATA_FLOAT;
	}
	else {
		cudnn_dtype = CUDNN_DATA_DOUBLE;
	}	

	BZ_CUDA::logger << "cuDNN version being used: " << (double)cudnnGetVersion() << "\n";
	BZ_CUDA::logger << "----------------------------   Printing charCNN stats   ----------------------------\n";
	BZ_CUDA::logger << "device number: " << device_number << "\n";
	BZ_CUDA::logger << "minibatch_size: " << minibatch_size << "\n";
	BZ_CUDA::logger << "longest_word: " << longest_word << "\n";
	BZ_CUDA::logger << "char_emb_size: " << char_emb_size << "\n";
	BZ_CUDA::logger << "num_unique_chars: " << num_unique_chars << "\n";
	BZ_CUDA::logger << "filter_size: " << filter_size << "\n";
	BZ_CUDA::logger << "num_filters: " << num_filters << "\n";
	BZ_CUDA::logger << "longest_sent: " << longest_sent << "\n";
	BZ_CUDA::logger << "num_highway_networks: " << num_highway_networks << "\n";
	BZ_CUDA::logger << "-----------------------------------------------------------------------------------\n";

	for(int i=0; i<longest_sent; i++) {
		nodes.push_back( new charCNN_node<dType>(cudnn_tensor_format,cudnn_dtype,
			minibatch_size,num_filters,longest_word,filter_size,char_emb_size) );
	}

	//set up cudnn convolution info
	checkCUDNN(cudnnCreateConvolutionDescriptor(&cudnn_conv_info));
	checkCUDNN(cudnnSetConvolution2dDescriptor( cudnn_conv_info,
		0, //pad_h
		0, //pad_w
		1, //u
		1, //v
		1, //upscalex
		1, //upscaley
		CUDNN_CONVOLUTION ));

	//set up highway networks
	for(int i = 0; i<num_highway_networks; i++) {
		highway_layers.push_back( new highway_network_layer<dType>() );
		highway_layers[i]->init(num_filters,minibatch_size,longest_sent,
			device_number,handle,s0,model,norm_clip);
	}

	top_highway_layer = highway_layers[highway_layers.size()-1];

	//Allocate Q embedding
	dType *h_temp;
	full_matrix_setup(&h_temp,&d_Q,char_emb_size,num_unique_chars);
	//full_matrix_setup(&h_temp,&d_C,char_emb_size,longest_word*minibatch_size); //this is actually a tensor
	full_matrix_setup(&h_temp,&d_H,char_emb_size,num_filters*filter_size); //this is actually a tensor
	//full_matrix_setup(&h_temp,&d_output_conv,num_filters*(longest_word - filter_size + 1),minibatch_size); //this is actually a tensor
	full_matrix_setup(&h_temp,&d_b,num_filters,1);
	//full_matrix_setup(&h_temp,&d_output_pooling,num_filters,minibatch_size); //this is actually a tensor
	
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_unique_chars_minibatch, num_unique_chars*sizeof(int)),"GPU memory allocation failed\n");

	full_matrix_setup(&h_temp,&d_Q_grad,char_emb_size,num_unique_chars);
	full_matrix_setup(&h_temp,&d_C_err,char_emb_size,longest_word*minibatch_size); //this is actually a tensor
	full_matrix_setup(&h_temp,&d_H_grad,char_emb_size,num_filters*filter_size); //this is actually a tensor
	full_matrix_setup(&h_temp,&d_output_conv_err,num_filters*(longest_word - filter_size + 1),minibatch_size); //this is actually a tensor
	full_matrix_setup(&h_temp,&d_b_grad,num_filters,1);
	full_matrix_setup(&h_temp,&d_output_pooling_err,num_filters,minibatch_size); //this is actually a tensor

	// std::cout << "INIT: longest_word " << longest_word << "\n";
	// std::cout << "INIT: minibatch_size " << minibatch_size << "\n";
	// std::cout << "INIT: longest_sent " << longest_sent << "\n";
	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies_full, longest_word*minibatch_size*longest_sent*sizeof(int)),"GPU memory allocation failed\n");

	// //allocation for tensor C
	// checkCUDNN(cudnnCreateTensorDescriptor(&tensor_C));
	// checkCUDNN(cudnnSetTensor4dDescriptor( tensor_C,
 //    	cudnn_tensor_format,
 //        cudnn_dtype,
 //        minibatch_size,  //n
 //        1,  //c
	// 	longest_word,  //h
	// 	char_emb_size ));  //w

	//allocate tensor for the bias
	checkCUDNN(cudnnCreateTensorDescriptor(&tensor_b));
	checkCUDNN(cudnnSetTensor4dDescriptor( tensor_b,
    	cudnn_tensor_format,
        cudnn_dtype,
        1,  //n
        num_filters,  //c
		1,  //h
		1 ));  //w

	// //allocation for tensor for output_conv
	// checkCUDNN(cudnnCreateTensorDescriptor(&tensor_output_conv));
	// checkCUDNN(cudnnSetTensor4dDescriptor( tensor_output_conv,
 //    	cudnn_tensor_format,
 //        cudnn_dtype,
 //        minibatch_size,  //n
 //        num_filters,  //c
	// 	longest_word - filter_size + 1,  //h
	// 	1 ));  //w


	// //allocation for tensor for output of pooling
	// checkCUDNN(cudnnCreateTensorDescriptor(&tensor_output_pooling));
	// checkCUDNN(cudnnSetTensor4dDescriptor( tensor_output_pooling,
 //    	cudnn_tensor_format,
 //        cudnn_dtype,
 //        minibatch_size,  //n
 //        num_filters,  //c
	// 	1,  //h
	// 	1 ));  //w


	//allocation for tensor for output of pooling gradient
	checkCUDNN(cudnnCreateTensorDescriptor(&tensor_output_pooling_err));
	checkCUDNN(cudnnSetTensor4dDescriptor( tensor_output_pooling_err,
    	cudnn_tensor_format,
        cudnn_dtype,
        minibatch_size,  //n
        num_filters,  //c
		1,  //h
		1 ));  //w


	//allocate the tensor for input embedding gradient
	checkCUDNN(cudnnCreateTensorDescriptor(&tensor_C_grad));
	checkCUDNN(cudnnSetTensor4dDescriptor( tensor_C_grad,
    	cudnn_tensor_format,
        cudnn_dtype,
        minibatch_size,  //n
        1,  //c
		longest_word,  //h
		char_emb_size ));  //w


	//allocate the tensor for filter gradients
	checkCUDNN(cudnnCreateFilterDescriptor(&filter_H_grad));
	checkCUDNN(cudnnSetFilter4dDescriptor( filter_H_grad,
        cudnn_dtype,
        num_filters, //k
        1, //c
        filter_size, //h
		char_emb_size ));  //w

	//allocate tensor for the bias
	checkCUDNN(cudnnCreateTensorDescriptor(&tensor_b_grad));
	checkCUDNN(cudnnSetTensor4dDescriptor( tensor_b_grad,
    	cudnn_tensor_format,
        cudnn_dtype,
        1,  //n
        num_filters,  //c
		1,  //h
		1 ));  //w


	//allocate the tensor for output of conv gradient
	checkCUDNN(cudnnCreateTensorDescriptor(&tensor_output_conv_err));
	checkCUDNN(cudnnSetTensor4dDescriptor( tensor_output_conv_err,
    	cudnn_tensor_format,
        cudnn_dtype,
        minibatch_size,  //n
        num_filters,  //c
		longest_word - filter_size + 1,  //h
		1 ));  //w



	//allocate the filters
	checkCUDNN(cudnnCreateFilterDescriptor(&filter_H));
	checkCUDNN(cudnnSetFilter4dDescriptor( filter_H,
        cudnn_dtype,
        num_filters, //k
        1, //c
        filter_size, //h
		char_emb_size ));  //w

	//get workspace size for conv forward
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize( cudnnHandle,
		nodes[0]->tensor_C,
		filter_H,
		cudnn_conv_info,
		nodes[0]->tensor_output_conv,
		cudnn_conv_algo,
		&workspace_conv_forward_size));

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_workspace_conv_forward,workspace_conv_forward_size),"GPU memory allocation failed\n");
	cudaDeviceSynchronize();
	//BZ_CUDA::logger << "Size of conv forward workspace: " << workspace_conv_forward_size << "\n";


	int temp_n = -1;
	int temp_c = -1;
	int temp_h = -1;
	int temp_w = -1;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim( cudnn_conv_info,
		nodes[0]->tensor_C,
		filter_H,
		&temp_n,
		&temp_c,
		&temp_h,
		&temp_w));

	// std::cout << "Printing convolution forward 2D output dimensions\n";
	// std::cout << "n = " << temp_n << "\n";
	// std::cout << "c = " << temp_c << "\n";
	// std::cout << "h = " << temp_h << "\n";
	// std::cout << "w = " << temp_w << "\n";
	// std::cout << "Printing my dimensions\n";
	// std::cout << "n = " << minibatch_size << "\n";
	// std::cout << "c = " << num_filters << "\n";
	// std::cout << "h = " << longest_word - filter_size + 1 << "\n";
	// std::cout << "w = " << 1 << "\n";

	if(temp_n!=minibatch_size) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for conv forward\n";
		exit (EXIT_FAILURE);
	}
	if(temp_c!=num_filters) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for conv forward\n";
		exit (EXIT_FAILURE);
	}
	if(temp_h!= (longest_word - filter_size + 1) ) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for conv forward\n";
		exit (EXIT_FAILURE);
	}
	if(temp_w!=1) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for conv forward\n";
		exit (EXIT_FAILURE);
	}


	// set pooling descriptor
	checkCUDNN(cudnnCreatePoolingDescriptor(&cudnn_poolingDesc));
	checkCUDNN(cudnnSetPooling2dDescriptor( cudnn_poolingDesc,
		CUDNN_POOLING_MAX,
		longest_word - filter_size + 1,
		1,
		0,
		0,
		longest_word - filter_size + 1,
		1 ));


	//check the output dimension for pooling forward
	
	checkCUDNN(cudnnGetPooling2dForwardOutputDim( cudnn_poolingDesc,
		nodes[0]->tensor_output_conv,
		&temp_n,
		&temp_c,
		&temp_h,
		&temp_w));

	// std::cout << "Printing convolution forward 2D output dimensions\n";
	// std::cout << "n = " << temp_n << "\n";
	// std::cout << "c = " << temp_c << "\n";
	// std::cout << "h = " << temp_h << "\n";
	// std::cout << "w = " << temp_w << "\n";
	// std::cout << "Printing my dimensions\n";
	// std::cout << "n = " << minibatch_size << "\n";
	// std::cout << "c = " << num_filters << "\n";
	// std::cout << "h = " << 1 << "\n";
	// std::cout << "w = " << 1 << "\n";

	if(temp_n!=minibatch_size) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for pool forward\n";
		exit (EXIT_FAILURE);
	}
	if(temp_c!=num_filters) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for pool forward\n";
		exit (EXIT_FAILURE);
	}
	if(temp_h!=1) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for pool forward\n";
		exit (EXIT_FAILURE);
	}
	if(temp_w!=1) {
		BZ_CUDA::logger << "ERROR: incorrect sizes for pool forward\n";
		exit (EXIT_FAILURE);
	}

	//----------------------------- get workspaces for convolution backward -----------------------------

	//get workspace for conv backward data
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize( cudnnHandle,
	    filter_H,
	    tensor_output_conv_err,
		cudnn_conv_info,
	    tensor_C_grad,
		cudnn_conv_back_data_algo,
		&workspace_conv_backward_data_size));
	//std::cout << "workspace size for gradient of convolution backward data " << workspace_conv_backward_data_size << "\n";
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_workspace_conv_backward_data,workspace_conv_backward_data_size),"GPU memory allocation failed\n");


	//get workspace for conv backward filter
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize( cudnnHandle,
		nodes[0]->tensor_C,
	    tensor_output_conv_err,
		cudnn_conv_info,
		filter_H_grad,
	    cudnn_conv_back_filter_algo,
	  	&workspace_conv_backward_filter_size ));
	//std::cout << "workspace size for gradient of convolution backward filter " << workspace_conv_backward_filter_size << "\n";
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_workspace_conv_backward_filter,workspace_conv_backward_filter_size),"GPU memory allocation failed\n");


	thrust_d_H_grad = thrust::device_pointer_cast(d_H_grad);
	thrust_d_Q_grad = thrust::device_pointer_cast(d_Q_grad);
	thrust_d_b_grad = thrust::device_pointer_cast(d_b_grad);

	clear_gradients();
}	


//cuda kernels
template<typename dType>
__global__
void tanh_kernel(dType *d_final,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_final[i] = tanh_wrapper(d_final[i]);
	}
}

//cuda kernels
template<typename dType>
__global__
void tanh_error_kernel(dType *d_final_error,dType *d_tanh_val,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_final_error[i] = d_final_error[i]*(1 - d_tanh_val[i]*d_tanh_val[i]);
	}
}


__device__
inline int tensor_index(int n,int c,int h,int w,int n_dim,int c_dim,int h_dim,int w_dim) {
	return w + (h * w_dim) + (c * w_dim * h_dim) + (n * w_dim * h_dim * c_dim);
}


/*
	-vocab indicies are in the format of word len, word len, word len, 
	-size is word_len*minibatch_size
	-the -1 is the NULL character that sets the embedding to zero
*/
template<typename dType>
__global__
void load_char_emb_kernel(dType *d_C,dType *d_Q,int *d_vocab_indicies,int char_emb_size,int minibatch_size,int longest_word) {

	int i_start = threadIdx.x;
	int i_end = char_emb_size*longest_word;
	int i_step = blockDim.x;

	for(int minibatch_index = blockIdx.x; minibatch_index < minibatch_size; minibatch_index+=gridDim.x) {

		for(int i=i_start; i<i_end; i+=i_step) {
			int char_emb_index = i%char_emb_size;
			int curr_word_pos = i/char_emb_size;
			int word_index = d_vocab_indicies[IDX2C(curr_word_pos,minibatch_index,longest_word)];
			if(word_index==-1) {
				d_C[tensor_index(minibatch_index,0,curr_word_pos,char_emb_index,minibatch_size,1,longest_word,char_emb_size)] = 0;
			}
			else {
				d_C[tensor_index(minibatch_index,0,curr_word_pos,char_emb_index,minibatch_size,1,longest_word,char_emb_size)] = \
					d_Q[IDX2C(char_emb_index,word_index,char_emb_size)];
			}
		}
	}
}




template<typename dType>
__global__
void update_char_embeddings(dType *d_C_err,dType *d_Q_grad,int *d_vocab_indicies,int char_emb_size,int minibatch_size,int longest_word) {

	int i_start = threadIdx.x;
	int i_end = char_emb_size*longest_word;
	int i_step = blockDim.x;

	for(int minibatch_index = blockIdx.x; minibatch_index < minibatch_size; minibatch_index+=gridDim.x) {
		for(int i=i_start; i<i_end; i+=i_step) {
			int char_emb_index = i%char_emb_size;
			int curr_word_pos = i/char_emb_size;
			int word_index = d_vocab_indicies[IDX2C(curr_word_pos,minibatch_index,longest_word)];

			if(word_index!=-1) {
				dType temp_val = d_C_err[tensor_index(minibatch_index,0,curr_word_pos,char_emb_index,minibatch_size,1,longest_word,char_emb_size)]; 
				atomicAdd(&(d_Q_grad[IDX2C(char_emb_index,word_index,char_emb_size)]),(temp_val));
			}
		}
	}
}

/*
	------------------ Inputs to function -----------------
	length = length of sentences/words (zero padded to be of same length)
	input_size = embedding_size
	feature_maps = table of feature maps (for each kernel width)
	kernels = table of kernel widths
	num_kernels = number of kernel widths


*/
template<typename dType>
void conv_char_layer<dType>::forward(int index) {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(device_number);

	d_vocab_indicies = d_vocab_indicies_full + minibatch_size*longest_word*index;

	if(decode_source || decode_target) {
		d_vocab_indicies = d_vocab_indicies_full + curr_decode_step;
		curr_decode_step += 1*longest_word;
		index = 0;
		if(decode_target) {
			d_vocab_indicies = d_vocab_indicies_full;
		}
	}



	// thrust::device_ptr<int> thrust_d_vocab = thrust::device_pointer_cast(d_vocab_indicies);
	// devSynchAll();
	// std::cout << "PRINTING first word\n";
	// for(int i=0; i<longest_word; i++) {
	// 	std::cout << thrust_d_vocab[i] << " ";
	// }
	// std::cout << "\n";

	//load in the correct character embeddings
	load_char_emb_kernel<<<256,256,0,s0>>>(nodes[index]->d_C,d_Q,d_vocab_indicies,char_emb_size,minibatch_size,longest_word);

	//run the convolution
	//alpha = 1
	//beta = 0
	dType alpha =1;
	dType beta = 0;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnConvolutionForward( cudnnHandle,
		&alpha,
		nodes[index]->tensor_C,
		nodes[index]->d_C,
		filter_H,
		d_H,
		cudnn_conv_info,
		cudnn_conv_algo,
		d_workspace_conv_forward,
		workspace_conv_forward_size,
		&beta,
		nodes[index]->tensor_output_conv,
		nodes[index]->d_output_conv ));


	//add in bias
	alpha = 1;
	beta = 1;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnAddTensor( cudnnHandle,
		&alpha,
		tensor_b,
		d_b,
		&beta,
		nodes[index]->tensor_output_conv,
		nodes[index]->d_output_conv ));

	//send through tanh
	tanh_kernel<<<256,256,0,s0>>>(nodes[index]->d_output_conv,minibatch_size*num_filters*(longest_word - filter_size + 1));


	// thrust::device_ptr<dType> thrust_d_tanh = thrust::device_pointer_cast(nodes[index]->d_output_conv);
	// devSynchAll();
	// std::cout << "PRINTING charCNN tanh\n";
	// std::cout << thrust_d_tanh[0] << " " << thrust_d_tanh[num_filters*(longest_word - filter_size + 1) - 1] << "\n";

	// thrust::device_ptr<dType> thrust_d_Q = thrust::device_pointer_cast(d_Q);
	// devSynchAll();
	// std::cout << "PRINTING charCNN Q\n";
	// std::cout << thrust_d_Q[55] << " " << thrust_d_Q[85] << "\n";



	//run pooling forward
	alpha = 1;
	beta = 0;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnPoolingForward( cudnnHandle,
		cudnn_poolingDesc,
		&alpha,
		nodes[index]->tensor_output_conv, 
		nodes[index]->d_output_conv,
		&beta,
		nodes[index]->tensor_output_pooling,
		nodes[index]->d_output_pooling ));

	//now send this to LSTM or highway network
	//call highway networks here
	for(int i=0; i<highway_layers.size(); i++) {
		dType *d_y_temp;
		if(i==0) {
			d_y_temp = nodes[index]->d_output_pooling;
		}
		else {
			d_y_temp = highway_layers[i-1]->nodes[index]->d_z;
		}
		highway_layers[i]->forward(index,d_y_temp);
	}

	cudaEventRecord(forward_prop_done,s0);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

}


template<typename dType>
void conv_char_layer<dType>::backward(int index) {

	/*	
		- compute gradients for highway networks
		- compute gradient from max pooling
		- compute gradients for filters
		- compute gradients for convolution bias
		- compute gradients with respect to embeddings
	*/

	cudaSetDevice(device_number);

	//wait for the backprop in lowest LSTM layer to finish
	cudaStreamWaitEvent(s0,back_prop_start,0);

	d_vocab_indicies = d_vocab_indicies_full + minibatch_size*longest_word*index;

	//call highway networks here
	for(int i=highway_layers.size()-1; i>=0 ; i--) {
		dType *d_Err_z_temp=NULL;
		if(i != (highway_layers.size()-1)) {
			d_Err_z_temp = highway_layers[i+1]->d_Err_y;
		}
		highway_layers[i]->backward(index,d_Err_z_temp);
	}

	cudaMemcpyAsync(d_output_pooling_err,highway_layers[0]->d_Err_y,num_filters*minibatch_size*sizeof(dType), cudaMemcpyDefault,s0);

	//run pooling backward
	dType alpha = 1;
	dType beta = 0;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnPoolingBackward( cudnnHandle,
		cudnn_poolingDesc,
		&alpha,
		nodes[index]->tensor_output_pooling,
		nodes[index]->d_output_pooling,
		tensor_output_pooling_err,
		d_output_pooling_err,
		nodes[index]->tensor_output_conv,
		nodes[index]->d_output_conv,
		&beta,
		tensor_output_conv_err,
		d_output_conv_err ));


	//run error through tanh backward
	tanh_error_kernel<<<256,256,0,s0>>>(d_output_conv_err,nodes[index]->d_output_conv,minibatch_size*num_filters*(longest_word - filter_size + 1));

	//run conv filter backward
	alpha = 1;
	beta = 1;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle,
		&alpha,
		nodes[index]->tensor_C,
		nodes[index]->d_C,
		tensor_output_conv_err,
		d_output_conv_err,
		cudnn_conv_info,
		cudnn_conv_back_filter_algo,
		d_workspace_conv_backward_filter,
		workspace_conv_backward_filter_size,
		&beta,
		filter_H_grad,
		d_H_grad ));


	//run conv data backward
	alpha = 1;
	beta = 0;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle,
		&alpha,
		filter_H,
		d_H,
		tensor_output_conv_err,
		d_output_conv_err,
		cudnn_conv_info,
		cudnn_conv_back_data_algo,
		d_workspace_conv_backward_data,
		workspace_conv_backward_data_size,
		&beta,
		tensor_C_grad,
		d_C_err ));


	//run conv bias backward
	alpha = 1;
	beta = 1;
	checkCUDNN(cudnnSetStream(cudnnHandle,s0));
	checkCUDNN(cudnnConvolutionBackwardBias( cudnnHandle,
		&alpha,
		tensor_output_conv_err,
		d_output_conv_err,
		&beta,
		tensor_b_grad,
		d_b_grad));

	//now send gradients to correct character embedding
	update_char_embeddings<<<256,256,0,s0>>>(d_C_err,d_Q_grad,d_vocab_indicies,char_emb_size,minibatch_size,longest_word);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

}


template<typename dType>
void conv_char_layer<dType>::check_gradients(dType epsilon) {

	std::cout << "--------------------GRADIENT CHECKING FOR HIGHWAY NETWORKS GPU-------------------------\n";
	for(int i=0; i<highway_layers.size(); i++) {
		std::cout << "--------------------GRADIENT CHECKING FOR HIGHWAY NETWORK LAYER:" << i+1 << "-------------------------\n";
		highway_layers[i]->check_gradients(epsilon);
	}

	std::cout << "--------------------GRADIENT CHECKING FOR CHAR-CNN LAYER GPU-------------------------\n";
	std::cout << "GRADIENT CHECKING FOR Q\n";
	check_gradient_GPU(epsilon,d_Q,d_Q_grad,char_emb_size,num_unique_chars);
	std::cout << "GRADIENT CHECKING FOR H\n";
	check_gradient_GPU(epsilon,d_H,d_H_grad,char_emb_size,num_filters*filter_size);
	std::cout << "GRADIENT CHECKING FOR bias in conv\n";
	check_gradient_GPU(epsilon,d_b,d_b_grad,num_filters,1);

}	

template<typename dType>
void conv_char_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {

	cudaSetDevice(device_number);

	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaSetDevice(device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaSetDevice(device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			//std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] <<"\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
			else if(d_thrust_grad[IDX2C(i,j,rows)]==0 ||loss/(2*epsilon) ==0) {
				// std::cout << "ZERO GRADIENTS\n";
				// std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				// std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				// std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				// std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}

