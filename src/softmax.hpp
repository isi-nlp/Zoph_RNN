

template<typename dType>
void softmax_layer<dType>::init_loss_layer(struct neuralMT_model<precision> *model,global_params &params)
{
	this->output_vocab_size = params.target_vocab_size;
	this->LSTM_size = params.LSTM_size;
	this->clip_gradients = params.clip_gradient;
	this->model = model;
	this->norm_clip = params.norm_clip;
	this->minibatch_size = params.minibatch_size;
	this->learning_rate = params.learning_rate;
	this->scaled =  true;
	this->train_perplexity = params.train_perplexity;
	this->truncated_softmax = false;
	this->dropout = params.dropout;
	this->dropout_rate = params.dropout_rate;

	init_softmax_layer_GPU(output_vocab_size,minibatch_size,model,params.norm_clip,params.LSTM_size, clip_gradients,learning_rate,params.longest_sent);
}


template<typename dType>
void softmax_layer<dType>::init_softmax_layer_GPU(int output_vocab_size,int minibatch_size,
	struct neuralMT_model<precision> *model,dType norm_clip,int LSTM_size, bool clip_gradients,dType learning_rate,int longest_sent) {

	cudaSetDevice(s_layer_info.device_number);

	thrust_h_outputdist.resize(output_vocab_size * minibatch_size);
	thrust_h_normalization.resize(1 * minibatch_size);
	thrust_d_outputdist.resize(output_vocab_size * minibatch_size);
	thrust_d_normalization.resize(1 * minibatch_size);

	initialize_thrust_vector(thrust_h_outputdist,output_vocab_size * minibatch_size);
    initialize_thrust_vector(thrust_h_normalization,1 * minibatch_size);

    thrust_d_outputdist = thrust_h_outputdist;
    thrust_d_normalization = thrust_h_normalization;

    d_outputdist = thrust::raw_pointer_cast(&thrust_d_outputdist[0]);
    d_normalization = thrust::raw_pointer_cast(&thrust_d_normalization[0]);

    full_matrix_setup(&h_D,&d_D,output_vocab_size,LSTM_size);
	full_matrix_setup(&h_h_t,&d_h_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_b_d,&d_b_d,output_vocab_size,1);
	full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,LSTM_size,minibatch_size);
	full_vector_setup_ones(&h_ones,&d_ones,output_vocab_size);
	full_matrix_setup(&h_D_grad,&d_D_grad,output_vocab_size,LSTM_size);
	full_matrix_setup_0(&h_output_vocab_indices,&d_output_vocab_indices,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_output_vocab_indices_01,&d_output_vocab_indices_01,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_output_vocab_indices_01_float,&d_output_vocab_indices_01_float,minibatch_size,longest_sent);
	full_vector_setup(&h_b_d_grad,&d_b_d_grad,output_vocab_size);

	thrust_d_D_grad = thrust::device_pointer_cast(d_D_grad);
	thrust_d_b_d_grad = thrust::device_pointer_cast(d_b_d_grad);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist_perp, output_vocab_size*minibatch_size*sizeof(double)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_train_perplexity, 1*sizeof(double)),"GPU memory allocation failed\n");
	cudaMemset(d_train_perplexity,0,1*sizeof(double));

	// if(truncated_softmax) {
	// 	init_truncated_softmax();
	// 	cudaSetDevice(s_layer_info.device_number);
	// }

	curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT);
	boost::uniform_int<> unif_boost( 1, 1000000 );
	curandSetPseudoRandomGeneratorSeed(rand_gen,unif_boost(BZ_CUDA::gen));


	for(int i=0; i<longest_sent; i++) {
		nodes.push_back( softmax_node<dType>(LSTM_size,minibatch_size,output_vocab_size,i,dropout) );
	}

	//now start clearning matrices at the end of the minibatch instead of beginning
	cudaSetDevice(s_layer_info.device_number);
	clear_gradients();

	cudaSetDevice(0);
}

template<typename dType>
void softmax_layer<dType>::init_truncated_softmax() {
	cudaSetDevice(s_layer_info.device_number);
	full_matrix_setup(&h_subset_D,&d_subset_D,shortlist_size+sampled_size,LSTM_size);
	full_matrix_setup(&h_subset_D_grad,&d_subset_D_grad,shortlist_size+sampled_size,LSTM_size);
	full_matrix_setup(&h_subset_b_d,&d_subset_b_d,shortlist_size+sampled_size,1);
	full_matrix_setup(&h_subset_b_d_grad,&d_subset_b_d_grad,shortlist_size+sampled_size,1);
	full_matrix_setup(&h_subset_outputdist,&d_subset_outputdist,shortlist_size+sampled_size,minibatch_size);
	full_vector_setup_ones(&h_truncated_vocab_mapping,&d_truncated_vocab_mapping,sampled_size);
	thrust_d_subset_D_grad = thrust::device_pointer_cast(d_subset_D_grad);
	thrust_d_subset_b_d_grad = thrust::device_pointer_cast(d_subset_b_d_grad);

	cudaSetDevice(0);
}


//called per minibatch
template<typename dType>
void softmax_layer<dType>::prep_trunc(int *h_sampled_indices,int len_unique_words_trunc_softmax) {

	cudaSetDevice(s_layer_info.device_number);

	sample_correction = ((dType)(output_vocab_size-shortlist_size-len_unique_words_trunc_softmax))/(sampled_size-len_unique_words_trunc_softmax);
	if( (output_vocab_size-shortlist_size-len_unique_words_trunc_softmax)==0 && (sampled_size-len_unique_words_trunc_softmax)==0) {
		sample_correction=1;
	}
	cudaMemcpy(d_truncated_vocab_mapping, h_sampled_indices, sampled_size*sizeof(int), cudaMemcpyHostToDevice);
	shortlist_size_plus = shortlist_size + len_unique_words_trunc_softmax;

	// std::cout << "sample correction: " <<sample_correction << "\n";
	// std::cout << "shortlist size:"  << shortlist_size << "\n";
	// std::cout << "sampled size: " << sampled_size << "\n";
	// std::cout << "shortlist plus: " << shortlist_size_plus << "\n";

	trunc_set_D<<<256,256>>>(d_D,d_subset_D,trunc_size,output_vocab_size,shortlist_size,d_truncated_vocab_mapping,LSTM_size);
	CUDA_GET_LAST_ERROR("trunc_set_D");
	trunc_set_D<<<256,256>>>(d_b_d,d_subset_b_d,trunc_size,output_vocab_size,shortlist_size,d_truncated_vocab_mapping,1);
	CUDA_GET_LAST_ERROR("trunc_set_b_d");
	cudaDeviceSynchronize();

}




template<typename dType>
void softmax_layer<dType>::clear_gradients() {
	clear_gradients_GPU();
}


template<typename dType>
void softmax_layer<dType>::clear_gradients_GPU() {

	cudaSetDevice(s_layer_info.device_number);

	if(truncated_softmax) {
		//cudaMemsetAsync(d_subset_D_grad,0,trunc_size*LSTM_size*sizeof(dType),s_layer_info.s0);
		//cudaMemsetAsync(d_subset_b_d_grad,0,trunc_size*1*sizeof(dType),s_layer_info.s1);
	}
	else {
		cudaMemsetAsync(d_D_grad,0,output_vocab_size*LSTM_size*sizeof(dType),s_layer_info.s0);
		cudaMemsetAsync(d_b_d_grad,0,output_vocab_size*1*sizeof(dType),s_layer_info.s1);
	}
	cudaDeviceSynchronize();

	cudaSetDevice(0);
}




template<typename dType>
void softmax_layer<dType>::update_weights() {
	update_weights_GPU();
}


template<typename dType>
void softmax_layer<dType>::calculate_global_norm() {

	cudaSetDevice(s_layer_info.device_number);

	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_D_grad,thrust_d_D_grad + output_vocab_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_d_grad,thrust_d_b_d_grad + output_vocab_size*1,unary_op);

	norm_clip_GPU_v2_p1(thrust_d_D_grad,d_D_grad,norm_clip,output_vocab_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);

	devSynchAll();
}


template<typename dType>
void softmax_layer<dType>::update_global_params() {

	cudaSetDevice(s_layer_info.device_number);

	dType alpha = learning_rate;
	dType beta = 1;

	norm_clip_GPU_v2_p2(thrust_d_D_grad,d_D_grad,norm_clip,output_vocab_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);
	

	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, LSTM_size, &alpha, 
		d_D_grad, output_vocab_size, &beta, d_D, output_vocab_size, d_D, output_vocab_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(s_layer_info.handle,s_layer_info.s1);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, 1, &alpha, d_b_d_grad, output_vocab_size, &beta, 
		d_b_d, output_vocab_size, d_b_d, output_vocab_size),"CUBLAS addition update parameter failed\n");
	
	devSynchAll();
}


template<typename dType>
void softmax_layer<dType>::update_weights_GPU() {

	cudaSetDevice(s_layer_info.device_number);

	scale_functor unary_op(minibatch_size);

	if(truncated_softmax) {
		thrust::for_each(thrust_d_subset_D_grad,thrust_d_subset_D_grad + trunc_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_subset_b_d_grad,thrust_d_subset_b_d_grad + trunc_size*1,unary_op);
	}
	else {
		thrust::for_each(thrust_d_D_grad,thrust_d_D_grad + output_vocab_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_b_d_grad,thrust_d_b_d_grad + output_vocab_size*1,unary_op);
	}

	if(BZ_CUDA::individual_grad_clip) {
		clip_mat_kernel<<<std::min(256,(LSTM_size + 256 - 1)/256),256,0,s_layer_info.s0>>>(d_D_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*output_vocab_size);
		clip_mat_kernel<<<std::min(256,(LSTM_size + 256 - 1)/256),256,0,s_layer_info.s0>>>(d_b_d_grad,BZ_CUDA::ind_norm_clip_thres,output_vocab_size*1);
		devSynchAll();
	}

	if(clip_gradients) {

		if(truncated_softmax) {
			norm_clip_GPU_v2(thrust_d_subset_D_grad,d_subset_D_grad,norm_clip,trunc_size*LSTM_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_subset_b_d_grad,d_subset_b_d_grad,norm_clip,trunc_size*1,d_temp_result,d_result);
		}
		else {
			norm_clip_GPU_v2(thrust_d_D_grad,d_D_grad,norm_clip,output_vocab_size*LSTM_size,d_temp_result,d_result);
			norm_clip_GPU_v2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);
		}
	}

	dType alpha = learning_rate;
	dType beta = 1;
	
	if(truncated_softmax) {

		//d_D
		trunc_D_grad_nonshort<<<256,256,0,s_layer_info.s0>>>(d_subset_D_grad,d_D,d_truncated_vocab_mapping,LSTM_size,trunc_size,output_vocab_size,learning_rate,shortlist_size);
		CUDA_GET_LAST_ERROR();
		trunc_D_grad_short<<<256,256,0,s_layer_info.s0>>>(d_subset_D_grad,d_subset_D,LSTM_size,shortlist_size,learning_rate,trunc_size);
		CUDA_GET_LAST_ERROR();

		//d_b_d
		//this is d_D, but with LSTM size of 1
		trunc_D_grad_nonshort<<<256,256,0,s_layer_info.s1>>>(d_subset_b_d_grad,d_b_d,d_truncated_vocab_mapping,1,trunc_size,output_vocab_size,learning_rate,shortlist_size);
		CUDA_GET_LAST_ERROR();
		trunc_D_grad_short<<<256,256,0,s_layer_info.s1>>>(d_subset_b_d_grad,d_subset_b_d,1,shortlist_size,learning_rate,trunc_size);
		CUDA_GET_LAST_ERROR();
	}
	else {
		cublasSetStream(s_layer_info.handle,s_layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, LSTM_size, &alpha, 
			d_D_grad, output_vocab_size, &beta, d_D, output_vocab_size, d_D, output_vocab_size),"CUBLAS addition update parameter failed\n");

		cublasSetStream(s_layer_info.handle,s_layer_info.s1);
		CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, 1, &alpha, d_b_d_grad, output_vocab_size, &beta, 
			d_b_d, output_vocab_size, d_b_d, output_vocab_size),"CUBLAS addition update parameter failed\n");
	}

}

template<typename dType>
void softmax_layer<dType>::dump_weights(std::ofstream &output) {

	cudaSetDevice(s_layer_info.device_number);

	if(truncated_softmax) {
		load_shortlist_D<<<256,256>>>(d_subset_D,d_D,LSTM_size,trunc_size,output_vocab_size,shortlist_size);
		load_shortlist_D<<<256,256>>>(d_subset_b_d,d_b_d,1,trunc_size,output_vocab_size,shortlist_size);
		cudaDeviceSynchronize();
	}
	dump_weights_GPU(output);

}


template<typename dType>
void softmax_layer<dType>::dump_weights_GPU(std::ofstream &output) {
	//std::cout << D << "\n";
	cudaSetDevice(s_layer_info.device_number);

	write_matrix_GPU(d_D,output_vocab_size,LSTM_size,output);
	write_matrix_GPU(d_b_d,output_vocab_size,1,output);
}

template<typename dType>
void softmax_layer<dType>::load_weights(std::ifstream &input) {

	load_weights_GPU(input);
}

template<typename dType>
void softmax_layer<dType>::load_weights_GPU(std::ifstream &input) {
	//std::cout << "----------------------READING D----------------------\n";
	cudaSetDevice(s_layer_info.device_number);

	read_matrix_GPU(d_D,output_vocab_size,LSTM_size,input);
	read_matrix_GPU(d_b_d,output_vocab_size,1,input);
}

template<typename dType>
void softmax_layer<dType>::check_all_gradients(dType epsilon) 
{	
	check_all_gradients_GPU(epsilon);
}



template<typename dType>
void softmax_layer<dType>::check_all_gradients_GPU(dType epsilon) 
{	
	cudaSetDevice(s_layer_info.device_number);

	std::cout << "--------------------GRADIENT CHECKING FOR SOFTMAX LAYER GPU-------------------------\n";
	std::cout << "GRADIENT CHECKING FOR D\n";
	check_gradient_GPU(epsilon,d_D,d_D_grad,output_vocab_size,LSTM_size);
	cudaSetDevice(s_layer_info.device_number);
		
	std::cout << "GRADIENT CHECKING FOR b_d\n";
	check_gradient_GPU(epsilon,d_b_d,d_b_d_grad,output_vocab_size,1);
	cudaSetDevice(s_layer_info.device_number);


	cudaSetDevice(0);
}


template<typename dType>
void softmax_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {
	cudaSetDevice(s_layer_info.device_number);
	cudaDeviceSynchronize();
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}

// template<typename dType>
// template<typename Derived,typename Derived2>
// void softmax_layer<dType>::compute_gradient(const Eigen::MatrixBase<Derived> &h_t,
// 	const Eigen::MatrixBase<Derived2> &vocab_indicies,int index)
// {
// 	compute_gradient_GPU(index);
// }

// template<typename dType>
// void softmax_layer<dType>::compute_gradient_GPU(int index) {

// 	#ifdef REMOVE_STREAMS
// 	devSynchAll();
// 	#endif
// 	if(truncated_softmax) {
// 		get_distribution_GPU(trunc_size,d_subset_outputdist,d_subset_D,d_subset_b_d,d_h_t);
// 		//std::cout << "Starting Get h_t Error in softmax\n";
// 		get_h_t_gradient_GPU(trunc_size,d_subset_D,d_subset_outputdist,d_d_ERRt_ht,index);
// 		//std::cout << "Starting Get D Gradient in softmax\n";
// 		compute_D_gradient_GPU(trunc_size,d_subset_outputdist,d_subset_D_grad,d_h_t);
// 		//std::cout << "Starting Get b_d Gradient in softmax\n";
// 		compute_b_d_gradient_GPU(trunc_size,d_subset_outputdist,d_subset_b_d_grad);
// 		return;
// 	}
// 	//std::cout << "Starting Get Dist in softmax\n";
// 	train_perplexity = false;
// 	get_distribution_GPU(output_vocab_size,d_outputdist,d_D,d_b_d,d_h_t);
// 	train_perplexity = true;
// 	//std::cout << "Starting Get h_t Error in softmax\n";
// 	get_h_t_gradient_GPU(output_vocab_size,d_D,d_outputdist,d_d_ERRt_ht,index);
// 	//std::cout << "Starting Get D Gradient in softmax\n";
// 	//compute_D_gradient_GPU(output_vocab_size,d_outputdist,d_D_grad,d_h_t);
// 	//std::cout << "Starting Get b_d Gradient in softmax\n";
// 	//compute_b_d_gradient_GPU(output_vocab_size,d_outputdist,d_b_d_grad);

// 	#ifdef REMOVE_STREAMS
// 	devSynchAll();
// 	#endif
// }

template<typename dType>
void softmax_layer<dType>::forward_prop(int index) {

	forward_prop_GPU(index);

}

template<typename dType>
void softmax_layer<dType>::forward_prop_GPU(int index) {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(s_layer_info.device_number);
	//wait for the h_t transfer to start
	if(lower_layer.lower_input) {
		cudaStreamWaitEvent(s_layer_info.s0,lower_layer.input_layer->ih_layer_info.h_t_below_transfer,0);
	}
	else {
		cudaStreamWaitEvent(s_layer_info.s0,lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);
	}

	if(dropout && !model->attent_params.attention_model) {
		curandSetStream(rand_gen, s_layer_info.s0);
		curandGenerateUniform_wrapper(nodes[index].d_dropout_mask,LSTM_size*minibatch_size,rand_gen); 
		dropout_kernel<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_dropout_mask,dropout_rate,nodes[index].d_h_t,LSTM_size*minibatch_size);
	}

	get_distribution_GPU(output_vocab_size,nodes[index].d_outputdist,d_D,d_b_d,nodes[index].d_h_t);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
}

//only pass back the error, not D or b_d gradients
template<typename dType>
void softmax_layer<dType>::back_prop1(int index) {

	back_prop1_GPU(index);
}

template<typename dType>
void softmax_layer<dType>::back_prop1_GPU(int index) {
	get_h_t_gradient_GPU(output_vocab_size,d_D,nodes[index].d_outputdist,nodes[index].d_d_ERRt_ht,index);
}

template<typename dType>
void softmax_layer<dType>::back_prop2(int index) {
	back_prop2_GPU(index);
}

template<typename dType>
void softmax_layer<dType>::back_prop2_GPU(int index) {
	compute_D_gradient_GPU(output_vocab_size,nodes[index].d_outputdist,d_D_grad,nodes[index].d_h_t);
	compute_b_d_gradient_GPU(output_vocab_size,nodes[index].d_outputdist,d_b_d_grad);
}




template<typename dType>
void softmax_layer<dType>::get_distribution_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D,dType *d_b_d,dType *d_h_t) 
{
	cudaSetDevice(s_layer_info.device_number);
	//wait until previous h_t,D and b_d gradients are finished because they need the outut dist
	//also wait until the previous backpropinit has finished
	cudaStreamWaitEvent(s_layer_info.s0,s_layer_info.d_ERR_ht_done,0);
	cudaStreamWaitEvent(s_layer_info.s0,s_layer_info.d_D_grad_done,0);
	cudaStreamWaitEvent(s_layer_info.s0,s_layer_info.d_b_d_grad_done,0);
	//cudaStreamWaitEvent(s_layer_info.s0,model->input_layer_target.ih_layer_info.backprop_init,0);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//multiply the D matrix with the hidden state matrix
	dType alpha = 1;
	dType beta = 0;
	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,
	 output_vocab_size, minibatch_size, LSTM_size, &alpha, d_D, output_vocab_size,
	  d_h_t, LSTM_size, &beta, d_outputdist, output_vocab_size),"get_distribution cuBLAS call failed\n");


	//add the bias vector to the matrix
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_bias_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s0 >>>(output_vocab_size,d_outputdist,d_b_d,d_outputdist);
	CUDA_GET_LAST_ERROR();

	if(!scaled) {
		cudaDeviceSynchronize();
		//exp every element in the outputDist matrix
		thrust::for_each(thrust_d_outputdist.begin(),thrust_d_outputdist.end(),exp_functor_gpu());

		//get the normalization vector
		CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(s_layer_info.handle,CUBLAS_OP_T,output_vocab_size,minibatch_size,&alpha,d_outputdist,output_vocab_size,
			d_ones,1,&beta,d_normalization,1),"cuBLAS normaliztion failed\n");

		//invert the values in the normalization matrix
		thrust::for_each(thrust_d_normalization.begin(),thrust_d_normalization.end(),inv_functor_gpu());

		//renormalize outputdist with the normalization vector
		CUBLAS_ERROR_WRAPPER(cublas_dgmm_wrapper(s_layer_info.handle,CUBLAS_SIDE_RIGHT,output_vocab_size,minibatch_size,d_outputdist,output_vocab_size,
			d_normalization,1,d_outputdist,output_vocab_size),"cuBLAS normalization part 2 failed\n");
		cudaDeviceSynchronize();
	}
	else {
		//std::cout << "OVERFLOW KERNEL\n";

		if(truncated_softmax) {
			outputdist_truncated_kernel<<<minibatch_size,SOFTMAX_THREADS,0,s_layer_info.s0>>>(d_outputdist, d_outputdist,output_vocab_size,
				sample_correction,shortlist_size_plus);
		}
		else {
			outputdist_overflow_prevention_kernel<<<minibatch_size,SOFTMAX_THREADS,0,s_layer_info.s0>>>(d_outputdist, d_outputdist, output_vocab_size);
			CUDA_GET_LAST_ERROR();
		}
	}

	
	if(train_perplexity) {
		train_perplexity_kernel<<<1,1,0,s_layer_info.s0>>>(d_output_vocab_indices_single,d_output_vocab_indices_01_single,d_outputdist,
			d_train_perplexity,minibatch_size,output_vocab_size); 
	}

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaEventRecord(s_layer_info.outputdist_done,s_layer_info.s0);

}


template<typename dType>
void softmax_layer<dType>::get_perplexity_GPU(dType *d_h_t,int index) 
{	

	//for passing gradient checking with dropout
	if(dropout && model->train && model->grad_check_flag) {
		dropout_kernel<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_dropout_mask,dropout_rate,d_h_t,LSTM_size*minibatch_size);
	}
	//cudaSetDevice(s_layer_info.device_number);
	//cudaStreamWaitEvent(s_layer_info.s0,model->ih_layer_info.htm1_done,0);
	//cudaStreamWaitEvent(s_layer_info.s0,model->ih_layer_info.ctm1_done,0);
	devSynchAll();
	//multiply the D matrix with the hidden state matrix
	dType alpha = 1;
	dType beta = 0;
	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,
	 output_vocab_size, minibatch_size, LSTM_size, &alpha, d_D, output_vocab_size,
	  d_h_t, LSTM_size, &beta, d_outputdist, output_vocab_size),"get_distribution cuBLAS call failed\n");


	//add the bias vector to the matrix
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_bias_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s0 >>>(output_vocab_size,d_outputdist,d_b_d,d_outputdist);
	CUDA_GET_LAST_ERROR("perplexity bias");

	//std::cout << "OVERFLOW KERNEL\n";
	outputdist_perplexity_kernel<<<minibatch_size,SOFTMAX_THREADS,0,s_layer_info.s0>>>(d_outputdist_perp, d_outputdist, output_vocab_size,false,NULL);
	CUDA_GET_LAST_ERROR("Perplexity Kernel");

	//cudaEventRecord(s_layer_info.outputdist_done,s_layer_info.s0);

	cudaDeviceSynchronize();

}


//get the error for the softmax with respect to h_t
//output vocab indicies should contain no -1's
//output vocab indicies should contain all 1's except for zeros where the column should be zeroed out
//for truncated softmax pass in trunc_size and special 
template<typename dType>
void softmax_layer<dType>::get_h_t_gradient_GPU(int output_vocab_size,dType *d_D,dType *d_outputdist,dType *d_d_ERRt_ht,int index) 
{
	cudaSetDevice(s_layer_info.device_number);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(s_layer_info.s1,s_layer_info.outputdist_done,0);
	dType alpha = -1;
	dType beta = 0;
	//multiply outputdist by D
	cublasSetStream(s_layer_info.handle,s_layer_info.s1);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,output_vocab_size,
		&alpha,d_D,output_vocab_size,d_outputdist,output_vocab_size,&beta,d_d_ERRt_ht,LSTM_size),"cuBLAS h_t gradient failed");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//add in the D rows
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_row_to_matrix_column_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s1 >>>(d_d_ERRt_ht,d_d_ERRt_ht,d_D,d_output_vocab_indices_single,LSTM_size,output_vocab_size);
	CUDA_GET_LAST_ERROR();

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//zero out columns
	int num_block_2 = (LSTM_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim_2(minibatch_size,num_block_2,1);
	zero_columns_kernel_128<<<kernel_dim_2,threads_per_block,0,s_layer_info.s1 >>>(LSTM_size,d_d_ERRt_ht,d_output_vocab_indices_01_single,d_d_ERRt_ht);
		
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	if(dropout && !model->attent_params.attention_model) {
		dropout_kernel<<<256,256,0,s_layer_info.s1>>>(nodes[index].d_dropout_mask,dropout_rate,d_d_ERRt_ht,LSTM_size*minibatch_size);
	}

	//mgpu stuff
	if(lower_layer.copy_d_Err_ht) {
		if(lower_layer.lower_input) {
			cudaMemcpyAsync(lower_layer.input_layer->nodes[index].d_d_ERRt_ht, d_d_ERRt_ht, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s_layer_info.s1);
		}
		else {
			cudaMemcpyAsync(lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht, d_d_ERRt_ht, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s_layer_info.s1);
		}
	}
	else {
		if(lower_layer.lower_input) {
			lower_layer.input_layer->nodes[index].d_d_ERRt_ht = d_d_ERRt_ht;
		}
		else {
			lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht = d_d_ERRt_ht;
		}
	}
	cudaEventRecord(s_layer_info.d_ERR_ht_done,s_layer_info.s1);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//cudaDeviceSynchronize();
}

template<typename dType>
void softmax_layer<dType>::compute_D_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D_grad,dType *d_h_t) 
{

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(s_layer_info.device_number);
	//cudaDeviceSynchronize();
	//zero out h_t
	cudaStreamWaitEvent(s_layer_info.s2,s_layer_info.outputdist_done,0);
	int threads_per_block = 128;
	int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	zero_columns_kernel_128<<<kernel_dim,threads_per_block,0,s_layer_info.s2 >>>(LSTM_size,d_h_t,d_output_vocab_indices_01_single,d_h_t);
	CUDA_GET_LAST_ERROR();

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//multiply output dist and h_t
	dType alpha = -1;
	dType beta = 1;
	cublasSetStream(s_layer_info.handle,s_layer_info.s2);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,output_vocab_size,LSTM_size,minibatch_size,&alpha,d_outputdist,output_vocab_size,
		d_h_t,LSTM_size,&beta,d_D_grad,output_vocab_size),"computing softmax D gradient failed in cuBLAS\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//add columns of h_t to D_grad
	matrix_column_to_matrix_row_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s2 >>>(d_D_grad,d_h_t,d_D_grad,d_output_vocab_indices_single,LSTM_size,output_vocab_size);
	CUDA_GET_LAST_ERROR();

	cudaEventRecord(s_layer_info.d_D_grad_done,s_layer_info.s2);

	//cudaDeviceSynchronize();

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(0);
}


template<typename dType>
void softmax_layer<dType>::compute_b_d_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_b_d_grad) {

	cudaSetDevice(s_layer_info.device_number);

	cudaStreamWaitEvent(s_layer_info.s3,s_layer_info.outputdist_done,0);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	
	//multiply
	dType alpha = -1;
	dType beta = 1;
	cublasSetStream(s_layer_info.handle,s_layer_info.s3);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(s_layer_info.handle,CUBLAS_OP_N,output_vocab_size,minibatch_size,&alpha,d_outputdist,output_vocab_size,
		d_output_vocab_indices_01_float_single,1,&beta,d_b_d_grad,1),"cuBLAS compute b_d_gradient failed");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//add ones
	int threads_per_block = 128;
	int num_block = (minibatch_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(1,num_block,1);
	add_ones_b_d_grad<<< kernel_dim,threads_per_block,0,s_layer_info.s3>>>(d_b_d_grad,d_output_vocab_indices_01_single,d_output_vocab_indices_single,minibatch_size);

	cudaEventRecord(s_layer_info.d_b_d_grad_done,s_layer_info.s3);
	
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

}


template<typename dType>
double softmax_layer<dType>::compute_loss_GPU(int index) {
	cudaSetDevice(s_layer_info.device_number);

	devSynchAll();
	get_perplexity_GPU(nodes[index].d_h_t,index);
	cudaSetDevice(s_layer_info.device_number);

	devSynchAll();
	double loss = 0;
	thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_output_vocab_indices_single);
	thrust::device_ptr<int> d_ptr_01 = thrust::device_pointer_cast(d_output_vocab_indices_01_single);
	thrust::device_ptr<double> d_ptr_sm = thrust::device_pointer_cast(d_outputdist_perp);
	for(int i=0; i < minibatch_size; i++) {
		if(d_ptr_01[i]==1) {
			//loss+=std::log((double)d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)]);
			loss+=d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)];
		}
	}
	return loss;
}






//Note d_h_t may lie on different GPU
//WARNING NEED A DEVICE SYNCHRONIZE WHEN DOING MULTIGPU
template<typename dType>
void softmax_layer<dType>::backprop_prep_GPU(dType *d_h_t,int step) 
{
	this->d_h_t = d_h_t;
	this->d_output_vocab_indices_single = d_output_vocab_indices + step;
	this->d_output_vocab_indices_01_single = d_output_vocab_indices_01 + step;
	this->d_output_vocab_indices_01_float_single = d_output_vocab_indices_01_float + step;
}

template<typename dType>
void softmax_layer<dType>::backprop_prep_GPU_mgpu(int step) 
{
	this->d_output_vocab_indices_single = d_output_vocab_indices + step;
	this->d_output_vocab_indices_01_single = d_output_vocab_indices_01 + step;
	this->d_output_vocab_indices_01_float_single = d_output_vocab_indices_01_float + step;
}



template<typename dType>
void softmax_layer<dType>::prep_GPU_vocab_indices(int *h_output_vocab_indices_target,int current_length) {
	cudaSetDevice(s_layer_info.device_number);

	cudaMemcpy(d_output_vocab_indices, h_output_vocab_indices_target, minibatch_size*current_length*sizeof(int), cudaMemcpyHostToDevice);
	//cudaDeviceSynchronize();

	int threads_per_block = 128;
	//int blocks_per_grid = std::min(current_length,128);
	int blocks_per_grid=128;
	vocab_softmax<<<blocks_per_grid,threads_per_block>>>(d_output_vocab_indices,d_output_vocab_indices_01,d_output_vocab_indices_01_float,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("softmax perp");

	//cudaDeviceSynchronize();
	// thrust::device_ptr<int> debug_ptr = thrust::device_pointer_cast(d_output_vocab_indices);
	// thrust::device_ptr<int> debug_ptr_2 = thrust::device_pointer_cast(d_output_vocab_indices_01);
	// thrust::device_ptr<dType> debug_ptr_3 = thrust::device_pointer_cast(d_output_vocab_indices_01_float);
	// for(int i=0; i<minibatch_size*current_length; i++) {
	// 	std::cout << h_output_vocab_indices_target[i] << " | " << debug_ptr[i] << " | " << debug_ptr_2[i] << " | " << debug_ptr_3[i] <<"\n";
	// }
	// std::cout << "\n\n";
	cudaSetDevice(0);
}

//outputdist is passed in because we only want a minibatch of one
template<typename dType>
int softmax_layer<dType>::stoic_generation(dType *h_outputdist,dType *d_outputdist,double temperature) {

	cudaSetDevice(s_layer_info.device_number);

	train_perplexity = false; //just to be sure ...
	minibatch_size = 1; //for get dist to not override other memory
	cudaDeviceSynchronize();
	get_distribution_GPU(output_vocab_size,d_outputdist,d_D,d_b_d,d_h_t);
	cudaSetDevice(s_layer_info.device_number);
	cudaDeviceSynchronize();

	//now generate a random number between 0 and 1
	boost::uniform_real<> distribution(0,1);
	double num = distribution(BZ_CUDA::gen);

	//now get a cumulative sum
	double cumul_sum=0;
	cudaMemcpy(h_outputdist, d_outputdist, output_vocab_size*sizeof(dType), cudaMemcpyDeviceToHost);

	//temperature schedule
	double total_sum=0;
	for(int i=0; i<output_vocab_size; i++) {
		h_outputdist[i] = std::pow(h_outputdist[i],temperature);
		total_sum+=h_outputdist[i];
	}

	for(int i=0; i<output_vocab_size; i++) {
		h_outputdist[i] = h_outputdist[i]/total_sum;
	}

	//double total_sum_check=0;
	// std::cout << "------------------printing out prob for all chars---------------\n";
	// for(int i=0; i<output_vocab_size; i++) {
	// 	total_sum_check+=h_outputdist[i];
	// 	std::cout << "Char: " << i << "   prob: " << h_outputdist[i] << "\n";
	// }
	// std::cout << "Total sum: " << total_sum_check << "\n";
	// std::cout << "Random num: " << num << "\n";
	for(int i=0; i<output_vocab_size; i++) {
		cumul_sum+=h_outputdist[i];
		if (cumul_sum >=num) {
			return i;
		}
	}

	cudaSetDevice(0);
}

template<typename dType>
void softmax_layer<dType>::dump_probs(std::ofstream &LSTM_dump_stream) {
	thrust::device_ptr<dType> output_ptr = thrust::device_pointer_cast(d_outputdist);
	LSTM_dump_stream <<"Output distribution:";
	for(int i=0; i<output_vocab_size; i++) {
		LSTM_dump_stream << output_ptr[i];
		if(i!=output_vocab_size-1) {
			LSTM_dump_stream << " ";
		}
	}
	LSTM_dump_stream << "\n";

}

template<typename dType>
void softmax_layer<dType>::update_learning_rate(dType learning_rate) {
	this->learning_rate = learning_rate;
}

template<typename dType>
double softmax_layer<dType>::get_train_perplexity() {
	cudaSetDevice(s_layer_info.device_number);
	double tmp_perp;
	cudaMemcpy(&tmp_perp,d_train_perplexity,1*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemset(d_train_perplexity,0,1*sizeof(double));
	return tmp_perp;
}


template<typename dType>
void softmax_layer<dType>::get_distribution_GPU_decoder_wrapper() {
	get_distribution_GPU(output_vocab_size,d_outputdist,d_D,d_b_d,d_h_t);
}


template<typename dType>
softmax_layer_gpu_info softmax_layer<dType>::gpu_init(int device_number) {
	s_layer_info.init(device_number);
	return s_layer_info;
}

template<typename dType>
void softmax_layer<dType>::init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer) {
	lower_layer.init_lower_transfer_layer(lower_input,copy_d_Err_ht,input_layer,hidden_layer);
}

template<typename dType>
dType *softmax_layer<dType>::get_ht_ptr(int index) {
	return nodes[index].d_h_t;
}

template<typename dType>
void softmax_layer<dType>::set_ht_ptr(int index,dType *d_h_t) {
	nodes[index].d_h_t = d_h_t;
}

template<typename dType>
cudaEvent_t softmax_layer<dType>::get_ERR_ht_event() {
	return s_layer_info.d_ERR_ht_done;
}

template<typename dType>
dType *softmax_layer<dType>::get_dist_ptr() {
	return d_outputdist;
}



