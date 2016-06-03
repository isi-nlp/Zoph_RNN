



template<typename dType>
attention_combiner_node<dType>::attention_combiner_node(global_params &params,attention_combiner_layer<dType> *model,int index) {

	this->LSTM_size = params.LSTM_size;
	this->minibatch_size = params.minibatch_size;
	this->model = model;
	this->index = index;
	this->add_ht = params.multi_src_params.add_ht;

	dType *h_temp;
	cudaSetDevice(model->device_number);
	full_matrix_setup(&h_temp,&d_ht_final,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp,&d_ERR_ht_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERR_ht_2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERR_ht_top_loss,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERR_ht_top_feed,LSTM_size,minibatch_size);
	full_vector_setup_ones(&h_temp,&d_ones_minibatch,minibatch_size);
}





template<typename dType>
void attention_combiner_node<dType>::forward() {

	dType alpha = 1;
	dType beta = 0;
	cudaSetDevice(model->device_number);
	cudaStreamWaitEvent(model->s0,model->start_forward,0);

	if(add_ht) {
		add_two_mats_into_third_kernel<<<256,256,0,model->s0>>>(d_ht_final,d_ht_1,d_ht_2,LSTM_size*minibatch_size); 
	}
	else {

		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_1,LSTM_size,
			d_ht_1,LSTM_size,&beta,d_ht_final,LSTM_size),"Combiner p1\n");

		beta = 1;
		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_2,LSTM_size,
			d_ht_2,LSTM_size,&beta,d_ht_final,LSTM_size),"Combiner p2\n");

		tanh_bi_forward_kernel<<<256,256,0,model->s0>>>(d_ht_final,model->d_b_d,LSTM_size,minibatch_size);
	}

	zero_h_t<<<256,256,0,model->s0>>>(d_ht_final,*d_indicies_mask,LSTM_size,minibatch_size);

	if( index != (model->longest_sent-1) ) {
		cudaMemcpyAsync(d_h_tild,d_ht_final,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->s0);
	}

	cudaEventRecord(model->forward_prop_done,model->s0);
}




template<typename dType>
void attention_combiner_node<dType>::backward() {

	cudaSetDevice(model->device_number);
	cudaStreamWaitEvent(model->s0,model->start_backward,0);

	// if(BZ_CUDA::print_norms) {
	// 	HPC_output << "ERROR AT INDEX: " << index << " with respect to softmax 2\n";
	// 	devSynchAll();
	// 	norm_clip_GPU_v2_p1_DEBUG(d_ERR_ht_top_loss,LSTM_size*minibatch_size,model->d_temp_result,model->d_result);
	// 	HPC_output << BZ_CUDA::recent_sum << "\n\n";
	// }


	if(model->transfer_done) {
		cudaStreamWaitEvent(model->s0,model->error_htild_below,0);

		// if(BZ_CUDA::print_norms) {
		// 	HPC_output << "ERROR AT INDEX: " << index << " with respect to feed_input\n";
		// 	devSynchAll();
		// 	norm_clip_GPU_v2_p1_DEBUG(d_ERR_ht_top_feed,LSTM_size*minibatch_size,model->d_temp_result,model->d_result);
		// 	HPC_output << BZ_CUDA::recent_sum << "\n\n";
		// }

		add_two_mats_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->s0>>>(d_ERR_ht_top_loss,d_ERR_ht_top_feed,LSTM_size*minibatch_size);
	}
	model->transfer_done = true;

	//zero out the error
	zero_h_t<<<256,256,0,model->s0>>>(d_ERR_ht_top_loss,*d_indicies_mask,LSTM_size,minibatch_size);

	if(add_ht) {
		cudaMemcpyAsync(d_ERR_ht_1,d_ERR_ht_top_loss,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->s0);
		cudaMemcpyAsync(d_ERR_ht_2,d_ERR_ht_top_loss,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->s0);
	}	
	else {

		dType alpha = 1;
		dType beta = 1;
		//multiply the gradient coming down by 1-tanh()^2
		tanh_grad_kernel<<<256,256,0,model->s0>>>(d_ERR_ht_top_loss,d_ERR_ht_top_loss,d_ht_final,LSTM_size*minibatch_size);
		CUDA_GET_LAST_ERROR("Bidirectional tanh grad");

		//calculate gradient with respect to d_top_param_rev
		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			d_ERR_ht_top_loss,LSTM_size,d_ht_1,LSTM_size,&beta,model->d_M_1_grad,LSTM_size),"Attention backprop W_c_1 grad\n");


		//calculate gradient with respect to d_top_param_nonrev
		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			d_ERR_ht_top_loss,LSTM_size,d_ht_2,LSTM_size,&beta,model->d_M_2_grad,LSTM_size),"Attention backprop W_c_2 grad\n");


		//calculate gradient with respect to d_top_bias
		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_ERR_ht_top_loss,LSTM_size,
			d_ones_minibatch,1,&beta,model->d_b_d_grad,1),"backprop b_i_grad failed\n");

		alpha = 1;
		beta = 0;
		//calculate error with respect to h_t_rev
		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha,model->d_M_1,LSTM_size,d_ERR_ht_top_loss,LSTM_size,&beta,d_ERR_ht_1,LSTM_size),"Attention backprop d_ERRnTOt_ct\n");

		//calculate error with respect to h_t_nonrev
		cublasSetStream(model->handle,model->s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha,model->d_M_2,LSTM_size,d_ERR_ht_top_loss,LSTM_size,&beta,d_ERR_ht_2,LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");
	}

	cudaEventRecord(model->backward_prop_done,model->s0);
}


