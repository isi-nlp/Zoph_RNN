
template<typename dType>
attention_node<dType>::attention_node(int LSTM_size,int minibatch_size,int device_number,int D,bool feed_input,attention_layer<dType> *attent_layer,int index) {
	this->LSTM_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->device_number = device_number;
	this->D = D;
	this->feed_input = feed_input;
	this->attent_layer = attent_layer;
	this->index = index;

	cudaSetDevice(device_number);
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_p_t, 1*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_sigma_1, 1*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_lower_upper, 2*minibatch_size*sizeof(int)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_alignments, (2*D+1)*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_tanh_1, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_hs_mat, (2*D+1)*LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_c_t, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_indicies, (2*D+1)*minibatch_size*sizeof(int)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_final_temp_1, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_final_temp_2, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_Wa_cache, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_cached_exp, (2*D+1)*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");

	sigma_sq = (D*D)/4.0;

	this->index = index;
}

template<typename dType>
void attention_node<dType>::feed_input_init(dType *d_ptr_htild) {
	cudaSetDevice(device_number);
	d_lower_htild = d_ptr_htild;
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_ERRtTOn_htild_below, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
}

template<typename dType>
void attention_node<dType>::forward_prop() {


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	
	/*
		1. Compute p_t for the entire minibatch
		2. get the lower and upper ranges for the alignments
		3. Get a padding vector, so the scores after exping can be zeroed
		4. Load in h_s vectors for the minibatch, fill with zeros if off one edge
		5. Load in precomputed W_a * h_s vectors too
		6. Compute v_t with W_a * h_s
		7. exp the scores and multiply them by the mask
		8. compute the alignments from the scores
		9. compute the c_t vectors using the h_s and alignments
		10. tanh( W_c1 * c_t + w_c2 * h_t + b_c)
	*/

	cudaSetDevice(device_number);
	cudaStreamWaitEvent(attent_layer->layer_info.s0,attent_layer->layer_info.start_forward,0);


	//event wait on stream zero

	dType alpha = 1;
	dType beta = 0;

	// devSynchAll();
	// std::cout << "PRINTING:  h_t \n";
	// print_GPU_Matrix(d_h_t,LSTM_size,minibatch_size);
	//W_p * h_t

	// std::cout << "PRINTING h_t\n";
	// devSynchAll();
	// print_GPU_Matrix(d_h_t,LSTM_size,minibatch_size);


	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_p,LSTM_size,
		d_h_t,LSTM_size,&beta,d_tanh_1,LSTM_size),"attention forward p_t part 1\n");

	// std::cout << "PRINTING W_p * h_t\n";
	// devSynchAll();
	// print_GPU_Matrix(d_tanh_1,LSTM_size,minibatch_size);


	//tanh(W_p * h_t)
	tanh_kernel<<< std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_tanh_1,d_tanh_1,LSTM_size*minibatch_size);
	CUDA_GET_LAST_ERROR("attention tanh1");


	//v_p * tanh(W_p * h_t)
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,1,minibatch_size,LSTM_size,&alpha,attent_layer->d_v_p,1,
		d_tanh_1,LSTM_size,&beta,d_sigma_1,1),"attention forward p_t part 2\n");

	// std::cout << "PRINTING v_p * tanh(W_p * h_t)\n";
	// devSynchAll();
	// print_GPU_Matrix(d_sigma_1,1,minibatch_size);


	//sigm(v_p * tanh(W_p * h_t))
	sigmoid_kernel<<<std::min(256,(minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_sigma_1,d_sigma_1,minibatch_size);
	CUDA_GET_LAST_ERROR("attention sigmoid");


	// std::cout << "sigm(v_p * tanh(W_p * h_t))\n";
	// devSynchAll();
	// print_GPU_Matrix(d_sigma_1,1,minibatch_size);

	//S*sigm(v_p * tanh(W_p * h_t))
	alignment_pos_kernel<<<std::min(256,(minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_sigma_1,d_p_t,minibatch_size,attent_layer->d_batch_info);
	CUDA_GET_LAST_ERROR("attention sigmoid 2");

	//at this point d_p_t is filled and is size 1xminibatch
	// std::cout << "P_T\n";
	// devSynchAll();
	// print_GPU_Matrix(d_p_t,1,minibatch_size);

	//get lower and upper ranges
	lower_upper_kernel<<<std::min(256,(2*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_p_t,d_lower_upper,D,attent_layer->d_batch_info,minibatch_size);
	CUDA_GET_LAST_ERROR("attention lower upper");


	// std::cout << "LOWER UPPER INDICIES\n";
	// devSynchAll();
	// print_GPU_Matrix(d_lower_upper,2,minibatch_size);


	//create d_incicies
	create_indicies_kernel<<<1,256,0,attent_layer->layer_info.s0>>>(d_indicies,D,minibatch_size,d_lower_upper,*d_indicies_mask);
	CUDA_GET_LAST_ERROR("attention create indicies");
		


	//get all the h_s vectors loaded in, also load in the W_a * h_s??????, could be a speedup
	load_in_hs_kernel<<<std::min(256,(2*D+1)*minibatch_size),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_mat,D,d_hs_mat,d_indicies,minibatch_size,LSTM_size,attent_layer->d_batch_info);
	CUDA_GET_LAST_ERROR("attention load in hs");


	//precompute h_t multipied by W_a
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
		d_h_t,LSTM_size,&beta,d_h_t_Wa_cache,LSTM_size),"attention forward h_t * W_a\n");

	//do W_a * h_s in the first step then trans(h_t) * (W_a * h_s) in the next
	// for(int i=0; i<2*D+1; i++) {
	// 	// cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	// 	// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
	// 	// 	d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention forward Wa_hs\n");

	// 	// elem_reduce_kernel<<<minibatch_size,NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(d_h_t,attent_layer->d_temp_1,d_alignments + i*(minibatch_size),LSTM_size,minibatch_size);


	// 	elem_reduce_kernel<<<minibatch_size,NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(d_hs_mat + i*(LSTM_size*minibatch_size),d_h_t_Wa_cache,d_alignments + i*(minibatch_size),LSTM_size,minibatch_size);
	// }


	//do one big reduction for the reduce
	elem_reduce_kernel_large<<<std::min(minibatch_size*(2*D+1),256),NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(d_hs_mat,d_h_t_Wa_cache,d_alignments,LSTM_size,minibatch_size,D);

	// std::cout << "Printing d_indicies\n";
	// devSynchAll();
	// print_GPU_Matrix(d_indicies,minibatch_size,2*D+1);


	// std::cout << "Printing p_t\n";
	// devSynchAll();
	// print_GPU_Matrix(d_p_t,1,minibatch_size);



	// std::cout << "Printing batch info\n";
	// devSynchAll();
	// print_GPU_Matrix(attent_layer->d_batch_info,minibatch_size,2);


	// std::cout << "Printing W_p\n";
	// devSynchAll();
	// print_GPU_Matrix(attent_layer->d_W_p,LSTM_size,LSTM_size);

	// std::cout << "Printing v_p\n";
	// devSynchAll();
	// print_GPU_Matrix(attent_layer->d_v_p,LSTM_size,1);


	// for(int i=0; i<2*D+1;i++) {
	// 	devSynchAll();
	// 	std::cout << "Printing h_s loaded in at index: " << i << "\n";
	// 	print_GPU_Matrix(d_hs_mat + i*LSTM_size*minibatch_size,LSTM_size,minibatch_size);
	// }


	//exp all the alignments and multiply them by a 0-1 mask 
	// exp_mask_kernel<<<std::min(256,(minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_indicies,d_alignments,minibatch_size,D);
	// CUDA_GET_LAST_ERROR("attention exp mask");


	//normalize the alignments
	alignment_reduction_kernel<<<1,minibatch_size,0,attent_layer->layer_info.s0>>>(d_alignments,LSTM_size,minibatch_size,D,sigma_sq,d_p_t,d_indicies,d_cached_exp);
	CUDA_GET_LAST_ERROR("attention alignment reduction");



	// std::cout << "Printing alignments\n";
	// devSynchAll();
	// thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(d_alignments);
	// for(int i=0; i<2*D+1; i++) {
	// 	for(int j=0; j<minibatch_size; j++) {
	// 		std::cout << debug_ptr[IDX2C(j,i,minibatch_size)] << " ";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << "\n";

	// std::cout << "Printing d_cached_exp\n";
	// devSynchAll();
	// print_GPU_Matrix(d_cached_exp,2*D+1,minibatch_size);


	//create the c_t vector
	create_c_t_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_alignments,d_hs_mat,d_c_t,LSTM_size,minibatch_size,D);
	CUDA_GET_LAST_ERROR("attention create ct");


	// std::cout << "Printing c_t (context vector)\n";
	// devSynchAll();
	// print_GPU_Matrix(d_c_t,LSTM_size,minibatch_size);


	//W_c_p1 * c_t
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_c_p1,LSTM_size,
		d_c_t,LSTM_size,&beta,d_final_temp_1,LSTM_size),"attention forward p_t part 1\n");


	// //W_c_p2 * h_t
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_c_p2,LSTM_size,
		d_h_t,LSTM_size,&beta,d_final_temp_2,LSTM_size),"attention forward p_t part 2\n");


	//add in the bias and tanh
	tanh_att_forward_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_final_temp_2,d_final_temp_1,d_final_temp_2,attent_layer->d_output_bias,LSTM_size,minibatch_size);
	CUDA_GET_LAST_ERROR("attention tanh forward");

	// std::cout << "MASK FOR ATTENTION\n";
	// devSynchAll();
	// print_GPU_Matrix(*d_indicies_mask,1,minibatch_size);


	zero_h_t<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_final_temp_2, *d_indicies_mask,LSTM_size,minibatch_size);
	//zero out cols based on 0 and 1 indicies

	//send h_tild to the lowest level
	//if last index, then there is nothing to copy to
	if(feed_input && index != (attent_layer->longest_sent-1) ) {
		cudaMemcpyAsync(d_lower_htild,d_final_temp_2,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,attent_layer->layer_info.s0);
	}

	cudaEventRecord(attent_layer->layer_info.forward_prop_done,attent_layer->layer_info.s0);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	// devSynchAll();
	// std::cout << "PRINTING HTILD FROM ATTENTION: \n";
	// print_GPU_Matrix(d_final_temp_2,LSTM_size,minibatch_size);

}


template<typename dType>
void attention_node<dType>::back_prop() {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(device_number);

	cudaStreamWaitEvent(attent_layer->layer_info.s0,attent_layer->layer_info.start_backward,0);

	if(feed_input && attent_layer->transfer_done) {

		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif

		//std::cout << "Adding attention error\n";
		//wait for the feed input backprop error to be sent
		cudaStreamWaitEvent(attent_layer->layer_info.s0,attent_layer->layer_info.error_htild_below,0);
		add_two_mats_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_d_ERRt_ht_tild,d_ERRtTOn_htild_below,LSTM_size*minibatch_size);
	

		// devSynchAll();
		// std::cout << "PRINTING ERROR OF HTILD FROM ATTENTION: \n";
		// print_GPU_Matrix(d_ERRtTOn_htild_below,LSTM_size,minibatch_size);
	}
	attent_layer->transfer_done = true; //for feed input errors, the first error we dont want to add

	dType alpha = 1;
	dType beta = 1;


	//test this for gradients
	zero_h_t<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_d_ERRt_ht_tild, *d_indicies_mask,LSTM_size,minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION zero h_t");

	//multiply the gradient coming down by 1-tanh()^2
	tanh_grad_kernel<<< std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_d_ERRt_ht_tild,d_d_ERRt_ht_tild,d_final_temp_2,LSTM_size*minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION tanh grad");

	//calculate gradient with respect to W_c_1
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRt_ht_tild,LSTM_size,d_c_t,LSTM_size,&beta,attent_layer->d_W_c_p1_grad,LSTM_size),"Attention backprop W_c_1 grad\n");


	//calculate gradient with respect to W_c_2
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRt_ht_tild,LSTM_size,d_h_t,LSTM_size,&beta,attent_layer->d_W_c_p2_grad,LSTM_size),"Attention backprop W_c_2 grad\n");


	//calculate gradient with respect to output_bias
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(attent_layer->handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRt_ht_tild,LSTM_size,
		attent_layer->d_ones_minibatch,1,&beta,attent_layer->d_output_bias_grad,1),"backprop b_i_grad failed\n");

	alpha = 1;
	beta = 0;


	//calculate error with respect to c_t
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha,attent_layer->d_W_c_p1,LSTM_size,d_d_ERRt_ht_tild,LSTM_size,&beta,attent_layer->d_ERRnTOt_ct,LSTM_size),"Attention backprop d_ERRnTOt_ct\n");


	//calculate first part of error with respect to h_t
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha,attent_layer->d_W_c_p2,LSTM_size,d_d_ERRt_ht_tild,LSTM_size,&beta,attent_layer->d_ERRnTOt_ht_p1,LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");


	//cudaMemset(attent_layer->d_ERRnTOt_as,0,(2*D+1)*minibatch_size*sizeof(dType));

	//more efficent version of the code below with less kernel launches
	error_alignments_kernel_large<<<std::min(minibatch_size*(2*D+1),256),NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ct,d_hs_mat,attent_layer->d_ERRnTOt_as,LSTM_size,minibatch_size,D);
	CUDA_GET_LAST_ERROR("ATTENTION error_alignments");

	//more efficent version of the code below with less kernel launches
	error_hs_ct_kernel_large<<<std::min(256,minibatch_size*(2*D+1)),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ct, d_alignments,d_indicies,attent_layer->d_batch_info,attent_layer->d_total_hs_error,LSTM_size,minibatch_size,D);

	//calculate the error with respect to the alignments
	// for(int i=0; i<2*D+1; i++) {
	// 	// error_alignments_kernel<<<minibatch_size,NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ct,d_hs_mat + i*(LSTM_size*minibatch_size), attent_layer->d_ERRnTOt_as, LSTM_size, minibatch_size,i,D);
	// 	// CUDA_GET_LAST_ERROR("ATTENTION error_alignments");
		
	// 	//send back error for h_s
	// 	error_hs_ct_kernel<<<std::min(256,minibatch_size),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ct, d_alignments,d_indicies,attent_layer->d_batch_info,attent_layer->d_total_hs_error,LSTM_size,minibatch_size,D,i);
	// 	CUDA_GET_LAST_ERROR("ATTENTION error_hs_ct");
	// }




	//calculate the error with respect to p_t
	error_pt_kernel<<<minibatch_size,NUM_ATTENTION_THREADS,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_pt,attent_layer->d_ERRnTOt_as,D,sigma_sq,d_indicies,minibatch_size,d_p_t,d_alignments);
	CUDA_GET_LAST_ERROR("ATTENTION error_pt");


	//calculate the error with respect to v_p
	att_vp_error<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(d_sigma_1,d_tanh_1,attent_layer->d_temp_1,attent_layer->d_ERRnTOt_pt,attent_layer->d_batch_info,LSTM_size,minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION att_vp_error");


	alpha = 1;
	beta = 1;

	//calculate the error with respect to v_p
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(attent_layer->handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,attent_layer->d_temp_1,LSTM_size,
		attent_layer->d_ones_minibatch,1,&beta,attent_layer->d_v_p_grad,1),"attention backprop v_p_grad failed\n");


	//calculate error with respect to W_p
	grad_W_p_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_v_p,attent_layer->d_temp_1,d_sigma_1,d_tanh_1,attent_layer->d_ERRnTOt_pt,attent_layer->d_batch_info,LSTM_size,minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION grad_W_p");

	//finish the gradient calculation of W_p with outer product
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,
		&alpha,attent_layer->d_temp_1,LSTM_size,d_h_t,LSTM_size,&beta,attent_layer->d_W_p_grad,LSTM_size),"Attention backprop W_p grad\n");


	//now get the second part of the error with respect to h_t and add it to the first part
	// *** STUFF IS ALREADY STORED IN THE ABOVE TEMP MATRIX ***
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha,attent_layer->d_W_p,LSTM_size,attent_layer->d_temp_1,LSTM_size,&beta,attent_layer->d_ERRnTOt_ht_p1,LSTM_size),"Attention backprop W_p grad\n");



	//fast implementation
	// alpha = 1;
	// beta = 1;
	// get_ht_scalings_Wa_grad_kernel<<<std::min(256,((2*D+1)*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_ERRnTOt_as,d_alignments,d_cached_exp,D,minibatch_size);
	// CUDA_GET_LAST_ERROR("ATTENTION get_ht_scalings_Wa_grad");
	// for(int i=0; i<2*D+1; i++) {

	// 	//for W_a gradient
	// 	beta = 1;
	// 	scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.alignment_streams[IDX2C(0,i,3)]>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,d_h_t,LSTM_size,minibatch_size,i,D);
	// 	CUDA_GET_LAST_ERROR("ATTENTION scale_ht");
	// 	cublasSetStream(attent_layer->handle,attent_layer->layer_info.alignment_streams[IDX2C(0,i,3)]);
	// 	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,attent_layer->d_temp_1,LSTM_size,
	// 		d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,&beta,attent_layer->d_W_a_grad,LSTM_size),"attention backprop Wa\n");


	// 	//for h_t gradient
	// 	beta = 0;
	// 	cublasSetStream(attent_layer->handle,attent_layer->layer_info.alignment_streams[IDX2C(1,i,3)]);
	// 	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
	// 		d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention backprop h_t in alignment\n");
	// 	scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.alignment_streams[IDX2C(1,i,3)]>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,attent_layer->d_temp_1,LSTM_size,minibatch_size,i,D);
	// 	CUDA_GET_LAST_ERROR("ATTENTION scale_ht");
	// 	add_two_mats_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.alignment_streams[IDX2C(1,i,3)]>>>(attent_layer->d_ERRnTOt_ht_p1,attent_layer->d_temp_1,LSTM_size*minibatch_size);
	// 	CUDA_GET_LAST_ERROR("ATTENTION add_two_mats");


	// 	//for h_s gradient
	// 	beta = 0;
	// 	// cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	// 	// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
	// 	// 	d_h_t,LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention backprop h_t in alignment\n");
	// 	// scale_ht_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,attent_layer->d_temp_1,LSTM_size,minibatch_size,i,D);
	// 	// CUDA_GET_LAST_ERROR("ATTENTION scale_ht 2");
	// 	// copy_errors_source<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_error,attent_layer->d_temp_1,d_indicies,LSTM_size,minibatch_size,D,i,attent_layer->d_batch_info);
	// 	// CUDA_GET_LAST_ERROR("ATTENTION copy_errors_source");

	// 	scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.alignment_streams[IDX2C(2,i,3)]>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,d_h_t_Wa_cache,LSTM_size,minibatch_size,i,D);
	// 	CUDA_GET_LAST_ERROR("ATTENTION scale_ht 2");
	// 	copy_errors_source<<<std::min(256,minibatch_size),256,0,attent_layer->layer_info.alignment_streams[IDX2C(2,i,3)]>>>(attent_layer->d_total_hs_error,attent_layer->d_temp_1,d_indicies,LSTM_size,minibatch_size,D,i,attent_layer->d_batch_info);
	// 	CUDA_GET_LAST_ERROR("ATTENTION copy_errors_source");


	// 	cudaEventRecord(attent_layer->layer_info.alignment_events[i],attent_layer->layer_info.alignment_streams[IDX2C(0,i,3)]);
	// 	cudaEventRecord(attent_layer->layer_info.alignment_events[i],attent_layer->layer_info.alignment_streams[IDX2C(1,i,3)]);
	// 	cudaEventRecord(attent_layer->layer_info.alignment_events[i],attent_layer->layer_info.alignment_streams[IDX2C(2,i,3)]);
	// }



	//fast implementation
	//MAKE OPERATIONS ATOMIC
	// alpha = 1;
	// beta = 1;
	// get_ht_scalings_Wa_grad_kernel<<<std::min(256,((2*D+1)*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_ERRnTOt_as,d_alignments,d_cached_exp,D,minibatch_size);
	// CUDA_GET_LAST_ERROR("ATTENTION get_ht_scalings_Wa_grad");
	// for(int i=0; i<2*D+1; i++) {

	// 	//for W_a gradient
	// 	beta = 1;
	// 	scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,d_h_t,LSTM_size,minibatch_size,i,D);
	// 	CUDA_GET_LAST_ERROR("ATTENTION scale_ht");
	// 	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	// 	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,attent_layer->d_temp_1,LSTM_size,
	// 		d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,&beta,attent_layer->d_W_a_grad,LSTM_size),"attention backprop Wa\n");


	// 	//for h_t gradient
	// 	beta = 0;
	// 	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	// 	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
	// 		d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention backprop h_t in alignment\n");
	// 	scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,attent_layer->d_temp_1,LSTM_size,minibatch_size,i,D);
	// 	CUDA_GET_LAST_ERROR("ATTENTION scale_ht");
	// 	add_two_mats_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ht_p1,attent_layer->d_temp_1,LSTM_size*minibatch_size);
	// 	CUDA_GET_LAST_ERROR("ATTENTION add_two_mats");


	// 	//for h_s gradient
	// 	beta = 0;
	// 	// cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	// 	// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
	// 	// 	d_h_t,LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention backprop h_t in alignment\n");
	// 	// scale_ht_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,attent_layer->d_temp_1,LSTM_size,minibatch_size,i,D);
	// 	// CUDA_GET_LAST_ERROR("ATTENTION scale_ht 2");
	// 	// copy_errors_source<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_error,attent_layer->d_temp_1,d_indicies,LSTM_size,minibatch_size,D,i,attent_layer->d_batch_info);
	// 	// CUDA_GET_LAST_ERROR("ATTENTION copy_errors_source");

	// 	scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,d_h_t_Wa_cache,LSTM_size,minibatch_size,i,D);
	// 	CUDA_GET_LAST_ERROR("ATTENTION scale_ht 2");
	// 	copy_errors_source<<<std::min(256,minibatch_size),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_error,attent_layer->d_temp_1,d_indicies,LSTM_size,minibatch_size,D,i,attent_layer->d_batch_info);
	// 	CUDA_GET_LAST_ERROR("ATTENTION copy_errors_source");
	// }
	cudaMemsetAsync(attent_layer->d_h_s_sum,0,LSTM_size*minibatch_size*sizeof(dType),attent_layer->layer_info.s0);
	//optimized version of the code above
	alpha = 1;
	beta = 1;
	get_ht_scalings_Wa_grad_kernel<<<std::min(256,((2*D+1)*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_ERRnTOt_as,d_alignments,d_cached_exp,D,minibatch_size);
	CUDA_GET_LAST_ERROR("ATTENTION get_ht_scalings_Wa_grad");
	for(int i=0; i<2*D+1; i++) {

		//for W_a gradient
		scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,minibatch_size,i,D);
		CUDA_GET_LAST_ERROR("ATTENTION scale_ht");
		add_two_mats_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_s_sum,attent_layer->d_temp_1,LSTM_size*minibatch_size);
		CUDA_GET_LAST_ERROR("ATTENTION add_two_mats");

		//for h_t gradient
		beta = 0;
		// cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
		// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
		// 	d_hs_mat + i*(LSTM_size*minibatch_size),LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention backprop h_t in alignment\n");
		// scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,attent_layer->d_temp_1,LSTM_size,minibatch_size,i,D);
		// CUDA_GET_LAST_ERROR("ATTENTION scale_ht");
		// add_two_mats_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_ERRnTOt_ht_p1,attent_layer->d_temp_1,LSTM_size*minibatch_size);
		// CUDA_GET_LAST_ERROR("ATTENTION add_two_mats");


		//for h_s gradient
		beta = 0;
		// cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
		// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
		// 	d_h_t,LSTM_size,&beta,attent_layer->d_temp_1,LSTM_size),"attention backprop h_t in alignment\n");
		// scale_ht_kernel<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,attent_layer->d_temp_1,LSTM_size,minibatch_size,i,D);
		// CUDA_GET_LAST_ERROR("ATTENTION scale_ht 2");
		// copy_errors_source<<<256,256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_error,attent_layer->d_temp_1,d_indicies,LSTM_size,minibatch_size,D,i,attent_layer->d_batch_info);
		// CUDA_GET_LAST_ERROR("ATTENTION copy_errors_source");x


		scale_ht_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_h_t_Wa_factor,attent_layer->d_temp_1,d_h_t_Wa_cache,LSTM_size,minibatch_size,i,D);
		CUDA_GET_LAST_ERROR("ATTENTION scale_ht 2");
		copy_errors_source<<<std::min(256,minibatch_size),256,0,attent_layer->layer_info.s0>>>(attent_layer->d_total_hs_error,attent_layer->d_temp_1,d_indicies,LSTM_size,minibatch_size,D,i,attent_layer->d_batch_info);
		CUDA_GET_LAST_ERROR("ATTENTION copy_errors_source");

	}

	beta = 1;
	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,d_h_t,LSTM_size,
		attent_layer->d_h_s_sum,LSTM_size,&beta,attent_layer->d_W_a_grad,LSTM_size),"attention backprop Wa\n");

	cublasSetStream(attent_layer->handle,attent_layer->layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(attent_layer->handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,attent_layer->d_W_a,LSTM_size,
		attent_layer->d_h_s_sum,LSTM_size,&beta,attent_layer->d_ERRnTOt_ht_p1,LSTM_size),"attention backprop h_t in alignment\n");



	cudaMemcpyAsync(d_d_ERRt_ht_tild,attent_layer->d_ERRnTOt_ht_p1,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDeviceToDevice,attent_layer->layer_info.s0);
	CUDA_GET_LAST_ERROR("TRANSFER COPY ATTENTION");

	cudaEventRecord(attent_layer->layer_info.backward_prop_done,attent_layer->layer_info.s0);


	// for(int i=0; i<2*D+1;i++) {
	// 	devSynchAll();
	// 	std::cout << "Printing Error h_s at index: " << i << "\n";
	// 	print_GPU_Matrix(d_hs_mat + i*LSTM_size*minibatch_size,LSTM_size,minibatch_size);
	// }

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

}


