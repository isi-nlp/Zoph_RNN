template<typename dType>
bi_encoder<dType>::bi_encoder() {

}


template<typename dType>
void bi_encoder<dType>::init_layer(global_params &params,int device_number,neuralMT_model<dType> *model,std::vector<int> &gpu_indicies) {

	this->num_layers = params.num_layers;
	this->LSTM_size = params.LSTM_size;
	this->minibatch_size = params.minibatch_size;
	this->longest_sent = params.longest_sent;
	this->model = model;
	this->norm_clip = norm_clip;
	this->learning_rate = learning_rate;
	layer_info.init(device_number);
	this->gpu_indicies = gpu_indicies;

	if(params.bi_dir_params.bi_dir_comb) {
		model_type = COMBINE;
	}

	dType *h_temp;
	for(int i=0; i<num_layers; i++) {
		d_hs_start_target.push_back(NULL);
		d_hs_final_target.push_back(NULL);
		d_horiz_param_rev.push_back(NULL); //for transforming the top indicies
		d_horiz_param_nonrev.push_back(NULL); //for transforming the top indicies
		d_horiz_bias.push_back(NULL); //for transforming the top indicies
		d_horiz_param_rev_grad.push_back(NULL);
		d_horiz_param_nonrev_grad.push_back(NULL);
		d_horiz_bias_grad.push_back(NULL);
		d_hs_rev_error_horiz.push_back(NULL);
		d_hs_nonrev_error_horiz.push_back(NULL);
		d_ct_rev_error_horiz.push_back(NULL);
		d_ct_nonrev_error_horiz.push_back(NULL);
		d_ct_final_target.push_back(NULL);
		d_horiz_param_rev_ct.push_back(NULL); 
		d_horiz_param_nonrev_ct.push_back(NULL); 
		d_horiz_param_rev_ct_grad.push_back(NULL); 
		d_horiz_param_nonrev_ct_grad.push_back(NULL); 
		d_ct_start_target.push_back(NULL);
		d_temp_result_vec.push_back(NULL);
		d_result_vec.push_back(NULL);
	}

	for(int i=0; i<minibatch_size; i++) {
		final_index_hs.push_back(-1);
	}

	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);
			full_matrix_setup(&h_temp,&d_hs_start_target[i],LSTM_size,minibatch_size);
			full_matrix_setup(&h_temp,&d_hs_final_target[i],LSTM_size,minibatch_size);

			full_matrix_setup(&h_temp,&d_horiz_param_rev[i],LSTM_size,LSTM_size);
			full_matrix_setup(&h_temp,&d_horiz_param_nonrev[i],LSTM_size,LSTM_size);
			full_matrix_setup(&h_temp,&d_horiz_bias[i],LSTM_size,1);
			full_matrix_setup(&h_temp,&d_horiz_param_rev_grad[i],LSTM_size,LSTM_size);
			full_matrix_setup(&h_temp,&d_horiz_param_nonrev_grad[i],LSTM_size,LSTM_size);
			full_matrix_setup(&h_temp,&d_horiz_bias_grad[i],LSTM_size,1);

			full_matrix_setup(&h_temp,&d_hs_rev_error_horiz[i],LSTM_size,minibatch_size);
			full_matrix_setup(&h_temp,&d_hs_nonrev_error_horiz[i],LSTM_size,minibatch_size);

			full_matrix_setup(&h_temp,&d_ct_final_target[i],LSTM_size,minibatch_size);

			full_matrix_setup(&h_temp,&d_ct_rev_error_horiz[i],LSTM_size,minibatch_size);
			full_matrix_setup(&h_temp,&d_ct_nonrev_error_horiz[i],LSTM_size,minibatch_size);

			//full_matrix_setup(&h_temp,&d_horiz_param_rev_ct[i],LSTM_size,LSTM_size);
			//full_matrix_setup(&h_temp,&d_horiz_param_nonrev_ct[i],LSTM_size,LSTM_size);
			//full_matrix_setup(&h_temp,&d_horiz_param_rev_ct_grad[i],LSTM_size,LSTM_size);
			//full_matrix_setup(&h_temp,&d_horiz_param_nonrev_ct_grad[i],LSTM_size,LSTM_size);

			full_matrix_setup(&h_temp,&d_ct_start_target[i],LSTM_size,minibatch_size);

			CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result_vec[i], NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
			CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result_vec[i], 1*sizeof(dType)),"GPU memory allocation failed\n");
		}
	}

	cudaSetDevice(layer_info.device_number);
	full_matrix_setup(&h_temp,&d_top_param_rev,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_top_param_nonrev,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_top_bias,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_top_param_rev_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_top_param_nonrev_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_top_bias_grad,LSTM_size,1);

	full_matrix_setup(&h_temp,&d_temp_error_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp_error_2,LSTM_size,minibatch_size);

	full_vector_setup_ones(&h_temp,&d_ones_minibatch,minibatch_size);

	for(int i=0; i<longest_sent; i++) {
		d_ht_rev_total.push_back(NULL);
		d_ht_nonrev_total.push_back(NULL);
		d_ht_rev_total_errors.push_back(NULL);
		d_ht_nonrev_total_errors.push_back(NULL);
		d_final_mats.push_back(NULL);
		d_final_errors.push_back(NULL);
		full_matrix_setup(&h_temp,&d_ht_rev_total[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_ht_nonrev_total[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_final_mats[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_final_errors[i],LSTM_size,minibatch_size);
	}

	h_source_indicies = (int *)malloc(longest_sent*minibatch_size*sizeof(int));
	h_source_indicies_mask = (int *)malloc(longest_sent*minibatch_size*sizeof(int));
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_source_indicies_mask, longest_sent*minibatch_size*sizeof(int)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");

	thrust_d_top_param_rev_grad = thrust::device_pointer_cast(d_top_param_rev_grad);
	thrust_d_top_param_nonrev_grad = thrust::device_pointer_cast(d_top_param_nonrev_grad);
	thrust_d_top_bias_grad = thrust::device_pointer_cast(d_top_bias_grad);

	clear_gradients();
}


template<typename dType>
void bi_encoder<dType>::clear_gradients() {

	cudaSetDevice(layer_info.device_number);

	cudaMemset(d_top_param_rev_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_top_param_nonrev_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_top_bias_grad,0,LSTM_size*1*sizeof(dType));

	for(int i=0; i<longest_sent; i++) {
		cudaMemset(d_final_errors[i],0,LSTM_size*minibatch_size*sizeof(dType));
	}

	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);
			cudaMemset(d_horiz_param_rev_grad[i],0,LSTM_size*LSTM_size*sizeof(dType));
			cudaMemset(d_horiz_param_nonrev_grad[i],0,LSTM_size*LSTM_size*sizeof(dType));
			cudaMemset(d_horiz_bias_grad[i],0,LSTM_size*1*sizeof(dType));

			//cudaMemset(d_horiz_param_rev_ct_grad[i],0,LSTM_size*LSTM_size*sizeof(dType));
			//cudaMemset(d_horiz_param_nonrev_ct_grad[i],0,LSTM_size*LSTM_size*sizeof(dType));
		}
	}
}


template<typename dType>
void bi_encoder<dType>::dump_weights(std::ofstream &output) {
	cudaSetDevice(layer_info.device_number);

	write_matrix_GPU(d_top_param_rev,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_top_param_nonrev,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_top_bias,LSTM_size,1,output);

	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);
			write_matrix_GPU(d_horiz_param_rev[i],LSTM_size,LSTM_size,output);
			write_matrix_GPU(d_horiz_param_nonrev[i],LSTM_size,LSTM_size,output);
			write_matrix_GPU(d_horiz_bias[i],LSTM_size,1,output);

			//write_matrix_GPU(d_horiz_param_rev_ct[i],LSTM_size,LSTM_size,output);
			//write_matrix_GPU(d_horiz_param_nonrev_ct[i],LSTM_size,LSTM_size,output);
		}
	}
}

template<typename dType>
void bi_encoder<dType>::load_weights(std::ifstream &input) {
	cudaSetDevice(layer_info.device_number);

	read_matrix_GPU(d_top_param_rev,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_top_param_nonrev,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_top_bias,LSTM_size,1,input);

	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);
			read_matrix_GPU(d_horiz_param_rev[i],LSTM_size,LSTM_size,input);
			read_matrix_GPU(d_horiz_param_nonrev[i],LSTM_size,LSTM_size,input);
			read_matrix_GPU(d_horiz_bias[i],LSTM_size,1,input);

			//read_matrix_GPU(d_horiz_param_rev_ct[i],LSTM_size,LSTM_size,input);
			//read_matrix_GPU(d_horiz_param_nonrev_ct[i],LSTM_size,LSTM_size,input);
		}
	}
}


template<typename dType>
void bi_encoder<dType>::calculate_global_norm() {

	cudaSetDevice(layer_info.device_number);

	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_top_param_rev_grad,thrust_d_top_param_rev_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_top_param_nonrev_grad,thrust_d_top_param_nonrev_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_top_bias_grad,thrust_d_top_bias_grad + LSTM_size*1,unary_op);

	norm_clip_GPU_v2_p1(thrust_d_top_param_rev_grad,d_top_param_rev_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_top_param_nonrev_grad,d_top_param_nonrev_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_top_bias_grad,d_top_bias_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);


	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);

			thrust::device_ptr<dType> thrust_d_horiz_param_rev_grad = thrust::device_pointer_cast(d_horiz_param_rev_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_param_nonrev_grad = thrust::device_pointer_cast(d_horiz_param_nonrev_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_bias_grad = thrust::device_pointer_cast(d_horiz_bias_grad[i]);
			//thrust::device_ptr<dType> thrust_d_horiz_param_rev_ct_grad = thrust::device_pointer_cast(d_horiz_param_rev_ct_grad[i]);
			//thrust::device_ptr<dType> thrust_d_horiz_param_nonrev_ct_grad = thrust::device_pointer_cast(d_horiz_param_nonrev_ct_grad[i]);

			thrust::for_each(thrust_d_horiz_param_rev_grad,thrust_d_horiz_param_rev_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_param_nonrev_grad,thrust_d_horiz_param_nonrev_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_bias_grad,thrust_d_horiz_bias_grad + LSTM_size*1,unary_op);
			//thrust::for_each(thrust_d_horiz_param_rev_ct_grad,thrust_d_horiz_param_rev_ct_grad + LSTM_size*LSTM_size,unary_op);
			//thrust::for_each(thrust_d_horiz_param_nonrev_ct_grad,thrust_d_horiz_param_nonrev_ct_grad + LSTM_size*LSTM_size,unary_op);

			norm_clip_GPU_v2_p1(thrust_d_horiz_param_rev_grad,d_horiz_param_rev_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2_p1(thrust_d_horiz_param_nonrev_grad,d_horiz_param_nonrev_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2_p1(thrust_d_horiz_bias_grad,d_horiz_bias_grad[i],norm_clip,LSTM_size*1,d_temp_result_vec[i],d_result_vec[i]);
			//norm_clip_GPU_v2_p1(thrust_d_horiz_param_rev_ct_grad,d_horiz_param_rev_ct_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			//norm_clip_GPU_v2_p1(thrust_d_horiz_param_nonrev_ct_grad,d_horiz_param_nonrev_ct_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
		}
	}

	devSynchAll();
}

template<typename dType>
void bi_encoder<dType>::update_global_params() {

	cudaSetDevice(layer_info.device_number);

	norm_clip_GPU_v2_p2(thrust_d_top_param_rev_grad,d_top_param_rev_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_top_param_nonrev_grad,d_top_param_nonrev_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_top_bias_grad,d_top_bias_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);

	gradient_update_mats<<<256,256,0,layer_info.s0>>>(d_top_param_rev,d_top_param_rev_grad,learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256,0,layer_info.s0>>>(d_top_param_nonrev,d_top_param_nonrev_grad,learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256,0,layer_info.s0>>>(d_top_bias,d_top_bias_grad,learning_rate,LSTM_size*1);

	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);

			thrust::device_ptr<dType> thrust_d_horiz_param_rev_grad = thrust::device_pointer_cast(d_horiz_param_rev_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_param_nonrev_grad = thrust::device_pointer_cast(d_horiz_param_nonrev_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_bias_grad = thrust::device_pointer_cast(d_horiz_bias_grad[i]);
			//thrust::device_ptr<dType> thrust_d_horiz_param_rev_ct_grad = thrust::device_pointer_cast(d_horiz_param_rev_ct_grad[i]);
			//thrust::device_ptr<dType> thrust_d_horiz_param_nonrev_ct_grad = thrust::device_pointer_cast(d_horiz_param_nonrev_ct_grad[i]);

			norm_clip_GPU_v2_p2(thrust_d_horiz_param_rev_grad,d_horiz_param_rev_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2_p2(thrust_d_horiz_param_nonrev_grad,d_horiz_param_nonrev_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2_p2(thrust_d_horiz_bias_grad,d_horiz_bias_grad[i],norm_clip,LSTM_size*1,d_temp_result_vec[i],d_result_vec[i]);
			//norm_clip_GPU_v2_p2(thrust_d_horiz_param_rev_ct_grad,d_horiz_param_rev_ct_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			//norm_clip_GPU_v2_p2(thrust_d_horiz_param_nonrev_ct_grad,d_horiz_param_nonrev_ct_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);

			gradient_update_mats<<<256,256>>>(d_horiz_param_rev[i],d_horiz_param_rev_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_param_nonrev[i],d_horiz_param_nonrev_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_bias[i],d_horiz_bias_grad[i],learning_rate,LSTM_size*1);
			//gradient_update_mats<<<256,256>>>(d_horiz_param_rev_ct[i],d_horiz_param_rev_ct_grad[i],learning_rate,LSTM_size*LSTM_size);
			//gradient_update_mats<<<256,256>>>(d_horiz_param_nonrev_ct[i],d_horiz_param_nonrev_ct_grad[i],learning_rate,LSTM_size*LSTM_size);
		}
	}

	devSynchAll();
}

template<typename dType>
void bi_encoder<dType>::update_weights() {

	cudaSetDevice(layer_info.device_number);

	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_top_param_rev_grad,thrust_d_top_param_rev_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_top_param_nonrev_grad,thrust_d_top_param_nonrev_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_top_bias_grad,thrust_d_top_bias_grad + LSTM_size*1,unary_op);

	norm_clip_GPU_v2(thrust_d_top_param_rev_grad,d_top_param_rev_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_top_param_nonrev_grad,d_top_param_nonrev_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_top_bias_grad,d_top_bias_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);

	gradient_update_mats<<<256,256,0,layer_info.s0>>>(d_top_param_rev,d_top_param_rev_grad,learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256,0,layer_info.s0>>>(d_top_param_nonrev,d_top_param_nonrev_grad,learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256,0,layer_info.s0>>>(d_top_bias,d_top_bias_grad,learning_rate,LSTM_size*1);

	if(model_type == COMBINE) {
		for(int i=0; i<num_layers; i++) {
			cudaSetDevice(gpu_indicies[i]);

			thrust::device_ptr<dType> thrust_d_horiz_param_rev_grad = thrust::device_pointer_cast(d_horiz_param_rev_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_param_nonrev_grad = thrust::device_pointer_cast(d_horiz_param_nonrev_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_bias_grad = thrust::device_pointer_cast(d_horiz_bias_grad[i]);
			//thrust::device_ptr<dType> thrust_d_horiz_param_rev_ct_grad = thrust::device_pointer_cast(d_horiz_param_rev_ct_grad[i]);
			//thrust::device_ptr<dType> thrust_d_horiz_param_nonrev_ct_grad = thrust::device_pointer_cast(d_horiz_param_nonrev_ct_grad[i]);

			thrust::for_each(thrust_d_horiz_param_rev_grad,thrust_d_horiz_param_rev_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_param_nonrev_grad,thrust_d_horiz_param_nonrev_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_bias_grad,thrust_d_horiz_bias_grad + LSTM_size*1,unary_op);
			//thrust::for_each(thrust_d_horiz_param_rev_ct_grad,thrust_d_horiz_param_rev_ct_grad + LSTM_size*LSTM_size,unary_op);
			//thrust::for_each(thrust_d_horiz_param_nonrev_ct_grad,thrust_d_horiz_param_nonrev_ct_grad + LSTM_size*LSTM_size,unary_op);

			norm_clip_GPU_v2(thrust_d_horiz_param_rev_grad,d_horiz_param_rev_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2(thrust_d_horiz_param_nonrev_grad,d_horiz_param_nonrev_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2(thrust_d_horiz_bias_grad,d_horiz_bias_grad[i],norm_clip,LSTM_size*1,d_temp_result_vec[i],d_result_vec[i]);
			//norm_clip_GPU_v2(thrust_d_horiz_param_rev_ct_grad,d_horiz_param_rev_ct_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			//norm_clip_GPU_v2(thrust_d_horiz_param_nonrev_ct_grad,d_horiz_param_nonrev_ct_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);

			gradient_update_mats<<<256,256>>>(d_horiz_param_rev[i],d_horiz_param_rev_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_param_nonrev[i],d_horiz_param_nonrev_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_bias[i],d_horiz_bias_grad[i],learning_rate,LSTM_size*1);
			//gradient_update_mats<<<256,256>>>(d_horiz_param_rev_ct[i],d_horiz_param_rev_ct_grad[i],learning_rate,LSTM_size*LSTM_size);
			//gradient_update_mats<<<256,256>>>(d_horiz_param_nonrev_ct[i],d_horiz_param_nonrev_ct_grad[i],learning_rate,LSTM_size*LSTM_size);
		}
	}

	devSynchAll();
}


template<typename dType>
void bi_encoder<dType>::reverse_indicies(int *h_vocab_indices,int len) {

	//first reverse all the indicies
	// std::cout << "Printing initial indicies\n";
	// for(int i=0; i<len*minibatch_size; i++) {
	// 	std::cout << h_vocab_indices[i] << "\n";
	// }

	for(int i=0; i<len*minibatch_size; i++) {
		h_source_indicies[i] = h_vocab_indices[len*minibatch_size - i - 1];
		if(h_vocab_indices[i]==-1) {
			h_source_indicies_mask[i] = 0;
		}
		else {
			h_source_indicies_mask[i] = 1;
		}	
	}

	cudaMemcpy(d_source_indicies_mask, h_source_indicies_mask, minibatch_size*len*sizeof(int), cudaMemcpyHostToDevice);

	//now reverse per minibatch
	for(int i=0; i<len; i++) {
		int low_index = IDX2C(0,i,minibatch_size);
		int high_index = IDX2C(minibatch_size-1,i,minibatch_size);
		while(low_index<=high_index) {
			int temp = h_source_indicies[low_index];
			h_source_indicies[low_index] = h_source_indicies[high_index];
			h_source_indicies[high_index] = temp;
			low_index++;
			high_index--;
		}
	}


	for(int i=0; i<num_layers; i++) {
		final_index_hs[i] = -1;
	}

	//fill the final_index_hs
	for(int i=0; i<len; i++) {
		for(int j=0; j<minibatch_size; j++) {
			if(final_index_hs[j]==-1) {
				if(h_source_indicies[IDX2C(j,i,minibatch_size)]==-1) {
					final_index_hs[j] = i-1;
				}
			}
		}
	}

	for(int j=0; j<minibatch_size; j++) {
		if(final_index_hs[j]==-1) {
			final_index_hs[j] = len-1;
		}
	}

	// std::cout << "Printing new indicies\n";
	// for(int i=0; i<len*minibatch_size; i++) {
	// 	std::cout << h_source_indicies[i] << " ";
	// }
	// std::cout << "\n";

	// std::cout << "Printing swap indicies\n";
	// for(int i=0; i<minibatch_size; i++) {
	// 	std::cout << final_index_hs[i] << " ";
	// }
	// std::cout << "\n";

}


template<typename dType>
void bi_encoder<dType>::forward_prop() {

	cudaSetDevice(layer_info.device_number);

	// std::cout << "Printing all rev hiddenstates\n";
	// for(int i=0; i<longest_sent_minibatch; i++) {
	// 	std::cout << "index: " << i << "\n";
	// 	print_GPU_Matrix(model->input_layer_source.nodes[i].d_h_t,LSTM_size,minibatch_size);
	// }

	// std::cout << "Printing all nonrev hiddenstates\n";
	// for(int i=0; i<longest_sent_minibatch; i++) {
	// 	std::cout << "index: " << i << "\n";
	// 	print_GPU_Matrix(model->input_layer_source_bi.nodes[i].d_h_t,LSTM_size,minibatch_size);
	// }

	//transform the top layer
	for(int i=0; i<longest_sent_minibatch; i++) {
		dType *d_rev_hs = d_ht_rev_total[i]; //hiddenstates from reversed LSTM
		dType *d_nonrev_hs = d_ht_nonrev_total[(longest_sent_minibatch-i-1)]; //hiddenstates from nonreversed LSTM
		dType *d_final_mat_temp = d_final_mats[i];
		int *d_mask = d_source_indicies_mask + i*minibatch_size;

		dType alpha = 1;
		dType beta = 0;

		//multiply the two hiddenstates by a matrix
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_top_param_rev,LSTM_size,
			d_rev_hs,LSTM_size,&beta,d_final_mat_temp,LSTM_size),"Bi directional forward failed p1\n");

		beta = 1;
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_top_param_nonrev,LSTM_size,
			d_nonrev_hs,LSTM_size,&beta,d_final_mat_temp,LSTM_size),"Bi directional forward failed p2\n");

		//add the bias and send through tanh
		tanh_bi_forward_kernel<<<256,256,0,layer_info.s0>>>(d_final_mat_temp,d_top_bias,LSTM_size,minibatch_size);

		//now zero out this value
		zero_h_t<<<256,256,0,layer_info.s0>>>(d_final_mat_temp,d_mask,LSTM_size,minibatch_size);
	}

	//tranform the hiddenstate going into the decoder
	if(model_type == COMBINE) {

		//now copy all the correct hidden and cell states
		for(int j=0; j<minibatch_size; j++) {
			cudaMemcpy(d_hs_start_target[0]+j*LSTM_size,model->input_layer_source_bi.nodes[final_index_hs[j]].d_h_t+j*LSTM_size,LSTM_size*sizeof(dType),cudaMemcpyDefault);
			cudaMemcpy(d_ct_start_target[0]+j*LSTM_size,model->input_layer_source_bi.nodes[final_index_hs[j]].d_c_t+j*LSTM_size,LSTM_size*sizeof(dType),cudaMemcpyDefault);
		}
		for(int i=1; i<num_layers; i++) {
			for(int j=0; j<minibatch_size; j++) {
				cudaMemcpy(d_hs_start_target[i]+j*LSTM_size,model->source_hidden_layers_bi[i-1].nodes[final_index_hs[j]].d_h_t+j*LSTM_size,LSTM_size*sizeof(dType),cudaMemcpyDefault);
				cudaMemcpy(d_ct_start_target[i]+j*LSTM_size,model->source_hidden_layers_bi[i-1].nodes[final_index_hs[j]].d_c_t+j*LSTM_size,LSTM_size*sizeof(dType),cudaMemcpyDefault);
			}
		}

		for(int i=0; i<num_layers; i++) {
			dType alpha = 1;
			dType beta = 0;
			cudaSetDevice(gpu_indicies[i]);
			cublasHandle_t temp_handle;

			if(i==0) {
				temp_handle = model->input_layer_source.ih_layer_info.handle;
				cublasSetStream(temp_handle,NULL);
				CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_rev[i],LSTM_size,
					model->input_layer_source.nodes[longest_sent_minibatch-1].d_h_t,LSTM_size,&beta,d_hs_final_target[i],LSTM_size),"Bi directional forward failed p1\n");
				

				// cublasSetStream(temp_handle,NULL);
				// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_rev_ct[i],LSTM_size,
				// 	model->input_layer_source.nodes[longest_sent_minibatch-1].d_c_t,LSTM_size,&beta,d_ct_final_target[i],LSTM_size),"Bi directional forward failed p2\n");
			
				add_two_mats<<<256,256>>>(d_ct_final_target[i],model->input_layer_source.nodes[longest_sent_minibatch-1].d_c_t,d_ct_start_target[i],LSTM_size*minibatch_size);
			}
			else {
				temp_handle = model->source_hidden_layers[i-1].hh_layer_info.handle;
				cublasSetStream(temp_handle,NULL);
				CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_rev[i],LSTM_size,
					model->source_hidden_layers[i-1].nodes[longest_sent_minibatch-1].d_h_t,LSTM_size,&beta,d_hs_final_target[i],LSTM_size),"Bi directional forward failed p1\n");


				// cublasSetStream(temp_handle,NULL);
				// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_rev_ct[i],LSTM_size,
				// 	model->source_hidden_layers[i-1].nodes[longest_sent_minibatch-1].d_c_t,LSTM_size,&beta,d_ct_final_target[i],LSTM_size),"Bi directional forward failed p2\n");
			
				add_two_mats<<<256,256>>>(d_ct_final_target[i],	model->source_hidden_layers[i-1].nodes[longest_sent_minibatch-1].d_c_t,d_ct_start_target[i],LSTM_size*minibatch_size);
			}

			beta = 1;
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_nonrev[i],LSTM_size,
				d_hs_start_target[i],LSTM_size,&beta,d_hs_final_target[i],LSTM_size),"Bi directional forward failed p2\n");


			//add the bias and send through tanh
			tanh_bi_forward_kernel<<<256,256>>>(d_hs_final_target[i],d_horiz_bias[i],LSTM_size,minibatch_size);

			// //now combine the cells
			// cublasSetStream(temp_handle,NULL);
			// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_nonrev_ct[i],LSTM_size,
			// 	d_ct_start_target[i],LSTM_size,&beta,d_ct_final_target[i],LSTM_size),"Bi directional forward failed p2\n");

			devSynchAll();
		}
	}

	// std::cout << "------------Printing all source nonrev hiddenstates----------\n";
	// for(int i=0; i<longest_sent_minibatch; i++) {
	// 	std::cout << "index: " << i << "\n";
	// 	print_GPU_Matrix(model->input_layer_source_bi.nodes[i].d_h_t,LSTM_size,minibatch_size);
	// }

	// std::cout << "---------------printing combined hiddenstates---------------\n";
	// print_GPU_Matrix(d_hs_start_target[0],LSTM_size,minibatch_size);

	// std::cout << "------------Printing all source nonrev cellstates----------\n";
	// for(int i=0; i<longest_sent_minibatch; i++) {
	// 	std::cout << "index: " << i << "\n";
	// 	print_GPU_Matrix(model->input_layer_source_bi.nodes[i].d_c_t,LSTM_size,minibatch_size);
	// }

	// std::cout << "---------------printing combined cellstates---------------\n";
	// print_GPU_Matrix(d_ct_start_target[0],LSTM_size,minibatch_size);
}


template<typename dType>
void bi_encoder<dType>::back_prop() {

	cudaSetDevice(layer_info.device_number);
	//The attention model will pool errors
	dType alpha;
	dType beta;

	//calculate errors with respect to the two matrices and bias for each position along with the rev and nonrev hiddenstates
	for(int i=0; i<longest_sent_minibatch; i++) {

		dType *d_final_mat_temp = d_final_mats[i];
		dType *d_errors_temp = d_final_errors[i];
		dType *d_rev_hs = d_ht_rev_total[i]; //hiddenstates from reversed LSTM
		dType *d_nonrev_hs = d_ht_nonrev_total[(longest_sent_minibatch-i-1)]; //hiddenstates from nonreversed LSTM
		int *d_mask = d_source_indicies_mask + i*minibatch_size;


		//zero out the error
		zero_h_t<<<256,256,0,layer_info.s0>>>(d_errors_temp,d_mask,LSTM_size,minibatch_size);

		alpha = 1;
		beta = 1;
		//multiply the gradient coming down by 1-tanh()^2
		tanh_grad_kernel<<<256,256,0,layer_info.s0>>>(d_errors_temp,d_errors_temp,d_final_mat_temp,LSTM_size*minibatch_size);
		CUDA_GET_LAST_ERROR("Bidirectional tanh grad");

		//calculate gradient with respect to d_top_param_rev
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			d_errors_temp,LSTM_size,d_rev_hs,LSTM_size,&beta,d_top_param_rev_grad,LSTM_size),"Attention backprop W_c_1 grad\n");


		//calculate gradient with respect to d_top_param_nonrev
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			d_errors_temp,LSTM_size,d_nonrev_hs,LSTM_size,&beta,d_top_param_nonrev_grad,LSTM_size),"Attention backprop W_c_2 grad\n");


		//calculate gradient with respect to d_top_bias
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_errors_temp,LSTM_size,
			d_ones_minibatch,1,&beta,d_top_bias_grad,1),"backprop b_i_grad failed\n");

		alpha = 1;
		beta = 0;
		//calculate error with respect to h_t_rev
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha,d_top_param_rev,LSTM_size,d_errors_temp,LSTM_size,&beta,d_temp_error_1,LSTM_size),"Attention backprop d_ERRnTOt_ct\n");


		//calculate error with respect to h_t_nonrev
		cublasSetStream(layer_info.handle,layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha,d_top_param_nonrev,LSTM_size,d_errors_temp,LSTM_size,&beta,d_temp_error_2,LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");
	

		//now copy errors 1 and 2 to their respective hidden states
		cudaMemcpyAsync(d_ht_rev_total_errors[i],d_temp_error_1,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,layer_info.s0);
		cudaMemcpyAsync(d_ht_nonrev_total_errors[(longest_sent_minibatch-i-1)],d_temp_error_2,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,layer_info.s0);
	}

	if(model_type == COMBINE) {

		for(int i=0; i<num_layers; i++) {

			cudaSetDevice(gpu_indicies[i]);
			dType *d_errors_temp;
			dType *d_ct_errors_temp;
			cublasHandle_t temp_handle;
			dType *d_hs_final_target_temp = d_hs_final_target[i];
			dType *d_rev_hs;
			//dType *d_rev_ct;
			dType *d_nonrev_hs = d_hs_start_target[i];
			dType *d_nonrev_ct = d_ct_start_target[i];
			dType *d_ones_minibatch_temp;

			dType alpha = 1;
			dType beta = 1;

			if(i==0) {
				temp_handle = model->input_layer_source.ih_layer_info.handle;
				d_errors_temp = model->input_layer_target.d_d_ERRnTOt_htM1;
				d_ct_errors_temp = model->input_layer_target.d_d_ERRnTOt_ctM1;
				d_rev_hs = model->input_layer_source.nodes[longest_sent_minibatch-1].d_h_t;
				//d_rev_ct = model->input_layer_source.nodes[longest_sent_minibatch-1].d_c_t;
				d_ones_minibatch_temp = model->input_layer_source.d_ones_minibatch;
			}
			else {
				temp_handle = model->target_hidden_layers[i-1].hh_layer_info.handle;
				d_errors_temp = model->target_hidden_layers[i-1].d_d_ERRnTOt_htM1;
				d_ct_errors_temp = model->target_hidden_layers[i-1].d_d_ERRnTOt_ctM1;
				d_rev_hs = model->source_hidden_layers[i-1].nodes[longest_sent_minibatch-1].d_h_t;
				//d_rev_ct = model->source_hidden_layers[i-1].nodes[longest_sent_minibatch-1].d_c_t;
				d_ones_minibatch_temp = model->source_hidden_layers[i-1].d_ones_minibatch;
			}

			//multiply the gradient coming down by 1-tanh()^2
			tanh_grad_kernel<<<256,256>>>(d_errors_temp,d_errors_temp,d_hs_final_target_temp,LSTM_size*minibatch_size);
			CUDA_GET_LAST_ERROR("Bidirectional tanh grad");

			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
				d_errors_temp,LSTM_size,d_rev_hs,LSTM_size,&beta,d_horiz_param_rev_grad[i],LSTM_size),"Attention backprop W_c_1 grad\n");


			//calculate gradient with respect to d_top_param_nonrev
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
				d_errors_temp,LSTM_size,d_nonrev_hs,LSTM_size,&beta,d_horiz_param_nonrev_grad[i],LSTM_size),"Attention backprop W_c_2 grad\n");


			//calculate gradient with respect to d_top_bias
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(temp_handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_errors_temp,LSTM_size,
				d_ones_minibatch_temp,1,&beta,d_horiz_bias_grad[i],1),"backprop b_i_grad failed\n");

			alpha = 1;
			beta = 0;
			//calculate error with respect to h_t_rev
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
				&alpha,d_horiz_param_rev[i],LSTM_size,d_errors_temp,LSTM_size,&beta,d_hs_rev_error_horiz[i],LSTM_size),"Attention backprop d_ERRnTOt_ct\n");


			//calculate error with respect to h_t_nonrev
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
				&alpha,d_horiz_param_nonrev[i],LSTM_size,d_errors_temp,LSTM_size,&beta,d_hs_nonrev_error_horiz[i],LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");

			//-------------------------------------now the errors for combining the cell---------------------------------------

			beta = 1;
			// cublasSetStream(temp_handle,NULL);
			// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			// 	d_ct_errors_temp,LSTM_size,d_rev_ct,LSTM_size,&beta,d_horiz_param_rev_ct_grad[i],LSTM_size),"Attention backprop W_c_1 grad\n");

			// cublasSetStream(temp_handle,NULL);
			// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			// 	d_ct_errors_temp,LSTM_size,d_nonrev_ct,LSTM_size,&beta,d_horiz_param_nonrev_ct_grad[i],LSTM_size),"Attention backprop W_c_2 grad\n");

			beta = 0;
			// cublasSetStream(temp_handle,NULL);
			// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			// 	&alpha,d_horiz_param_rev_ct[i],LSTM_size,d_ct_errors_temp,LSTM_size,&beta,d_ct_rev_error_horiz[i],LSTM_size),"Attention backprop d_ERRnTOt_ct\n");

			// cublasSetStream(temp_handle,NULL);
			// CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			// 	&alpha,d_horiz_param_nonrev_ct[i],LSTM_size,d_ct_errors_temp,LSTM_size,&beta,d_ct_nonrev_error_horiz[i],LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");
			
			cudaMemcpy(d_ct_rev_error_horiz[i], d_ct_errors_temp, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_ct_nonrev_error_horiz[i], d_ct_errors_temp, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);

			devSynchAll();
		}
	}
}





template<typename dType>
void bi_encoder<dType>::check_all_gradients(dType epsilon) {

	cudaSetDevice(layer_info.device_number);

	std::cout << "--------------------GRADIENT CHECKING FOR BI-DIR LAYER GPU-------------------------\n";
	std::cout << "GRADIENT CHECKING FOR \n";
		
	std::cout << "GRADIENT CHECKING FOR top_param_rev_grad\n";
	check_gradient_GPU(epsilon,d_top_param_rev,d_top_param_rev_grad,LSTM_size,LSTM_size);
	cudaSetDevice(layer_info.device_number);

	std::cout << "GRADIENT CHECKING FOR top_param_nonrev_grad\n";
	check_gradient_GPU(epsilon,d_top_param_nonrev,d_top_param_nonrev_grad,LSTM_size,LSTM_size);
	cudaSetDevice(layer_info.device_number);

	std::cout << "GRADIENT CHECKING FOR top_bias_grad\n";
	check_gradient_GPU(epsilon,d_top_bias,d_top_bias_grad,LSTM_size,1);
	cudaSetDevice(layer_info.device_number);

	if(model_type == COMBINE) {
		std::cout << "--------------------GRADIENT CHECKING FOR BI-DIR (COMBINE) LAYER GPU-------------------------\n";
		for(int i=0; i<num_layers; i++) {

			std::cout << "--------------------------CURRENT LAYER " << i+1 << " ---------------------------------\n";
			cudaSetDevice(gpu_indicies[i]);
			std::cout << "GRADIENT CHECKING FOR horiz_param_rev_grad\n";
			check_gradient_GPU(epsilon,d_horiz_param_rev[i],d_horiz_param_rev_grad[i],LSTM_size,LSTM_size);

			cudaSetDevice(gpu_indicies[i]);
			std::cout << "GRADIENT CHECKING FOR horiz_param_nonrev_grad\n";
			check_gradient_GPU(epsilon,d_horiz_param_nonrev[i],d_horiz_param_nonrev_grad[i],LSTM_size,LSTM_size);

			cudaSetDevice(gpu_indicies[i]);
			std::cout << "GRADIENT CHECKING FOR horiz_bias_grad\n";
			check_gradient_GPU(epsilon,d_horiz_bias[i],d_horiz_bias_grad[i],LSTM_size,1);

			// cudaSetDevice(gpu_indicies[i]);
			// std::cout << "GRADIENT CHECKING FOR horiz_param_rev_ct_grad\n";
			// check_gradient_GPU(epsilon,d_horiz_param_rev_ct[i],d_horiz_param_rev_ct_grad[i],LSTM_size,LSTM_size);

			// cudaSetDevice(gpu_indicies[i]);
			// std::cout << "GRADIENT CHECKING FOR horiz_param_nonrev_ct_grad\n";
			// check_gradient_GPU(epsilon,d_horiz_param_nonrev_ct[i],d_horiz_param_nonrev_ct_grad[i],LSTM_size,LSTM_size);
		}
	}
}




template<typename dType>
void bi_encoder<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {

	cudaSetDevice(layer_info.device_number);
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaSetDevice(layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaSetDevice(layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			//std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
			else if(d_thrust_grad[IDX2C(i,j,rows)]==0 ||loss/(2*epsilon) ==0) {
				std::cout << "ZERO GRADIENTS\n";
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}





