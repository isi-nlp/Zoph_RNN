template<typename dType>
encoder_multi_source<dType>::encoder_multi_source() {

}


template<typename dType>
void encoder_multi_source<dType>::init_layer(global_params &params,neuralMT_model<dType> *model,std::vector<int> &gpu_indicies) {

	this->num_layers = params.num_layers;
	this->LSTM_size = params.LSTM_size;
	this->minibatch_size = params.minibatch_size;
	this->model = model;
	this->norm_clip = norm_clip;
	this->learning_rate = learning_rate;
	this->gpu_indicies = gpu_indicies;
	this->lstm_combine = params.multi_src_params.lstm_combine;


	dType *h_temp;
	for(int i=0; i<num_layers; i++) {
		d_hs_final_target.push_back(NULL);
		d_horiz_param_s1.push_back(NULL); //for transforming the top indicies
		d_horiz_param_s2.push_back(NULL); //for transforming the top indicies
		d_horiz_bias.push_back(NULL); //for transforming the top indicies
		d_horiz_param_s1_grad.push_back(NULL);
		d_horiz_param_s2_grad.push_back(NULL);
		d_horiz_bias_grad.push_back(NULL);
		d_hs_s1_error_horiz.push_back(NULL);
		d_hs_s2_error_horiz.push_back(NULL);
		d_ct_s1_error_horiz.push_back(NULL);
		d_ct_s2_error_horiz.push_back(NULL);
		d_ct_final_target.push_back(NULL);
		d_horiz_param_s1_ct.push_back(NULL); 
		d_horiz_param_s2_ct.push_back(NULL); 
		d_horiz_param_s1_ct_grad.push_back(NULL); 
		d_horiz_param_s2_ct_grad.push_back(NULL); 
		d_temp_result_vec.push_back(NULL);
		d_result_vec.push_back(NULL);
		lstm_combiner_layers.push_back(NULL);
	}

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);
		full_matrix_setup(&h_temp,&d_hs_final_target[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s1[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s2[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_bias[i],LSTM_size,1);
		full_matrix_setup(&h_temp,&d_horiz_param_s1_grad[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s2_grad[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_bias_grad[i],LSTM_size,1);
		full_matrix_setup(&h_temp,&d_hs_s1_error_horiz[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_hs_s2_error_horiz[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_ct_final_target[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_ct_s1_error_horiz[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_ct_s2_error_horiz[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s1_ct[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s2_ct[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s1_ct_grad[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s2_ct_grad[i],LSTM_size,LSTM_size);

		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result_vec[i], NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result_vec[i], 1*sizeof(dType)),"GPU memory allocation failed\n");

		if(lstm_combine) {
			lstm_combiner_layers[i] = new tree_LSTM<dType>(params,gpu_indicies[i],this);
		}
	}
	clear_gradients();
}

template<typename dType>
void encoder_multi_source<dType>::init_layer_decoder(neuralMT_model<dType> *model,int gpu_num,bool lstm_combine,int LSTM_size,int num_layers) {

	this->num_layers = num_layers;
	this->LSTM_size = LSTM_size;
	this->minibatch_size = 1;
	this->model = model;
	this->lstm_combine = lstm_combine;
	this->decode = true;

	for(int i=0; i<num_layers; i++) {
		gpu_indicies.push_back(gpu_num);
	}

	dType *h_temp;
	for(int i=0; i<num_layers; i++) {
		d_hs_final_target.push_back(NULL);
		d_horiz_param_s1.push_back(NULL); //for transforming the top indicies
		d_horiz_param_s2.push_back(NULL); //for transforming the top indicies
		d_horiz_bias.push_back(NULL); //for transforming the top indicies
		d_ct_final_target.push_back(NULL);
		d_horiz_param_s1_ct.push_back(NULL); 
		d_horiz_param_s2_ct.push_back(NULL); 
		d_temp_result_vec.push_back(NULL);
		d_result_vec.push_back(NULL);
		lstm_combiner_layers.push_back(NULL);
	}

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);
		full_matrix_setup(&h_temp,&d_hs_final_target[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s1[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s2[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_bias[i],LSTM_size,1);
		full_matrix_setup(&h_temp,&d_ct_final_target[i],LSTM_size,minibatch_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s1_ct[i],LSTM_size,LSTM_size);
		full_matrix_setup(&h_temp,&d_horiz_param_s2_ct[i],LSTM_size,LSTM_size);

		if(lstm_combine) {
			//special constructor for decoding
			lstm_combiner_layers[i] = new tree_LSTM<dType>(LSTM_size,gpu_num,this);
		}
	}
}



template<typename dType>
void encoder_multi_source<dType>::clear_gradients() {

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);

		if(lstm_combine) {
			lstm_combiner_layers[i]->clear_gradients();
		}
		else {
			cudaMemset(d_horiz_param_s1_grad[i],0,LSTM_size*LSTM_size*sizeof(dType));
			cudaMemset(d_horiz_param_s2_grad[i],0,LSTM_size*LSTM_size*sizeof(dType));
			cudaMemset(d_horiz_bias_grad[i],0,LSTM_size*1*sizeof(dType));
		}
	}
	devSynchAll();
}


template<typename dType>
void encoder_multi_source<dType>::dump_weights(std::ofstream &output) {

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);

		if(lstm_combine) {
			lstm_combiner_layers[i]->dump_weights(output);
		}
		else {
			write_matrix_GPU(d_horiz_param_s1[i],LSTM_size,LSTM_size,output);
			write_matrix_GPU(d_horiz_param_s2[i],LSTM_size,LSTM_size,output);
			write_matrix_GPU(d_horiz_bias[i],LSTM_size,1,output);
		}
	}
}

template<typename dType>
void encoder_multi_source<dType>::load_weights(std::ifstream &input) {

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);

		if(lstm_combine) {
			lstm_combiner_layers[i]->load_weights(input);
		}
		else {
			read_matrix_GPU(d_horiz_param_s1[i],LSTM_size,LSTM_size,input);
			read_matrix_GPU(d_horiz_param_s2[i],LSTM_size,LSTM_size,input);
			read_matrix_GPU(d_horiz_bias[i],LSTM_size,1,input);
		}

		//read_matrix_GPU(d_horiz_param_s1_ct[i],LSTM_size,LSTM_size,input);
		//read_matrix_GPU(d_horiz_param_s2_ct[i],LSTM_size,LSTM_size,input);
	}
}


template<typename dType>
void encoder_multi_source<dType>::calculate_global_norm() {

	scale_functor unary_op(minibatch_size);

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);
	
		if(lstm_combine) {
			lstm_combiner_layers[i]->calculate_global_norm();
		}
		else {
			//HPC_output << "----------------------- LAYER IN ENCODER MULTI SOURCE " << i << " -----------------------\n";
			thrust::device_ptr<dType> thrust_d_horiz_param_s1_grad = thrust::device_pointer_cast(d_horiz_param_s1_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_param_s2_grad = thrust::device_pointer_cast(d_horiz_param_s2_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_bias_grad = thrust::device_pointer_cast(d_horiz_bias_grad[i]);

			thrust::for_each(thrust_d_horiz_param_s1_grad,thrust_d_horiz_param_s1_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_param_s2_grad,thrust_d_horiz_param_s2_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_bias_grad,thrust_d_horiz_bias_grad + LSTM_size*1,unary_op);

			norm_clip_GPU_v2_p1(thrust_d_horiz_param_s1_grad,d_horiz_param_s1_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			//HPC_output << "----------------------- PRINTING GRAD NORM FOR d_horiz_param_s1_grad -----------------------\n";
			//HPC_output << BZ_CUDA::recent_sum << "\n";

			norm_clip_GPU_v2_p1(thrust_d_horiz_param_s2_grad,d_horiz_param_s2_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			//HPC_output << "----------------------- PRINTING GRAD NORM FOR d_horiz_param_s2_grad -----------------------\n";
			//HPC_output << BZ_CUDA::recent_sum << "\n";

			norm_clip_GPU_v2_p1(thrust_d_horiz_bias_grad,d_horiz_bias_grad[i],norm_clip,LSTM_size*1,d_temp_result_vec[i],d_result_vec[i]);
			// HPC_output << "----------------------- PRINTING GRAD NORM FOR d_horiz_bias_grad -----------------------\n";
			// HPC_output << BZ_CUDA::recent_sum << "\n";
			// HPC_output.flush();
		}
	}
	devSynchAll();
}

template<typename dType>
void encoder_multi_source<dType>::update_global_params() {


	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);
	
		if(lstm_combine) {
			lstm_combiner_layers[i]->update_global_params();
		}
		else {
			thrust::device_ptr<dType> thrust_d_horiz_param_s1_grad = thrust::device_pointer_cast(d_horiz_param_s1_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_param_s2_grad = thrust::device_pointer_cast(d_horiz_param_s2_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_bias_grad = thrust::device_pointer_cast(d_horiz_bias_grad[i]);
			
			norm_clip_GPU_v2_p2(thrust_d_horiz_param_s1_grad,d_horiz_param_s1_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2_p2(thrust_d_horiz_param_s2_grad,d_horiz_param_s2_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2_p2(thrust_d_horiz_bias_grad,d_horiz_bias_grad[i],norm_clip,LSTM_size*1,d_temp_result_vec[i],d_result_vec[i]);
		
			gradient_update_mats<<<256,256>>>(d_horiz_param_s1[i],d_horiz_param_s1_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_param_s2[i],d_horiz_param_s2_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_bias[i],d_horiz_bias_grad[i],learning_rate,LSTM_size*1);
		}
	}

	devSynchAll();
}

template<typename dType>
void encoder_multi_source<dType>::update_weights() {

	scale_functor unary_op(minibatch_size);

	for(int i=0; i<num_layers; i++) {
		cudaSetDevice(gpu_indicies[i]);

		if(lstm_combine) {
			lstm_combiner_layers[i]->update_weights();
		}
		else {
			thrust::device_ptr<dType> thrust_d_horiz_param_s1_grad = thrust::device_pointer_cast(d_horiz_param_s1_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_param_s2_grad = thrust::device_pointer_cast(d_horiz_param_s2_grad[i]);
			thrust::device_ptr<dType> thrust_d_horiz_bias_grad = thrust::device_pointer_cast(d_horiz_bias_grad[i]);

			thrust::for_each(thrust_d_horiz_param_s1_grad,thrust_d_horiz_param_s1_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_param_s2_grad,thrust_d_horiz_param_s2_grad + LSTM_size*LSTM_size,unary_op);
			thrust::for_each(thrust_d_horiz_bias_grad,thrust_d_horiz_bias_grad + LSTM_size*1,unary_op);
		
			norm_clip_GPU_v2(thrust_d_horiz_param_s1_grad,d_horiz_param_s1_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2(thrust_d_horiz_param_s2_grad,d_horiz_param_s2_grad[i],norm_clip,LSTM_size*LSTM_size,d_temp_result_vec[i],d_result_vec[i]);
			norm_clip_GPU_v2(thrust_d_horiz_bias_grad,d_horiz_bias_grad[i],norm_clip,LSTM_size*1,d_temp_result_vec[i],d_result_vec[i]);
	
			gradient_update_mats<<<256,256>>>(d_horiz_param_s1[i],d_horiz_param_s1_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_param_s2[i],d_horiz_param_s2_grad[i],learning_rate,LSTM_size*LSTM_size);
			gradient_update_mats<<<256,256>>>(d_horiz_bias[i],d_horiz_bias_grad[i],learning_rate,LSTM_size*1);
		}
	}
	devSynchAll();
}


template<typename dType>
void encoder_multi_source<dType>::forward_prop() {

	//tranform the hiddenstate and cellstate going into the decoder
	for(int i=0; i<num_layers; i++) {
		dType alpha = 1;
		dType beta = 0;
		cudaSetDevice(gpu_indicies[i]);
		cublasHandle_t temp_handle;

		dType *d_h_t_1;
		dType *d_h_t_2;
		dType *d_c_t_1;
		dType *d_c_t_2;

		if(decode) {
			if(i==0) {
				temp_handle = model->input_layer_source.ih_layer_info.handle;
				d_h_t_1 = model->input_layer_source.nodes[0].d_h_t;
				d_h_t_2 = model->input_layer_source_bi.nodes[0].d_h_t;
				d_c_t_1 = model->input_layer_source.nodes[0].d_c_t;
				d_c_t_2 = model->input_layer_source_bi.nodes[0].d_c_t;
			}
			else {
				temp_handle = model->source_hidden_layers[i-1].hh_layer_info.handle;
				d_h_t_1 = model->source_hidden_layers[i-1].nodes[0].d_h_t;
				d_h_t_2 = model->source_hidden_layers_bi[i-1].nodes[0].d_h_t;
				d_c_t_1 = model->source_hidden_layers[i-1].nodes[0].d_c_t;
				d_c_t_2 = model->source_hidden_layers_bi[i-1].nodes[0].d_c_t;
			}
		}
		else {
			if(i==0) {
				temp_handle = model->input_layer_source.ih_layer_info.handle;
				d_h_t_1 = model->input_layer_source.nodes[longest_sent_minibatch_s1-1].d_h_t;
				d_h_t_2 = model->input_layer_source_bi.nodes[longest_sent_minibatch_s2-1].d_h_t;
				d_c_t_1 = model->input_layer_source.nodes[longest_sent_minibatch_s1-1].d_c_t;
				d_c_t_2 = model->input_layer_source_bi.nodes[longest_sent_minibatch_s2-1].d_c_t;
			}
			else {
				temp_handle = model->source_hidden_layers[i-1].hh_layer_info.handle;
				d_h_t_1 = model->source_hidden_layers[i-1].nodes[longest_sent_minibatch_s1-1].d_h_t;
				d_h_t_2 = model->source_hidden_layers_bi[i-1].nodes[longest_sent_minibatch_s2-1].d_h_t;
				d_c_t_1 = model->source_hidden_layers[i-1].nodes[longest_sent_minibatch_s1-1].d_c_t;
				d_c_t_2 = model->source_hidden_layers_bi[i-1].nodes[longest_sent_minibatch_s2-1].d_c_t;
			}
		}

		if(lstm_combine) {

			//copy hidden and cells
			cudaMemcpy(lstm_combiner_layers[i]->d_child_ht_1, d_h_t_1, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(lstm_combiner_layers[i]->d_child_ht_2, d_h_t_2, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(lstm_combiner_layers[i]->d_child_ct_1, d_c_t_1, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(lstm_combiner_layers[i]->d_child_ct_2, d_c_t_2, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			devSynchAll();
			lstm_combiner_layers[i]->forward();
			devSynchAll();
			cudaMemcpy(d_hs_final_target[i], lstm_combiner_layers[i]->d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_ct_final_target[i], lstm_combiner_layers[i]->d_c_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
		}
		else {
		
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_s1[i],LSTM_size,
				d_h_t_1,LSTM_size,&beta,d_hs_final_target[i],LSTM_size),"Bi directional forward failed p1\n");
			
			beta = 1;
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_horiz_param_s2[i],LSTM_size,
				d_h_t_2,LSTM_size,&beta,d_hs_final_target[i],LSTM_size),"Bi directional forward failed p2\n");

			add_two_mats<<<256,256>>>(d_ct_final_target[i],d_c_t_1,d_c_t_2,LSTM_size*minibatch_size);

			//add the bias and send through tanh
			tanh_bi_forward_kernel<<<256,256>>>(d_hs_final_target[i],d_horiz_bias[i],LSTM_size,minibatch_size);
		}
		devSynchAll();
	}
}


template<typename dType>
void encoder_multi_source<dType>::back_prop() {


	for(int i=0; i<num_layers; i++) {

		cudaSetDevice(gpu_indicies[i]);
		dType *d_errors_temp;
		dType *d_ct_errors_temp;
		cublasHandle_t temp_handle;
		dType *d_hs_final_target_temp = d_hs_final_target[i];
		dType *d_s1_hs;
		//dType *d_s1_ct;
		dType *d_s2_hs;
		//dType *d_s2_ct;
		dType *d_ones_minibatch_temp;

		dType alpha = 1;
		dType beta = 1;


		if(i==0) {
			temp_handle = model->input_layer_source.ih_layer_info.handle;
			d_errors_temp = model->input_layer_target.d_d_ERRnTOt_htM1;
			d_ct_errors_temp = model->input_layer_target.d_d_ERRnTOt_ctM1;
			d_s1_hs = model->input_layer_source.nodes[longest_sent_minibatch_s1-1].d_h_t;
			//d_s1_ct = model->input_layer_source.nodes[longest_sent_minibatch_s1-1].d_c_t;
			d_ones_minibatch_temp = model->input_layer_source.d_ones_minibatch;

			d_s2_hs = model->input_layer_source_bi.nodes[longest_sent_minibatch_s2-1].d_h_t;
			//d_s2_ct = model->input_layer_source_bi.nodes[longest_sent_minibatch_s2-1].d_c_t;
		}
		else {
			temp_handle = model->target_hidden_layers[i-1].hh_layer_info.handle;
			d_errors_temp = model->target_hidden_layers[i-1].d_d_ERRnTOt_htM1;
			d_ct_errors_temp = model->target_hidden_layers[i-1].d_d_ERRnTOt_ctM1;
			d_s1_hs = model->source_hidden_layers[i-1].nodes[longest_sent_minibatch_s1-1].d_h_t;
			//d_s1_ct = model->source_hidden_layers[i-1].nodes[longest_sent_minibatch_s1-1].d_c_t;
			d_ones_minibatch_temp = model->source_hidden_layers[i-1].d_ones_minibatch;

			d_s2_hs = model->source_hidden_layers_bi[i-1].nodes[longest_sent_minibatch_s2-1].d_h_t;
			//d_s2_ct = model->source_hidden_layers_bi[i-1].nodes[longest_sent_minibatch_s2-1].d_c_t;
		}

		if(lstm_combine) {

			cudaMemcpy(lstm_combiner_layers[i]->d_d_ERRnTOt_ht, d_errors_temp, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(lstm_combiner_layers[i]->d_d_ERRnTOtp1_ct, d_ct_errors_temp, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			devSynchAll();
			lstm_combiner_layers[i]->backward();
			devSynchAll();
			cudaMemcpy(d_hs_s1_error_horiz[i], lstm_combiner_layers[i]->d_d_ERRnTOt_h1, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_hs_s2_error_horiz[i], lstm_combiner_layers[i]->d_d_ERRnTOt_h2, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_ct_s1_error_horiz[i], lstm_combiner_layers[i]->d_d_ERRnTOt_c1, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_ct_s2_error_horiz[i], lstm_combiner_layers[i]->d_d_ERRnTOt_c2, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
		}
		else {

			//multiply the gradient coming down by 1-tanh()^2
			tanh_grad_kernel<<<256,256>>>(d_errors_temp,d_errors_temp,d_hs_final_target_temp,LSTM_size*minibatch_size);
			CUDA_GET_LAST_ERROR("Bidirectional tanh grad");

			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
				d_errors_temp,LSTM_size,d_s1_hs,LSTM_size,&beta,d_horiz_param_s1_grad[i],LSTM_size),"Attention backprop W_c_1 grad\n");


			//calculate gradient with respect to d_top_param_nonrev
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
				d_errors_temp,LSTM_size,d_s2_hs,LSTM_size,&beta,d_horiz_param_s2_grad[i],LSTM_size),"Attention backprop W_c_2 grad\n");


			//calculate gradient with respect to d_top_bias
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(temp_handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_errors_temp,LSTM_size,
				d_ones_minibatch_temp,1,&beta,d_horiz_bias_grad[i],1),"backprop b_i_grad failed\n");

			alpha = 1;
			beta = 0;
			//calculate error with respect to h_t_rev
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
				&alpha,d_horiz_param_s1[i],LSTM_size,d_errors_temp,LSTM_size,&beta,d_hs_s1_error_horiz[i],LSTM_size),"Attention backprop d_ERRnTOt_ct\n");


			//calculate error with respect to h_t_nonrev
			cublasSetStream(temp_handle,NULL);
			CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(temp_handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
				&alpha,d_horiz_param_s2[i],LSTM_size,d_errors_temp,LSTM_size,&beta,d_hs_s2_error_horiz[i],LSTM_size),"Attention backprop d_ERRnTOt_h_t_p1\n");

			//-------------------------------------now the errors for combining the cell---------------------------------------

			beta = 1;
			
			beta = 0;
			cudaMemcpy(d_ct_s1_error_horiz[i], d_ct_errors_temp, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_ct_s2_error_horiz[i], d_ct_errors_temp, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDeviceToDevice);
		}
	
		devSynchAll();
	}
}





template<typename dType>
void encoder_multi_source<dType>::check_all_gradients(dType epsilon) {

	std::cout << "--------------------GRADIENT CHECKING FOR BI-DIR (COMBINE) LAYER GPU-------------------------\n";
	for(int i=0; i<num_layers; i++) {
	
		if(lstm_combine) {
			lstm_combiner_layers[i]->check_all_gradients(epsilon);
		}
		else {
			std::cout << "--------------------------CURRENT LAYER " << i+1 << " ---------------------------------\n";
			cudaSetDevice(gpu_indicies[i]);
			std::cout << "GRADIENT CHECKING FOR horiz_param_s1_grad\n";
			check_gradient_GPU(epsilon,d_horiz_param_s1[i],d_horiz_param_s1_grad[i],LSTM_size,LSTM_size,gpu_indicies[i]);

			cudaSetDevice(gpu_indicies[i]);
			std::cout << "GRADIENT CHECKING FOR horiz_param_s2_grad\n";
			check_gradient_GPU(epsilon,d_horiz_param_s2[i],d_horiz_param_s2_grad[i],LSTM_size,LSTM_size,gpu_indicies[i]);

			cudaSetDevice(gpu_indicies[i]);
			std::cout << "GRADIENT CHECKING FOR horiz_bias_grad\n";
			check_gradient_GPU(epsilon,d_horiz_bias[i],d_horiz_bias_grad[i],LSTM_size,1,gpu_indicies[i]);
		}
	}
}




template<typename dType>
void encoder_multi_source<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols,int gpu_index) {

	cudaSetDevice(gpu_index);
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaSetDevice(gpu_index);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaSetDevice(gpu_index);
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





