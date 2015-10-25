
template<typename dType>
attention_layer<dType>::attention_layer(int LSTM_size,int minibatch_size, int device_number, int D, int longest_sent,cublasHandle_t &handle,neuralMT_model<dType> *model,
		bool feed_input,bool clip_gradients,dType norm_clip) 
{
	this->handle = handle;
	this->model = model;
	this->device_number = device_number;
	this->LSTM_size = LSTM_size;
	this->minibatch_size = minibatch_size;
	this->clip_gradients = clip_gradients;
	this->norm_clip = norm_clip;
	this->feed_input = feed_input;
	this->longest_sent = longest_sent;


	cudaSetDevice(device_number);
	layer_info.init(device_number,D);
	dType *h_temp;
	full_matrix_setup(&h_temp,&d_W_a,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_W_p,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_v_p,1,LSTM_size);
	full_matrix_setup(&h_temp,&d_output_bias,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_W_c_p1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_W_c_p2,LSTM_size,LSTM_size);

	full_matrix_setup(&h_temp,&d_W_a_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_W_p_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_v_p_grad,1,LSTM_size);
	full_matrix_setup(&h_temp,&d_output_bias_grad,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_W_c_p1_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_W_c_p2_grad,LSTM_size,LSTM_size);


	full_matrix_setup(&h_temp,&d_ERRnTOt_tan_htild,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERRnTOt_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERRnTOt_ht_p1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERRnTOt_as,2*D+1,minibatch_size);
	full_matrix_setup(&h_temp,&d_ERRnTOt_pt,1,minibatch_size);

	full_matrix_setup(&h_temp,&d_temp_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_h_t_Wa_factor,2*D+1,minibatch_size);

	full_vector_setup_ones(&h_temp,&d_ones_minibatch,minibatch_size);


	thrust_d_W_a_grad = thrust::device_pointer_cast(d_W_a_grad);
	thrust_d_v_p_grad = thrust::device_pointer_cast(d_v_p_grad);
	thrust_d_W_p_grad = thrust::device_pointer_cast(d_W_p_grad);
	thrust_d_W_c_p1_grad = thrust::device_pointer_cast(d_W_c_p1_grad);
	thrust_d_W_c_p2_grad = thrust::device_pointer_cast(d_W_c_p2_grad);
	thrust_d_output_bias_grad = thrust::device_pointer_cast(d_output_bias_grad);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");

	clear_gradients();

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_sum, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_s_sum, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");


	for(int i=0; i<longest_sent; i++) {
		nodes.push_back( attention_node<dType>(LSTM_size,minibatch_size,device_number,D,feed_input,this,i) );
	}

	//now construct d_total_hs_mat
	dType **h_total_hs_mat = (dType **)malloc(longest_sent*sizeof(dType*));
	dType **h_total_hs_error = (dType **)malloc(longest_sent*sizeof(dType*));

	for(int i=0; i<longest_sent; i++) {
		if(model->source_hidden_layers.size() == 0) {
			h_total_hs_mat[i] = model->input_layer_source.nodes[i].d_h_t;
			h_total_hs_error[i] = model->input_layer_source.nodes[i].d_d_ERRt_ht;
		}
		else {
			h_total_hs_mat[i] = model->source_hidden_layers[model->source_hidden_layers.size()-1].nodes[i].d_h_t;
			h_total_hs_error[i] = model->source_hidden_layers[model->source_hidden_layers.size()-1].nodes[i].d_d_ERRt_ht;
		}
	}

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_total_hs_mat, longest_sent*sizeof(dType*)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_total_hs_error, longest_sent*sizeof(dType*)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_batch_info, 2*minibatch_size*sizeof(int)),"GPU memory allocation failed\n");
	cudaMemcpy(d_total_hs_mat,h_total_hs_mat,longest_sent*sizeof(dType*),cudaMemcpyHostToDevice);
	cudaMemcpy(d_total_hs_error,h_total_hs_error,longest_sent*sizeof(dType*),cudaMemcpyHostToDevice);

	free(h_total_hs_mat);
}



template<typename dType>
void attention_layer<dType>::clear_gradients() {
	cudaSetDevice(device_number);

	cudaMemsetAsync(d_W_a_grad,0,LSTM_size*LSTM_size*sizeof(dType),layer_info.s0);
	cudaMemsetAsync(d_W_p_grad,0,LSTM_size*LSTM_size*sizeof(dType),layer_info.s0);
	cudaMemsetAsync(d_v_p_grad,0,LSTM_size*1*sizeof(dType),layer_info.s0);
	cudaMemsetAsync(d_output_bias_grad,0,LSTM_size*1*sizeof(dType),layer_info.s0);
	cudaMemsetAsync(d_W_c_p1_grad,0,LSTM_size*LSTM_size*sizeof(dType),layer_info.s0);
	cudaMemsetAsync(d_W_c_p2_grad,0,LSTM_size*LSTM_size*sizeof(dType),layer_info.s0);

}


template<typename dType>
void attention_layer<dType>::clip_gradients_func() {

	norm_clip_GPU_v2(thrust_d_W_a_grad,d_W_a_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_W_p_grad,d_W_p_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_v_p_grad,d_v_p_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_output_bias_grad,d_output_bias_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_W_c_p1_grad,d_W_c_p1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_W_c_p2_grad,d_W_c_p2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

}

template<typename dType>
void attention_layer<dType>::scale_gradients() {

	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_W_a_grad,thrust_d_W_a_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_W_p_grad,thrust_d_W_p_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_v_p_grad,thrust_d_v_p_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_output_bias_grad,thrust_d_output_bias_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_W_c_p1_grad,thrust_d_W_c_p1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_W_c_p2_grad,thrust_d_W_c_p2_grad + LSTM_size*LSTM_size,unary_op);

}


template<typename dType>
void attention_layer<dType>::update_params() {

	gradient_update_mats<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_a,d_W_a_grad,model->input_layer_target.learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_p,d_W_p_grad,model->input_layer_target.learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<std::min(256,(LSTM_size*1 + 256 - 1)/256),256,0,layer_info.s0>>>(d_v_p,d_v_p_grad,model->input_layer_target.learning_rate,LSTM_size*1);
	gradient_update_mats<<<std::min(256,(LSTM_size*1 + 256 - 1)/256),256,0,layer_info.s0>>>(d_output_bias,d_output_bias_grad,model->input_layer_target.learning_rate,LSTM_size*1);
	gradient_update_mats<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_c_p1,d_W_c_p1_grad,model->input_layer_target.learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_c_p2,d_W_c_p2_grad,model->input_layer_target.learning_rate,LSTM_size*LSTM_size);

	devSynchAll();
}


template<typename dType>
void attention_layer<dType>::norm_p1() {

	norm_clip_GPU_v2_p1(thrust_d_W_a_grad,d_W_a_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_W_p_grad,d_W_p_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_v_p_grad,d_v_p_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_output_bias_grad,d_output_bias_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_W_c_p1_grad,d_W_c_p1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_W_c_p2_grad,d_W_c_p2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

}

template<typename dType>
void attention_layer<dType>::norm_p2() {

	norm_clip_GPU_v2_p2(thrust_d_W_a_grad,d_W_a_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_W_p_grad,d_W_p_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_v_p_grad,d_v_p_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_output_bias_grad,d_output_bias_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_W_c_p1_grad,d_W_c_p1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_W_c_p2_grad,d_W_c_p2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

}



template<typename dType>
void attention_layer<dType>::clip_indiv() {

	clip_mat_kernel<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_a_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*LSTM_size);
	clip_mat_kernel<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_p_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*LSTM_size);
	clip_mat_kernel<<<std::min(256,(LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_v_p_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*1);
	clip_mat_kernel<<<std::min(256,(LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_output_bias_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*1);
	clip_mat_kernel<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_c_p1_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*LSTM_size);
	clip_mat_kernel<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,layer_info.s0>>>(d_W_c_p2_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*LSTM_size);

}


template<typename dType>
void attention_layer<dType>::dump_weights(std::ofstream &output) {
	write_matrix_GPU(d_W_a,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_W_p,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_v_p,LSTM_size,1,output);
	write_matrix_GPU(d_output_bias,LSTM_size,1,output);
	write_matrix_GPU(d_W_c_p1,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_W_c_p2,LSTM_size,LSTM_size,output);
}


template<typename dType>
void attention_layer<dType>::load_weights(std::ifstream &input) {
	read_matrix_GPU(d_W_a,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_W_p,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_v_p,LSTM_size,1,input);
	read_matrix_GPU(d_output_bias,LSTM_size,1,input);
	read_matrix_GPU(d_W_c_p1,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_W_c_p2,LSTM_size,LSTM_size,input);
}



template<typename dType>
void attention_layer<dType>::check_gradients(dType epsilon) {

		std::cout << "--------------------GRADIENT CHECKING FOR ATTENTION LAYER GPU-------------------------\n";
		std::cout << "GRADIENT CHECKING FOR W_c_p1\n";
		check_gradient_GPU(epsilon,d_W_c_p1,d_W_c_p1_grad,LSTM_size,LSTM_size);
		
		std::cout << "GRADIENT CHECKING FOR W_c_p2\n";
		check_gradient_GPU(epsilon,d_W_c_p2,d_W_c_p2_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR OUTPUT BIAS\n";
		check_gradient_GPU(epsilon,d_output_bias,d_output_bias_grad,LSTM_size,1);

		std::cout << "GRADIENT CHECKING FOR v_p\n";
		check_gradient_GPU(epsilon,d_v_p,d_v_p_grad,LSTM_size,1);

		std::cout << "GRADIENT CHECKING FOR W_p\n";
		check_gradient_GPU(epsilon,d_W_p,d_W_p_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR W_a\n";
		check_gradient_GPU(epsilon,d_W_a,d_W_a_grad,LSTM_size,LSTM_size);
}


template<typename dType>
void attention_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {

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
				std::cout << "ZERO GRADIENTS\n";
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}

template<typename dType>
void attention_layer<dType>::prep_minibatch_info(int *h_batch_info) {
	cudaSetDevice(device_number);
	cudaMemcpy(d_batch_info,h_batch_info,2*minibatch_size*sizeof(int),cudaMemcpyHostToDevice);
}


