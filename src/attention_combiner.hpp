//hpp

template<typename dType>
attention_combiner_layer<dType>::attention_combiner_layer(global_params &params,int device_number,neuralMT_model<dType> *model) {

	this->LSTM_size = params.LSTM_size;
	this->minibatch_size = params.minibatch_size;
	this->longest_sent = params.longest_sent;
	this->device_number = device_number;
	this->model = model;
	this->norm_clip = params.norm_clip;
	this->add_ht = params.multi_src_params.add_ht;

	cudaSetDevice(device_number);

	cudaStreamCreate(&s0);
	cudaEventCreate(&forward_prop_done);
	cudaEventCreate(&start_forward);
	cudaEventCreate(&start_backward);
	cudaEventCreate(&backward_prop_done);

	CUBLAS_ERROR_WRAPPER(cublasCreate(&handle),"CUBLAS handler initialization failed\n");

	dType *h_temp;
	full_matrix_setup(&h_temp,&d_M_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_b_d,LSTM_size,1);

	full_matrix_setup(&h_temp,&d_M_1_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_2_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_b_d_grad,LSTM_size,1);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");

	thrust_d_M_1_grad = thrust::device_pointer_cast(d_M_1_grad);
	thrust_d_M_2_grad = thrust::device_pointer_cast(d_M_2_grad);
	thrust_d_b_d_grad = thrust::device_pointer_cast(d_b_d_grad);

	// cudaMemset(d_b_d,0,LSTM_size*1*sizeof(dType));


	for(int i=0; i<longest_sent; i++) {
		nodes.push_back(new attention_combiner_node<dType>(params,this,i));
	}

	clear_gradients();
}



template<typename dType>
void attention_combiner_layer<dType>::clear_gradients() {

	if(!add_ht) {
		cudaMemset(d_M_1_grad,0,LSTM_size*LSTM_size*sizeof(dType));
		cudaMemset(d_M_2_grad,0,LSTM_size*LSTM_size*sizeof(dType));
		cudaMemset(d_b_d_grad,0,LSTM_size*1*sizeof(dType));
	}
}


template<typename dType>
void attention_combiner_layer<dType>::check_gradients(dType epsilon) {

	if(!add_ht) {
		std::cout << "--------------------GRADIENT CHECKING FOR ATTENTION COMBINER LAYER GPU-------------------------\n";
		std::cout << "GRADIENT CHECKING FOR d_M_1\n";
		check_gradient_GPU(epsilon,d_M_1,d_M_1_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR d_M_2\n";
		check_gradient_GPU(epsilon,d_M_2,d_M_2_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR d_b_d\n";
		check_gradient_GPU(epsilon,d_b_d,d_b_d_grad,LSTM_size,1);
	}
}



template<typename dType>
void attention_combiner_layer<dType>::dump_weights(std::ofstream &output) {
	if(!add_ht) {
		cudaSetDevice(device_number);
		write_matrix_GPU(d_M_1,LSTM_size,LSTM_size,output);
		write_matrix_GPU(d_M_2,LSTM_size,LSTM_size,output);
		write_matrix_GPU(d_b_d,LSTM_size,1,output);
	}
}


template<typename dType>
void attention_combiner_layer<dType>::load_weights(std::ifstream &input) {
	if(!add_ht) {
		cudaSetDevice(device_number);
		read_matrix_GPU(d_M_1,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_M_2,LSTM_size,LSTM_size,input);
		read_matrix_GPU(d_b_d,LSTM_size,1,input);
	}
}

template<typename dType>
void attention_combiner_layer<dType>::clip_gradients_func() {
	if(!add_ht) {
		cudaSetDevice(device_number);
		norm_clip_GPU_v2(thrust_d_M_1_grad,d_M_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_2_grad,d_M_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	}
}


template<typename dType>
void attention_combiner_layer<dType>::scale_gradients() {
	if(!add_ht) {
		scale_functor unary_op(minibatch_size);
		thrust::for_each(thrust_d_M_1_grad,thrust_d_M_1_grad + LSTM_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_M_2_grad,thrust_d_M_2_grad + LSTM_size*LSTM_size,unary_op);
		thrust::for_each(thrust_d_b_d_grad,thrust_d_b_d_grad + LSTM_size*1,unary_op);
	}
}


template<typename dType>
void attention_combiner_layer<dType>::update_params() {
	if(!add_ht) {
		gradient_update_mats<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,s0>>>(d_M_1,d_M_1_grad,model->input_layer_target.learning_rate,LSTM_size*LSTM_size);
		gradient_update_mats<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,s0>>>(d_M_2,d_M_2_grad,model->input_layer_target.learning_rate,LSTM_size*LSTM_size);
		gradient_update_mats<<<std::min(256,(LSTM_size*1 + 256 - 1)/256),256,0,s0>>>(d_b_d,d_b_d_grad,model->input_layer_target.learning_rate,LSTM_size*1);
		//clip_weights_kernel<<<256,256,0,s0>>>(d_M_1,LSTM_size*LSTM_size);
		//clip_weights_kernel<<<256,256,0,s0>>>(d_M_2,LSTM_size*LSTM_size);
		devSynchAll();
	}
}


template<typename dType>
void attention_combiner_layer<dType>::norm_p1() {

	if(!add_ht) {

		// if(BZ_CUDA::print_norms) {
		// 	HPC_output << "----------------------- PRINTING TOP GRADIENTS FOR ATTENTION COMBINER -----------------------\n";
		// }

		norm_clip_GPU_v2_p1(thrust_d_M_1_grad,d_M_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		// if(BZ_CUDA::print_norms) {
		// 	HPC_output << "----------------------- PRINTING GRAD NORM FOR d_M_1_grad -----------------------\n";
		// 	HPC_output << BZ_CUDA::recent_sum << "\n";
		// }

		norm_clip_GPU_v2_p1(thrust_d_M_2_grad,d_M_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		// if(BZ_CUDA::print_norms) {
		// 	HPC_output << "----------------------- PRINTING GRAD NORM FOR d_M_2_grad -----------------------\n";
		// 	HPC_output << BZ_CUDA::recent_sum << "\n";
		// }

		norm_clip_GPU_v2_p1(thrust_d_b_d_grad,d_b_d_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		// if(BZ_CUDA::print_norms) {
		// 	HPC_output << "----------------------- PRINTING GRAD NORM FOR d_b_d -----------------------\n";
		// 	HPC_output << BZ_CUDA::recent_sum << "\n";
		// 	thrust::device_ptr<dType> thrust_d_b_d = thrust::device_pointer_cast(d_b_d);
		// 	std::cout << "Printing 20 random bias values\n";
		// 	for(int i=0; i<20; i++) {
		// 		std::cout << thrust_d_b_d[i] << " ";
		// 	}
		// 	std::cout << "\n";

		// }
	}
}

template<typename dType>
void attention_combiner_layer<dType>::norm_p2() {
	if(!add_ht) {
		norm_clip_GPU_v2_p2(thrust_d_M_1_grad,d_M_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2_p2(thrust_d_M_2_grad,d_M_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2_p2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	}
}


template<typename dType>
void attention_combiner_layer<dType>::clip_indiv() {
	if(!add_ht) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,s0>>>(d_M_1_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*LSTM_size);
		clip_mat_kernel<<<std::min(256,(LSTM_size*LSTM_size + 256 - 1)/256),256,0,s0>>>(d_M_2_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*LSTM_size);
		clip_mat_kernel<<<std::min(256,(LSTM_size + 256 - 1)/256),256,0,s0>>>(d_b_d_grad,BZ_CUDA::ind_norm_clip_thres,LSTM_size*1);
		devSynchAll();
	}
}


template<typename dType>
void attention_combiner_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {

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





