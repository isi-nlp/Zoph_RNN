//cuda kernels
template<typename dType>
__global__
void sigmoid_bias_kernel(dType *d_final,dType *d_bias,int state_size,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_final[i] = 1.0/(1.0 + cuda_exp_wrapper(-1*(d_final[i] + d_bias[i%state_size])));
	}
}


//cuda kernels
template<typename dType>
__global__
void ReLU_bias_kernel(dType *d_final,dType *d_bias,int state_size,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		dType temp = d_final[i] + d_bias[i%state_size];
		#ifdef RELU_NONLIN_BZ
		d_final[i] = temp * (temp > 0);
		#else
		d_final[i] = tanh_wrapper(temp);
		#endif
	}
}

//cuda kernels
template<typename dType>
__global__
void highway_compute_z_kernel(dType *d_z,dType *d_g,dType *d_t,dType *d_y,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_z[i] = d_t[i]*d_g[i] + (1-d_t[i])*d_y[i];
	}
}

template<typename dType>
__global__
void error_g_kernel(dType *d_Err_g,dType *d_Err_z,dType *d_t,dType *d_g,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		#ifdef RELU_NONLIN_BZ
		d_Err_g[i] = d_Err_z[i] * d_t[i] * (d_g[i] > 0);
		#else
		d_Err_g[i] = d_Err_z[i] * d_t[i] * (1-d_g[i]*d_g[i]);
		#endif
	}
}

template<typename dType>
__global__
void error_t_kernel(dType *d_Err_t,dType *d_Err_z,dType *d_t,dType *d_g,dType *d_y,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_Err_t[i] = d_Err_z[i] * (d_g[i] - d_y[i]) * d_t[i] * (1 - d_t[i]);
	}
}

template<typename dType>
__global__
void error_y_final(dType *d_Err_z,dType *d_t,dType *d_Err_y,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_Err_y[i] += d_Err_z[i] * (1 - d_t[i]);
	}
}


template<typename dType>
void highway_network_layer<dType>::init(int state_size,int minibatch_size,int longest_sent,int device_number,
	cublasHandle_t &handle,cudaStream_t &s0,neuralMT_model<dType> *model,dType norm_clip) 
{
	
	this->state_size = state_size;
	this->minibatch_size = minibatch_size;
	this->device_number = device_number;
	this->handle = handle;
	this->s0 = s0;
	this->model = model;
	this->norm_clip = norm_clip;

	cudaSetDevice(device_number);

	dType *h_temp;
	full_matrix_setup(&h_temp,&d_W_h,state_size,state_size);
	full_matrix_setup(&h_temp,&d_W_t,state_size,state_size);
	full_matrix_setup(&h_temp,&d_b_h,state_size,1);
	full_matrix_setup(&h_temp,&d_b_t,state_size,1);

	thrust::device_ptr<dType> bias_ptr = thrust::device_pointer_cast(d_b_t);
	for(int i=0; i<state_size; i++) {
		bias_ptr[i] = -2;
	}

	full_matrix_setup(&h_temp,&d_W_h_grad,state_size,state_size);
	full_matrix_setup(&h_temp,&d_W_t_grad,state_size,state_size);
	full_matrix_setup(&h_temp,&d_b_h_grad,state_size,1);
	full_matrix_setup(&h_temp,&d_b_t_grad,state_size,1);

	full_matrix_setup(&h_temp,&d_temp,state_size,minibatch_size);
	full_vector_setup_ones(&h_temp,&d_ones_minibatch,minibatch_size);

	full_matrix_setup(&h_temp,&d_Err_t,state_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_Err_y,state_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_Err_g,state_size,minibatch_size);
	//full_matrix_setup(&h_temp,&d_Err_z,state_size,minibatch_size);


	for(int i=0; i<longest_sent; i++) {
		nodes.push_back( new highway_node<dType>(state_size,minibatch_size,i) );
	}

	thrust_d_W_h_grad = thrust::device_pointer_cast(d_W_h_grad);
	thrust_d_W_t_grad = thrust::device_pointer_cast(d_W_t_grad);
	thrust_d_b_h_grad = thrust::device_pointer_cast(d_b_h_grad);
	thrust_d_b_t_grad = thrust::device_pointer_cast(d_b_t_grad);


	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");

	clear_gradients();
}

template<typename dType>
void highway_network_layer<dType>::forward(int index,dType *d_y_temp) {

	cudaSetDevice(device_number);

	dType *d_t = nodes[index]->d_t; //gate value
	dType *d_y = nodes[index]->d_y; //input
	dType *d_g = nodes[index]->d_g; //new value from ReLU
	dType *d_z = nodes[index]->d_z;//output

	cudaMemcpyAsync(d_y, d_y_temp, state_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s0);

	//calculate t
	dType alpha = 1;
	dType beta = 0;
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,state_size,minibatch_size,state_size,&alpha,d_W_t,state_size,
		d_y,state_size,&beta,d_t,state_size),"Forward prop o_t temp1 failed\n");

	sigmoid_bias_kernel<<<256,256,0,s0>>>(d_t,d_b_t,state_size,state_size*minibatch_size);


	//calculate g
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,state_size,minibatch_size,state_size,&alpha,d_W_h,state_size,
		d_y,state_size,&beta,d_g,state_size),"Forward prop o_t temp1 failed\n");

	ReLU_bias_kernel<<<256,256,0,s0>>>(d_g,d_b_h,state_size,state_size*minibatch_size);

	//calculate z
	highway_compute_z_kernel<<<256,256,0,s0>>>(d_z,d_g,d_t,d_y,state_size*minibatch_size);

}



template<typename dType>
void highway_network_layer<dType>::backward(int index,dType *d_Err_z_temp) {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(device_number);

	dType *d_t = nodes[index]->d_t; //gate value
	dType *d_y = nodes[index]->d_y; //input
	dType *d_g = nodes[index]->d_g; //new value from ReLU
	dType *d_z = nodes[index]->d_z;//output	

	if(d_Err_z_temp==NULL) {
		//do nothing since it had already been copied
	}
	else {
		cudaMemcpyAsync(nodes[index]->d_Err_z, d_Err_z_temp, state_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s0);
	}

	//compute error with respect to g
	error_g_kernel<<<256,256,0,s0>>>(d_Err_g,nodes[index]->d_Err_z,d_t,d_g,state_size*minibatch_size);

	//compute error for W_h
	dType alpha = 1;
	dType beta = 1;
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,state_size,state_size,minibatch_size,&alpha,
		d_Err_g,state_size,d_y,state_size,&beta,d_W_h_grad,state_size),"HIGHWAY backprop W_h failed\n");

	//error for b_h
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,state_size,minibatch_size,&alpha,d_Err_g,state_size,
		d_ones_minibatch,1,&beta,d_b_h_grad,1),"HIGHWAY backprop b_h failed\n");

	
	alpha = 1;
	beta = 0;
	//for partial y
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,state_size,minibatch_size,state_size,
		&alpha,d_W_h,state_size,d_Err_g,state_size,&beta,d_Err_y,state_size),"HIGHWAY BACKPROP y 1\n");

	
	//compute error with respect to t
	error_t_kernel<<<256,256,0,s0>>>(d_Err_t,nodes[index]->d_Err_z,d_t,d_g,d_y,state_size*minibatch_size);

	
	//compute error for W_h
	alpha = 1;
	beta = 1;
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,state_size,state_size,minibatch_size,&alpha,
		d_Err_t,state_size,d_y,state_size,&beta,d_W_t_grad,state_size),"HIGHWAY backprop W_h failed\n");


	//error for b_h
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,state_size,minibatch_size,&alpha,d_Err_t,state_size,
		d_ones_minibatch,1,&beta,d_b_t_grad,1),"HIGHWAY backprop b_h failed\n");

	//for partial y
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,state_size,minibatch_size,state_size,
		&alpha,d_W_t,state_size,d_Err_t,state_size,&beta,d_Err_y,state_size),"HIGHWAY BACKPROP y 1\n");

	error_y_final<<<256,256,0,s0>>>(nodes[index]->d_Err_z,d_t,d_Err_y,state_size*minibatch_size);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
}



template<typename dType>
void highway_network_layer<dType>::clear_gradients() {

	cudaMemset(d_W_h_grad,0,state_size*state_size*sizeof(dType));
	cudaMemset(d_W_t_grad,0,state_size*state_size*sizeof(dType));
	cudaMemset(d_b_h_grad,0,state_size*1*sizeof(dType));
	cudaMemset(d_b_t_grad,0,state_size*1*sizeof(dType));
}

template<typename dType>
void highway_network_layer<dType>::norm_p1() {

	norm_clip_GPU_v2_p1(thrust_d_W_h_grad,d_W_h_grad,norm_clip,state_size*state_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_W_t_grad,d_W_t_grad,norm_clip,state_size*state_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_h_grad,d_b_h_grad,norm_clip,state_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_t_grad,d_b_t_grad,norm_clip,state_size,d_temp_result,d_result);
}

template<typename dType>
void highway_network_layer<dType>::norm_p2() {

	norm_clip_GPU_v2_p2(thrust_d_W_h_grad,d_W_h_grad,norm_clip,state_size*state_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_W_t_grad,d_W_t_grad,norm_clip,state_size*state_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_h_grad,d_b_h_grad,norm_clip,state_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_t_grad,d_b_t_grad,norm_clip,state_size,d_temp_result,d_result);
}

template<typename dType>
void highway_network_layer<dType>::scale_gradients() {
	scale_functor unary_op(minibatch_size);

	thrust::for_each(thrust_d_W_h_grad,thrust_d_W_h_grad + state_size*state_size,unary_op);
	thrust::for_each(thrust_d_W_t_grad,thrust_d_W_t_grad + state_size*state_size,unary_op);
	thrust::for_each(thrust_d_b_h_grad,thrust_d_b_h_grad + state_size,unary_op);
	thrust::for_each(thrust_d_b_t_grad,thrust_d_b_t_grad + state_size,unary_op);
}


template<typename dType>
void highway_network_layer<dType>::update_params() {
	gradient_update_mats<<<256,256,0,s0>>>(d_W_h,d_W_h_grad,model->input_layer_target.learning_rate,state_size*state_size);
	gradient_update_mats<<<256,256,0,s0>>>(d_W_t,d_W_t_grad,model->input_layer_target.learning_rate,state_size*state_size);
	gradient_update_mats<<<256,256,0,s0>>>(d_b_h,d_b_h_grad,model->input_layer_target.learning_rate,state_size);
	gradient_update_mats<<<256,256,0,s0>>>(d_b_t,d_b_t_grad,model->input_layer_target.learning_rate,state_size);
}


template<typename dType>
void highway_network_layer<dType>::clip_gradients_func() {

	norm_clip_GPU_v2(thrust_d_W_h_grad,d_W_h_grad,norm_clip,state_size*state_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_W_t_grad,d_W_t_grad,norm_clip,state_size*state_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_b_h_grad,d_b_h_grad,norm_clip,state_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_b_t_grad,d_b_t_grad,norm_clip,state_size,d_temp_result,d_result);
}

template<typename dType>
void highway_network_layer<dType>::dump_weights(std::ofstream &output) {

	write_matrix_GPU(d_W_h,state_size,state_size,output);
	write_matrix_GPU(d_W_t,state_size,state_size,output);
	write_matrix_GPU(d_b_h,state_size,1,output);
	write_matrix_GPU(d_b_t,state_size,1,output);
}

template<typename dType>
void highway_network_layer<dType>::load_weights(std::ifstream &input) {

	read_matrix_GPU(d_W_h,state_size,state_size,input);
	read_matrix_GPU(d_W_t,state_size,state_size,input);
	read_matrix_GPU(d_b_h,state_size,1,input);
	read_matrix_GPU(d_b_t,state_size,1,input);
}

template<typename dType>
void highway_network_layer<dType>::check_gradients(dType epsilon) {
	std::cout << "GRADIENT CHECKING FOR W_h\n";
	check_gradient_GPU(epsilon,d_W_h,d_W_h_grad,state_size,state_size);
	std::cout << "GRADIENT CHECKING FOR W_t\n";
	check_gradient_GPU(epsilon,d_W_t,d_W_t_grad,state_size,state_size);
	std::cout << "GRADIENT CHECKING FOR b_h\n";
	check_gradient_GPU(epsilon,d_b_h,d_b_h_grad,state_size,1);
	std::cout << "GRADIENT CHECKING FOR b_t\n";
	check_gradient_GPU(epsilon,d_b_t,d_b_t_grad,state_size,1);
}


template<typename dType>
void highway_network_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {

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





