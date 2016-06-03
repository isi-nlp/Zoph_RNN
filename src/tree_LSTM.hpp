//tree LSTM implementation



template<typename dType>
tree_LSTM<dType>::tree_LSTM(global_params &params,int device_number,encoder_multi_source<dType> *model) {

	this->device_number = device_number;
	this->minibatch_size = params.minibatch_size;
	this->LSTM_size = params.LSTM_size;
	this->clip_gradients = params.clip_gradient; //If true then clip gradients
	this->norm_clip = params.norm_clip; //For gradient clipping
	this->model = model;

	cudaSetDevice(device_number);

	CUBLAS_ERROR_WRAPPER(cublasCreate(&handle),"CUBLAS handler initialization failed\n");
	cudaStreamCreate(&s0);

	dType *h_temp;
	full_vector_setup_ones(&h_temp,&d_ones_minibatch,minibatch_size);


	full_matrix_setup(&h_temp,&d_child_ht_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_child_ht_2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_child_ct_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_child_ct_2,LSTM_size,minibatch_size);


	full_matrix_setup(&h_temp,&d_i_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_f_t_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_f_t_2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_c_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_h_t,LSTM_size,minibatch_size);


	full_matrix_setup(&h_temp,&d_b_i,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_f,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_o,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_c,LSTM_size,1);

	thrust::device_ptr<dType> bias_ptr = thrust::device_pointer_cast(d_b_f);
	for(int i=0; i<LSTM_size; i++) {
		bias_ptr[i] = 1;
	}

	full_matrix_setup(&h_temp,&d_M_i_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_f_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_o_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_c_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_i_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_f_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_o_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_c_2,LSTM_size,LSTM_size);

	full_matrix_setup(&h_temp,&d_b_i_grad,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_f_grad,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_o_grad,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_c_grad,LSTM_size,1);

	full_matrix_setup(&h_temp,&d_M_i_1_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_f_1_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_o_1_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_c_1_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_i_2_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_f_2_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_o_2_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_c_2_grad,LSTM_size,LSTM_size);

	full_matrix_setup(&h_temp,&d_temp1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp3,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp4,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp5,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp6,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp7,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp8,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp,&d_d_ERRnTOt_ht,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOtp1_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRt_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_it,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_ot,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_ft_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_ft_2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_tanhcpt,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp,&d_d_ERRnTOt_h1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_h2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_c1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_d_ERRnTOt_c2,LSTM_size,minibatch_size);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");

	clear_gradients();
}


//this one if for decoding
template<typename dType>
tree_LSTM<dType>::tree_LSTM(int LSTM_size,int device_number,encoder_multi_source<dType> *model) {
	this->device_number = device_number;
	this->minibatch_size = 1;
	this->LSTM_size = LSTM_size;
	this->model = model;

	cudaSetDevice(device_number);

	CUBLAS_ERROR_WRAPPER(cublasCreate(&handle),"CUBLAS handler initialization failed\n");
	cudaStreamCreate(&s0);

	dType *h_temp;
	full_vector_setup_ones(&h_temp,&d_ones_minibatch,minibatch_size);

	full_matrix_setup(&h_temp,&d_child_ht_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_child_ht_2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_child_ct_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_child_ct_2,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp,&d_i_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_f_t_1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_f_t_2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_c_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_h_t,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp,&d_b_i,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_f,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_o,LSTM_size,1);
	full_matrix_setup(&h_temp,&d_b_c,LSTM_size,1);

	full_matrix_setup(&h_temp,&d_M_i_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_f_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_o_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_c_1,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_i_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_f_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_o_2,LSTM_size,LSTM_size);
	full_matrix_setup(&h_temp,&d_M_c_2,LSTM_size,LSTM_size);


	full_matrix_setup(&h_temp,&d_temp1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp3,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp4,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp5,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp6,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp7,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_temp8,LSTM_size,minibatch_size);
}

template<typename dType>
void tree_LSTM<dType>::clear_gradients() {

	cudaSetDevice(device_number);
	cudaMemset(d_M_i_1_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_f_1_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_o_1_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_c_1_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_i_2_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_f_2_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_o_2_grad,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_M_c_2_grad,0,LSTM_size*LSTM_size*sizeof(dType));

	cudaMemset(d_b_i_grad,0,LSTM_size*1*sizeof(dType));
	cudaMemset(d_b_f_grad,0,LSTM_size*1*sizeof(dType));
	cudaMemset(d_b_o_grad,0,LSTM_size*1*sizeof(dType));
	cudaMemset(d_b_c_grad,0,LSTM_size*1*sizeof(dType));

	devSynchAll();
}

template<typename dType>
void tree_LSTM<dType>::dump_weights(std::ofstream &output) {

	cudaSetDevice(device_number);
	write_matrix_GPU(d_M_i_1,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_f_1,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_o_1,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_c_1,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_i_2,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_f_2,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_o_2,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_c_2,LSTM_size,LSTM_size,output);

	write_matrix_GPU(d_b_i,LSTM_size,1,output);
	write_matrix_GPU(d_b_f,LSTM_size,1,output);
	write_matrix_GPU(d_b_o,LSTM_size,1,output);
	write_matrix_GPU(d_b_c,LSTM_size,1,output);
}

template<typename dType>
void tree_LSTM<dType>::load_weights(std::ifstream &input) {
	cudaSetDevice(device_number);
	read_matrix_GPU(d_M_i_1,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_f_1,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_o_1,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_c_1,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_i_2,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_f_2,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_o_2,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_c_2,LSTM_size,LSTM_size,input);

	read_matrix_GPU(d_b_i,LSTM_size,1,input);
	read_matrix_GPU(d_b_f,LSTM_size,1,input);
	read_matrix_GPU(d_b_o,LSTM_size,1,input);
	read_matrix_GPU(d_b_c,LSTM_size,1,input);
}


template<typename dType>
void tree_LSTM<dType>::update_weights() {

	scale_functor unary_op(minibatch_size);

	thrust::device_ptr<dType> thrust_d_M_i_1_grad = thrust::device_pointer_cast(d_M_i_1_grad);
	thrust::device_ptr<dType> thrust_d_M_f_1_grad = thrust::device_pointer_cast(d_M_f_1_grad);
	thrust::device_ptr<dType> thrust_d_M_o_1_grad = thrust::device_pointer_cast(d_M_o_1_grad);
	thrust::device_ptr<dType> thrust_d_M_c_1_grad = thrust::device_pointer_cast(d_M_c_1_grad);
	thrust::device_ptr<dType> thrust_d_M_i_2_grad = thrust::device_pointer_cast(d_M_i_2_grad);
	thrust::device_ptr<dType> thrust_d_M_f_2_grad = thrust::device_pointer_cast(d_M_f_2_grad);
	thrust::device_ptr<dType> thrust_d_M_o_2_grad = thrust::device_pointer_cast(d_M_o_2_grad);
	thrust::device_ptr<dType> thrust_d_M_c_2_grad = thrust::device_pointer_cast(d_M_c_2_grad);

	thrust::device_ptr<dType> thrust_d_b_i_grad= thrust::device_pointer_cast(d_b_i_grad);
	thrust::device_ptr<dType> thrust_d_b_f_grad= thrust::device_pointer_cast(d_b_f_grad);
	thrust::device_ptr<dType> thrust_d_b_o_grad= thrust::device_pointer_cast(d_b_o_grad);
	thrust::device_ptr<dType> thrust_d_b_c_grad= thrust::device_pointer_cast(d_b_c_grad);
	
	thrust::for_each(thrust_d_M_i_1_grad,thrust_d_M_i_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_f_1_grad,thrust_d_M_f_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_o_1_grad,thrust_d_M_o_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_c_1_grad,thrust_d_M_c_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_i_2_grad,thrust_d_M_i_2_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_f_2_grad,thrust_d_M_f_2_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_o_2_grad,thrust_d_M_o_2_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_c_2_grad,thrust_d_M_c_2_grad + LSTM_size*LSTM_size,unary_op);
	
	thrust::for_each(thrust_d_b_i_grad,thrust_d_b_i_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_b_f_grad,thrust_d_b_f_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_b_o_grad,thrust_d_b_o_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_b_c_grad,thrust_d_b_c_grad + LSTM_size*1,unary_op);


	norm_clip_GPU_v2(thrust_d_M_i_1_grad,d_M_i_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_f_1_grad,d_M_f_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_o_1_grad,d_M_o_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_c_1_grad,d_M_c_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_i_2_grad,d_M_i_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_f_2_grad,d_M_f_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_o_2_grad,d_M_o_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_M_c_2_grad,d_M_c_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

	norm_clip_GPU_v2(thrust_d_b_i_grad,d_b_i_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_b_f_grad,d_b_f_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_b_o_grad,d_b_o_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2(thrust_d_b_c_grad,d_b_c_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);


	gradient_update_mats<<<256,256>>>(d_M_i_1,d_M_i_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_f_1,d_M_f_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_o_1,d_M_o_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_c_1,d_M_c_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_i_2,d_M_i_2_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_f_2,d_M_f_2_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_o_2,d_M_o_2_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_c_2,d_M_c_2_grad,model->learning_rate,LSTM_size*LSTM_size);

	gradient_update_mats<<<256,256>>>(d_b_i,d_b_i_grad,model->learning_rate,LSTM_size*1);
	gradient_update_mats<<<256,256>>>(d_b_f,d_b_f_grad,model->learning_rate,LSTM_size*1);
	gradient_update_mats<<<256,256>>>(d_b_o,d_b_o_grad,model->learning_rate,LSTM_size*1);
	gradient_update_mats<<<256,256>>>(d_b_c,d_b_c_grad,model->learning_rate,LSTM_size*1);

	devSynchAll();
}

template<typename dType>
void tree_LSTM<dType>::calculate_global_norm() {

	scale_functor unary_op(minibatch_size);

	thrust::device_ptr<dType> thrust_d_M_i_1_grad = thrust::device_pointer_cast(d_M_i_1_grad);
	thrust::device_ptr<dType> thrust_d_M_f_1_grad = thrust::device_pointer_cast(d_M_f_1_grad);
	thrust::device_ptr<dType> thrust_d_M_o_1_grad = thrust::device_pointer_cast(d_M_o_1_grad);
	thrust::device_ptr<dType> thrust_d_M_c_1_grad = thrust::device_pointer_cast(d_M_c_1_grad);
	thrust::device_ptr<dType> thrust_d_M_i_2_grad = thrust::device_pointer_cast(d_M_i_2_grad);
	thrust::device_ptr<dType> thrust_d_M_f_2_grad = thrust::device_pointer_cast(d_M_f_2_grad);
	thrust::device_ptr<dType> thrust_d_M_o_2_grad = thrust::device_pointer_cast(d_M_o_2_grad);
	thrust::device_ptr<dType> thrust_d_M_c_2_grad = thrust::device_pointer_cast(d_M_c_2_grad);

	thrust::device_ptr<dType> thrust_d_b_i_grad= thrust::device_pointer_cast(d_b_i_grad);
	thrust::device_ptr<dType> thrust_d_b_f_grad= thrust::device_pointer_cast(d_b_f_grad);
	thrust::device_ptr<dType> thrust_d_b_o_grad= thrust::device_pointer_cast(d_b_o_grad);
	thrust::device_ptr<dType> thrust_d_b_c_grad= thrust::device_pointer_cast(d_b_c_grad);
	
	thrust::for_each(thrust_d_M_i_1_grad,thrust_d_M_i_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_f_1_grad,thrust_d_M_f_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_o_1_grad,thrust_d_M_o_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_c_1_grad,thrust_d_M_c_1_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_i_2_grad,thrust_d_M_i_2_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_f_2_grad,thrust_d_M_f_2_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_o_2_grad,thrust_d_M_o_2_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_c_2_grad,thrust_d_M_c_2_grad + LSTM_size*LSTM_size,unary_op);
	
	thrust::for_each(thrust_d_b_i_grad,thrust_d_b_i_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_b_f_grad,thrust_d_b_f_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_b_o_grad,thrust_d_b_o_grad + LSTM_size*1,unary_op);
	thrust::for_each(thrust_d_b_c_grad,thrust_d_b_c_grad + LSTM_size*1,unary_op);



	norm_clip_GPU_v2_p1(thrust_d_M_i_1_grad,d_M_i_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_f_1_grad,d_M_f_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_o_1_grad,d_M_o_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_c_1_grad,d_M_c_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_i_2_grad,d_M_i_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_f_2_grad,d_M_f_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_o_2_grad,d_M_o_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_M_c_2_grad,d_M_c_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

	norm_clip_GPU_v2_p1(thrust_d_b_i_grad,d_b_i_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_f_grad,d_b_f_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_o_grad,d_b_o_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p1(thrust_d_b_c_grad,d_b_c_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);


	devSynchAll();
}


template<typename dType>
void tree_LSTM<dType>::update_global_params() {

	thrust::device_ptr<dType> thrust_d_M_i_1_grad = thrust::device_pointer_cast(d_M_i_1_grad);
	thrust::device_ptr<dType> thrust_d_M_f_1_grad = thrust::device_pointer_cast(d_M_f_1_grad);
	thrust::device_ptr<dType> thrust_d_M_o_1_grad = thrust::device_pointer_cast(d_M_o_1_grad);
	thrust::device_ptr<dType> thrust_d_M_c_1_grad = thrust::device_pointer_cast(d_M_c_1_grad);
	thrust::device_ptr<dType> thrust_d_M_i_2_grad = thrust::device_pointer_cast(d_M_i_2_grad);
	thrust::device_ptr<dType> thrust_d_M_f_2_grad = thrust::device_pointer_cast(d_M_f_2_grad);
	thrust::device_ptr<dType> thrust_d_M_o_2_grad = thrust::device_pointer_cast(d_M_o_2_grad);
	thrust::device_ptr<dType> thrust_d_M_c_2_grad = thrust::device_pointer_cast(d_M_c_2_grad);

	thrust::device_ptr<dType> thrust_d_b_i_grad= thrust::device_pointer_cast(d_b_i_grad);
	thrust::device_ptr<dType> thrust_d_b_f_grad= thrust::device_pointer_cast(d_b_f_grad);
	thrust::device_ptr<dType> thrust_d_b_o_grad= thrust::device_pointer_cast(d_b_o_grad);
	thrust::device_ptr<dType> thrust_d_b_c_grad= thrust::device_pointer_cast(d_b_c_grad);

	norm_clip_GPU_v2_p2(thrust_d_M_i_1_grad,d_M_i_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_f_1_grad,d_M_f_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_o_1_grad,d_M_o_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_c_1_grad,d_M_c_1_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_i_2_grad,d_M_i_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_f_2_grad,d_M_f_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_o_2_grad,d_M_o_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_M_c_2_grad,d_M_c_2_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

	norm_clip_GPU_v2_p2(thrust_d_b_i_grad,d_b_i_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_f_grad,d_b_f_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_o_grad,d_b_o_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
	norm_clip_GPU_v2_p2(thrust_d_b_c_grad,d_b_c_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);


	gradient_update_mats<<<256,256>>>(d_M_i_1,d_M_i_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_f_1,d_M_f_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_o_1,d_M_o_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_c_1,d_M_c_1_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_i_2,d_M_i_2_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_f_2,d_M_f_2_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_o_2,d_M_o_2_grad,model->learning_rate,LSTM_size*LSTM_size);
	gradient_update_mats<<<256,256>>>(d_M_c_2,d_M_c_2_grad,model->learning_rate,LSTM_size*LSTM_size);

	gradient_update_mats<<<256,256>>>(d_b_i,d_b_i_grad,model->learning_rate,LSTM_size*1);
	gradient_update_mats<<<256,256>>>(d_b_f,d_b_f_grad,model->learning_rate,LSTM_size*1);
	gradient_update_mats<<<256,256>>>(d_b_o,d_b_o_grad,model->learning_rate,LSTM_size*1);
	gradient_update_mats<<<256,256>>>(d_b_c,d_b_c_grad,model->learning_rate,LSTM_size*1);

	devSynchAll();
}

template<typename dType>
void tree_LSTM<dType>::forward() {
		
	cudaSetDevice(device_number);
	//OPERATION
	//i_t = ((model->M_i*h_t_below + model->W_hi*h_t_prev).colwise() + model->b_i).array().unaryExpr(sigmoid_functor());
	dType alpha =1;
	dType beta = 0;

	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_i_1,LSTM_size,
		d_child_ht_1,LSTM_size,&beta,d_temp1,LSTM_size),"Forward prop i_t temp1 failed\n");


	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_i_2,LSTM_size,
		d_child_ht_2,LSTM_size,&beta,d_temp2,LSTM_size),"Forward prop i_t temp2 failed\n");

	forward_sigmoid_kernel<<<kernel,threads_per_block,0,s0>>>(d_i_t,d_temp1,d_temp2,d_b_i,LSTM_size);
	CUDA_GET_LAST_ERROR("i_t tree LSTM");


	//OPERATION
	//f_t = ((model->M_f*temp_mat + model->W_hf*h_t_prev).colwise() + model->b_f).array().unaryExpr(sigmoid_functor());
	alpha =1;
	beta = 0;

	//first forget gate
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_f_1,LSTM_size,
		d_child_ht_1,LSTM_size,&beta,d_temp3,LSTM_size),"Forward prop f_t temp3 failed\n");

	forward_sigmoid_kernel_small<<<kernel,threads_per_block,0,s0>>>(d_f_t_1,d_temp3,d_b_f,LSTM_size);


	//second forget gate
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_f_2,LSTM_size,
		d_child_ht_2,LSTM_size,&beta,d_temp4,LSTM_size),"Forward prop f_t temp4 failed\n");

	forward_sigmoid_kernel_small<<<kernel,threads_per_block,0,s0>>>(d_f_t_2,d_temp4,d_b_f,LSTM_size);


	//OPERATION
	//c_prime_t_tanh = ((model->M_c*temp_mat + model->W_hc*h_t_prev).colwise() + model->b_c).array().unaryExpr(tanh_functor());
	alpha =1;
	beta = 0;
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_c_1,LSTM_size,
		d_child_ht_1,LSTM_size,&beta,d_temp5,LSTM_size),"Forward prop c_prime_t_tanh temp5 failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_c_2,LSTM_size,
		d_child_ht_2,LSTM_size,&beta,d_temp6,LSTM_size),"Forward prop c_prime_t_tanh temp6 failed\n");

	forward_tanh_kernel<<<kernel,threads_per_block,0,s0>>>(d_c_prime_t_tanh,d_temp5,d_temp6,d_b_c,LSTM_size);
	CUDA_GET_LAST_ERROR("c_prime_t_tanh");


	//OPERATION
	//USING STREAMS 7 and 8
	//o_t = ((model->M_o*temp_mat + model->W_ho*h_t_prev).colwise() + model->b_o).unaryExpr(sigmoid_functor());
	alpha = 1;
	beta = 0;
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_o_1,LSTM_size,
		d_child_ht_1,LSTM_size,&beta,d_temp7,LSTM_size),"Forward prop o_t temp1 failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,d_M_o_2,LSTM_size,
		d_child_ht_2,LSTM_size,&beta,d_temp8,LSTM_size),"Forward prop o_t temp2 failed ZZZZZZZ\n");

	forward_sigmoid_kernel<<<kernel,threads_per_block,0,s0>>>(d_o_t,d_temp7,d_temp8,d_b_o,LSTM_size);
	CUDA_GET_LAST_ERROR("o_t");


	//OPERATION
	//FOR NOW THE REST ARE USING THE DEFAULT STREAM
	//c_t = ((f_t.array())*(c_t_prev.array())).matrix() + (i_t.array()*(c_prime_t_tanh.array())).matrix();
	forward_c_t_kernel_tree<<<kernel,threads_per_block,0,s0>>>(d_c_t,d_f_t_1,d_child_ct_1,d_f_t_2,d_child_ct_2,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("c_t");

	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,s0>>>(d_c_t,BZ_CUDA::cell_clip_threshold,LSTM_size*minibatch_size);
	}
	//cudaDeviceSynchronize();
	//OPERATION
	//h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));
	forward_h_t_kernel<<<kernel,threads_per_block,0,s0>>>(d_h_t,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("h_t");
}



template<typename dType>
void tree_LSTM<dType>::backward() {

	cudaSetDevice(device_number);

	dType alpha = 1;
	dType beta = 1;

	//OPERATION
	//d_ERRt_ct.transpose() = d_ERRnTOt_ht.transpose().array() * (o_t.array()*(1-(c_t).array().unaryExpr(tanh_sq_functor())));
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	d_ERRt_ct_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRt_ct,d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP c_t");


	//OPERATION
	//d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ct,LSTM_size,
		&beta,d_d_ERRt_ct,LSTM_size,d_d_ERRnTOt_ct,LSTM_size),"backprop addition failed, d_ERRnTOt_ct \n");


	//STARTING FROM THIS POINT STREAMS WILL BE USED
	//OPERATION
	//d_ERRnTOt_ot.transpose() = d_ERRnTOt_ht.transpose().array()*( c_t.array().unaryExpr(tanh_functor()) )*o_t*(1-o_t);
	d_ERRnTOt_ot_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_ot,d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP o_tn");

	//OPERATION
	//d_ERRnTOt_ft.transpose() = d_ERRnTOt_ct.transpose().array()*(c_t_prev.array())*f_t*(1-f_t);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_ft_1,d_d_ERRnTOt_ct,d_child_ct_1,d_f_t_1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");

	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_ft_2,d_d_ERRnTOt_ct,d_child_ct_2,d_f_t_2,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");


	//OPERATION
	//d_ERRnTOt_tanhcpt.transpose() = d_ERRnTOt_ct.transpose().array()*(i_t.array());
	d_ERRnTOt_tanhcpt_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_tanhcpt,d_d_ERRnTOt_ct,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("BP tanh_tn");
			

	//OPERATION
	//d_ERRnTOt_it.transpose() = d_ERRnTOt_ct.transpose().array()*(c_prime_t_tanh.array());
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_it,d_d_ERRnTOt_ct,d_c_prime_t_tanh,d_i_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP it_tn");	


	dType alpha2 = 1;
	dType beta2 = 0;
	//OPERATION
	//this is for the error being passed to the d_child_ht_1 layer
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_o_1,LSTM_size,d_d_ERRnTOt_ot,LSTM_size,&beta2,d_temp1,LSTM_size),"Error backprop temp1 htM1\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_f_1,LSTM_size,d_d_ERRnTOt_ft_1,LSTM_size,&beta2,d_temp2,LSTM_size),"Error backprop temp2 htM1\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_i_1,LSTM_size,d_d_ERRnTOt_it,LSTM_size,&beta2,d_temp3,LSTM_size),"Error backprop temp3 htM1\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_c_1,LSTM_size,d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,d_temp4,LSTM_size),"Error backprop temp4 htM1\n");

	add_four_matrices_kernel<<< kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_h1,d_temp1,d_temp2,d_temp3,d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1 below");

	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,s0>>>(d_d_ERRnTOt_h1,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_o_2,LSTM_size,d_d_ERRnTOt_ot,LSTM_size,&beta2,d_temp1,LSTM_size),"Error backprop temp1 htM1\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_f_2,LSTM_size,d_d_ERRnTOt_ft_2,LSTM_size,&beta2,d_temp2,LSTM_size),"Error backprop temp2 htM1\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_i_2,LSTM_size,d_d_ERRnTOt_it,LSTM_size,&beta2,d_temp3,LSTM_size),"Error backprop temp3 htM1\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,d_M_c_2,LSTM_size,d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,d_temp4,LSTM_size),"Error backprop temp4 htM1\n");

	add_four_matrices_kernel<<< kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_h2,d_temp1,d_temp2,d_temp3,d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1 below");

	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,s0>>>(d_d_ERRnTOt_h2,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}

	

	//OPERATION
	//USING STREAM 10
	//d_ERRnTOt_ctM1.transpose() = (d_ERRnTOt_ct.transpose().array()*f_t.array());
	elementwise_mult_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_ct,d_f_t_1,d_d_ERRnTOt_c1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");

	elementwise_mult_kernel<<<kernel,threads_per_block,0,s0>>>(d_d_ERRnTOt_ct,d_f_t_2,d_d_ERRnTOt_c2,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");


	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,s0>>>(d_d_ERRnTOt_c1,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,s0>>>(d_d_ERRnTOt_c2,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}



	//----------------------------------------------------------------------now computing gradients----------------------------------------------------------------------


	//OPERATION
	//model->W_hi_grad.noalias() += (h_t_prev*(d_ERRnTOt_it.array() * i_t.transpose().array()*(1-i_t.transpose().array())).matrix()).transpose();
	//model->W_hf_grad.noalias() += (h_t_prev*(d_ERRnTOt_ft.array()*f_t.transpose().array()*(1-f_t.transpose().array())).matrix()).transpose();
	//model->W_hc_grad.noalias() += (h_t_prev*(d_ERRnTOt_ct.array()*(i_t.transpose().array())*(1-c_prime_t_tanh.transpose().array().square())).matrix()).transpose();
	//model->W_ho_grad.noalias() += (h_t_prev*(d_ERRnTOt_ot.array()*o_t.transpose().array()*(1-o_t.transpose().array())).matrix()).transpose();
	alpha = 1;
	beta = 1;

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_it,LSTM_size,d_child_ht_1,LSTM_size,&beta,d_M_i_1_grad,LSTM_size),"Backprop W_hi grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_it,LSTM_size,d_child_ht_2,LSTM_size,&beta,d_M_i_2_grad,LSTM_size),"Backprop W_hi grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_ft_1,LSTM_size,d_child_ht_1,LSTM_size,&beta,d_M_f_1_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_ft_2,LSTM_size,d_child_ht_2,LSTM_size,&beta,d_M_f_2_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_tanhcpt,LSTM_size,d_child_ht_1,LSTM_size,&beta,d_M_c_1_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_tanhcpt,LSTM_size,d_child_ht_2,LSTM_size,&beta,d_M_c_2_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_ot,LSTM_size,d_child_ht_1,LSTM_size,&beta,d_M_o_1_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		d_d_ERRnTOt_ot,LSTM_size,d_child_ht_2,LSTM_size,&beta,d_M_o_2_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");


	//OPERATION
	//USING STREAMS 19,20,21,22
	//b_i_grad.noalias() += ((d_ERRnTOt_it.array() * (i_t.array() * (1-i_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	//b_f_grad.noalias() += ((d_ERRnTOt_ft.array() * (f_t.array() * (1-f_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	//b_c_grad.noalias() += (d_ERRnTOt_tanhcpt.array() * (1-c_prime_t_tanh.array().square()).matrix().transpose().array()).colwise().sum().matrix().transpose();
	//b_o_grad.noalias() += ((d_ERRnTOt_ot.array() * (o_t.array() * (1-o_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOt_it,LSTM_size,
		d_ones_minibatch,1,&beta,d_b_i_grad,1),"backprop b_i_grad failed\n");
	
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOt_ft_1,LSTM_size,
		d_ones_minibatch,1,&beta,d_b_f_grad,1),"backprop b_f_grad failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOt_ft_2,LSTM_size,
		d_ones_minibatch,1,&beta,d_b_f_grad,1),"backprop b_f_grad failed\n");
	
	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOt_ot,LSTM_size,
		d_ones_minibatch,1,&beta,d_b_o_grad,1),"backprop b_o_grad failed\n");

	cublasSetStream(handle,s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOt_tanhcpt,LSTM_size,
		d_ones_minibatch,1,&beta,d_b_c_grad,1),"backprop b_c_grad failed\n");

}



template<typename dType>
void tree_LSTM<dType>::check_all_gradients(dType epsilon) {
	std::cout << "--------------------------TREE LSTM COMBINER---------------------------------\n";
	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_i_1\n";
	check_gradient_GPU(epsilon,d_M_i_1,d_M_i_1_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_f_1\n";
	check_gradient_GPU(epsilon,d_M_f_1,d_M_f_1_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_o_1\n";
	check_gradient_GPU(epsilon,d_M_o_1,d_M_o_1_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_c_1\n";
	check_gradient_GPU(epsilon,d_M_c_1,d_M_c_1_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_i_2\n";
	check_gradient_GPU(epsilon,d_M_i_2,d_M_i_2_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_f_2\n";
	check_gradient_GPU(epsilon,d_M_f_2,d_M_f_2_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_o_2\n";
	check_gradient_GPU(epsilon,d_M_o_2,d_M_o_2_grad,LSTM_size,LSTM_size,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR M_c_2\n";
	check_gradient_GPU(epsilon,d_M_c_2,d_M_c_2_grad,LSTM_size,LSTM_size,device_number);


	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR b_i\n";
	check_gradient_GPU(epsilon,d_b_i,d_b_i_grad,LSTM_size,1,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR b_f\n";
	check_gradient_GPU(epsilon,d_b_f,d_b_f_grad,LSTM_size,1,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR b_o\n";
	check_gradient_GPU(epsilon,d_b_o,d_b_o_grad,LSTM_size,1,device_number);

	cudaSetDevice(device_number);
	std::cout << "GRADIENT CHECKING FOR b_c\n";
	check_gradient_GPU(epsilon,d_b_c,d_b_c_grad,LSTM_size,1,device_number);

}






template<typename dType>
void tree_LSTM<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols,int gpu_index) {
	cudaSetDevice(gpu_index);
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->model->getError(true);
			cudaSetDevice(gpu_index);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->model->getError(true);
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



