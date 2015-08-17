
template<typename dType>
void Input_To_Hidden_Layer<dType>::init_Input_To_Hidden_Layer_CPU(int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed)
{
	#ifdef CPU_DEBUG
	W.resize(LSTM_size,vocab_size);
 	initMatrix(W);
	M_i.resize(LSTM_size,LSTM_size);
	initMatrix(M_i);
	M_f.resize(LSTM_size,LSTM_size);
	initMatrix(M_f);
	M_o.resize(LSTM_size,LSTM_size);
	initMatrix(M_o);
	M_c.resize(LSTM_size,LSTM_size);
	initMatrix(M_c);

	//Initialize all of the weights

	W_hi.resize(LSTM_size,LSTM_size);
	initMatrix(W_hi);
	b_i.resize(LSTM_size,1);
	initMatrix(b_i);

	W_hf.resize(LSTM_size,LSTM_size);
	initMatrix(W_hf);
	b_f.resize(LSTM_size,1);
	initMatrix(b_f);

	W_hc.resize(LSTM_size,LSTM_size);
	initMatrix(W_hc);
	b_c.resize(LSTM_size,1);
	initMatrix(b_c);

	W_ho.resize(LSTM_size,LSTM_size);
	initMatrix(W_ho);
	b_o.resize(LSTM_size,1);
	initMatrix(b_o);
	
	//Initialize the gradients here
	W_hi_grad.setZero(LSTM_size,LSTM_size);
	b_i_grad.setZero(LSTM_size,1);

	W_hf_grad.setZero(LSTM_size,LSTM_size);
	b_f_grad.setZero(LSTM_size,1);

	W_hc_grad.setZero(LSTM_size,LSTM_size);
	b_c_grad.setZero(LSTM_size,1);

	W_ho_grad.setZero(LSTM_size,LSTM_size);
	b_o_grad.setZero(LSTM_size,1);

	W_grad.setZero(LSTM_size,vocab_size);
	M_i_grad.setZero(LSTM_size,LSTM_size);
	M_f_grad.setZero(LSTM_size,LSTM_size);
	M_o_grad.setZero(LSTM_size,LSTM_size);
	M_c_grad.setZero(LSTM_size,LSTM_size);

	//Initalize the initial hidden state and cell state vector
	init_hidden_vector.setZero(LSTM_size,minibatch_size);
	init_cell_vector.setZero(LSTM_size,minibatch_size);

	//Initialize the messages being passed into first cells for backprop
	init_d_ERRnTOtp1_ht.setZero(minibatch_size,LSTM_size); //Initial hidden state vector
	init_d_ERRnTOtp1_ct.setZero(minibatch_size,LSTM_size); //Initial hidden state vector
	#endif

	// //Initialize the vector of LSTM nodes to longest sentence
	// nodes.clear();
	// for(int i=0;i < longest_sent; i++) {
	// 	nodes.push_back(LSTM_IH_Node<dType>(LSTM_size,minibatch_size,vocab_size,this));
	// }

	// //Set the debug mode
	// debug = debug_temp;
	// this->minibatch_size = minibatch_size;
	// this->learning_rate = learning_rate;
	// this->clip_gradients = clip_gradients;
	// this->norm_clip = norm_clip;
	// this->model = model;
	// gen.seed(seed);
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::init_Input_To_Hidden_Layer_GPU(int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed)
{

	full_matrix_setup(&h_W_ho,&d_W_ho,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hf,&d_W_hf,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hi,&d_W_hi,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hc,&d_W_hc,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hi_grad,&d_W_hi_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hf_grad,&d_W_hf_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_hc_grad,&d_W_hc_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_W_ho_grad,&d_W_ho_grad,LSTM_size,LSTM_size);

	full_matrix_setup(&h_M_i,&d_M_i,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_f,&d_M_f,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_o,&d_M_o,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_c,&d_M_c,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_i_grad,&d_M_i_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_f_grad,&d_M_f_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_o_grad,&d_M_o_grad,LSTM_size,LSTM_size);
	full_matrix_setup(&h_M_c_grad,&d_M_c_grad,LSTM_size,LSTM_size);

	full_matrix_setup(&h_b_i,&d_b_i,LSTM_size,1);
	full_matrix_setup(&h_b_f,&d_b_f,LSTM_size,1);
	full_matrix_setup(&h_b_c,&d_b_c,LSTM_size,1);
	full_matrix_setup(&h_b_o,&d_b_o,LSTM_size,1);
	full_matrix_setup(&h_b_i_grad,&d_b_i_grad,LSTM_size,1);
	full_matrix_setup(&h_b_f_grad,&d_b_f_grad,LSTM_size,1);
	full_matrix_setup(&h_b_c_grad,&d_b_c_grad,LSTM_size,1);
	full_matrix_setup(&h_b_o_grad,&d_b_o_grad,LSTM_size,1);


	full_matrix_setup(&h_W,&d_W,LSTM_size,vocab_size);
	full_matrix_setup(&h_W_grad,&d_W_grad,LSTM_size,vocab_size);

	input_vocab_size = vocab_size;

	full_matrix_setup_0(&h_init_hidden_vector,&d_init_hidden_vector,LSTM_size,minibatch_size);
	full_matrix_setup_0(&h_init_cell_vector,&d_init_cell_vector,LSTM_size,minibatch_size);
	full_matrix_setup_0(&h_init_d_ERRnTOtp1_ht,&d_init_d_ERRnTOtp1_ht,LSTM_size,minibatch_size);
	full_matrix_setup_0(&h_init_d_ERRnTOtp1_ct,&d_init_d_ERRnTOtp1_ct,LSTM_size,minibatch_size);

	full_matrix_setup(&h_temp1,&d_temp1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp2,&d_temp2,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp3,&d_temp3,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp4,&d_temp4,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp5,&d_temp5,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp6,&d_temp6,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp7,&d_temp7,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp8,&d_temp8,LSTM_size,minibatch_size);

	full_matrix_setup_0(&h_input_vocab_indicies,&d_input_vocab_indicies,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indices_full,&d_input_vocab_indices_full,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indices_01_full,&d_input_vocab_indices_01_full,minibatch_size,longest_sent);
	full_matrix_setup_0(&h_input_vocab_indicies_Wgrad,&d_input_vocab_indicies_Wgrad,minibatch_size,longest_sent);

	//Set to zero
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_zeros, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed zeros\n");
	cudaMemset(d_zeros,0,LSTM_size*minibatch_size*sizeof(dType));

	//set to all ones
	full_vector_setup_ones(&h_ones_minibatch,&d_ones_minibatch,minibatch_size);

	//get device pointers
	thrust_d_W_ho_grad = thrust::device_pointer_cast(d_W_ho_grad); 
	thrust_d_W_hf_grad = thrust::device_pointer_cast(d_W_hf_grad);
	thrust_d_W_hi_grad = thrust::device_pointer_cast(d_W_hi_grad); 
	thrust_d_W_hc_grad = thrust::device_pointer_cast(d_W_hc_grad);

	thrust_d_M_i_grad = thrust::device_pointer_cast(d_M_i_grad);
	thrust_d_M_f_grad = thrust::device_pointer_cast(d_M_f_grad);
	thrust_d_M_o_grad = thrust::device_pointer_cast(d_M_o_grad);
	thrust_d_M_c_grad = thrust::device_pointer_cast(d_M_c_grad);

	//Eventually this should be removed, since a custom reduction kernel does this
	thrust_d_W_grad = thrust::device_pointer_cast(d_W_grad);

	thrust_d_b_i_grad = thrust::device_pointer_cast(d_b_i_grad);
	thrust_d_b_f_grad = thrust::device_pointer_cast(d_b_f_grad);
	thrust_d_b_c_grad = thrust::device_pointer_cast(d_b_c_grad);
	thrust_d_b_o_grad = thrust::device_pointer_cast(d_b_o_grad);

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");


	//for saving space in the LSTM
	full_matrix_setup(&h_d_ERRnTOt_ht,&d_d_ERRnTOt_ht,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRt_ct,&d_d_ERRt_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ct,&d_d_ERRnTOt_ct,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ot,&d_d_ERRnTOt_ot,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ft,&d_d_ERRnTOt_ft,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_tanhcpt,&d_d_ERRnTOt_tanhcpt,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_it,&d_d_ERRnTOt_it,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_htM1,&d_d_ERRnTOt_htM1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRnTOt_ctM1,&d_d_ERRnTOt_ctM1,LSTM_size,minibatch_size);

	//debug
	#ifdef CPU_DEBUG
	copy_to_eigen(W_ho,h_W_ho);
	copy_to_eigen(W_hf,h_W_hf);
	copy_to_eigen(W_hi,h_W_hi);
	copy_to_eigen(W_hc,h_W_hc);
	copy_to_eigen(W,h_W);
	copy_to_eigen(M_i,h_M_i);
	copy_to_eigen(M_f,h_M_f);
	copy_to_eigen(M_o,h_M_o);
	copy_to_eigen(M_c,h_M_c);
	copy_to_eigen(b_i,h_b_i);
	copy_to_eigen(b_f,h_b_f);
	copy_to_eigen(b_c,h_b_c);
	copy_to_eigen(b_o,h_b_o);
	#endif

	clear_gradients(true);

	cudaDeviceSynchronize();
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::init_Input_To_Hidden_Layer(int LSTM_size,int minibatch_size,int vocab_size,
 		int longest_sent,bool debug_temp,dType learning_rate,bool clip_gradients,dType norm_clip,
 		struct neuralMT_model<precision> *model,int seed)
{

	//Set the debug mode
	debug = debug_temp;
	this->minibatch_size = minibatch_size;
	this->learning_rate = learning_rate;
	this->clip_gradients = clip_gradients;
	this->norm_clip = norm_clip;
	this->model = model;
	this->LSTM_size = LSTM_size;
	this->longest_sent = longest_sent;
	gen.seed(seed);

	init_Input_To_Hidden_Layer_CPU(LSTM_size,minibatch_size,vocab_size,longest_sent,debug_temp,learning_rate,
		clip_gradients,norm_clip,model,seed);

	init_Input_To_Hidden_Layer_GPU(LSTM_size,minibatch_size,vocab_size,
 		longest_sent,debug_temp,learning_rate,clip_gradients,norm_clip,
 		model,seed);

	//Initialize the vector of LSTM nodes to longest sentence
	nodes.clear();
	for(int i=0;i < longest_sent; i++) {
		nodes.push_back(LSTM_IH_Node<dType>(LSTM_size,minibatch_size,vocab_size,this));
	}
}



template<typename dType>
void Input_To_Hidden_Layer<dType>::clear_gradients(bool init) {
	#ifdef CPU_DEBUG
	clear_gradients_CPU();
	#endif
	clear_gradients_GPU(init);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::clear_gradients_CPU() {

	W_hi_grad.setZero();
	b_i_grad.setZero();

	W_hf_grad.setZero();
	b_f_grad.setZero();

	W_hc_grad.setZero();
	b_c_grad.setZero();

	W_ho_grad.setZero();
	b_o_grad.setZero();

	W_grad.setZero();
	M_i_grad.setZero();
	M_f_grad.setZero();
	M_o_grad.setZero();
	M_c_grad.setZero();
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::clear_gradients_GPU(bool init) {
	cudaDeviceSynchronize();
	cudaMemsetAsync(d_W_hi_grad, 0, LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s0);
	cudaMemsetAsync(d_b_i_grad, 0, LSTM_size*1*sizeof(dType),ih_layer_info.s1);

	cudaMemsetAsync(d_W_hf_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s2);
	cudaMemsetAsync(d_b_f_grad,0,LSTM_size*1*sizeof(dType),ih_layer_info.s3);

	cudaMemsetAsync(d_W_hc_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s4);
	cudaMemsetAsync(d_b_c_grad,0,LSTM_size*1*sizeof(dType),ih_layer_info.s5);

	cudaMemsetAsync(d_W_ho_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s6);
	cudaMemsetAsync(d_b_o_grad,0,LSTM_size*1*sizeof(dType),ih_layer_info.s7);

	//CHANGE THIS TO NON NAIVE KERNEL
	if(init) {
		cudaMemset(d_W_grad,0,LSTM_size*input_vocab_size*sizeof(dType));
	}
	else {
		int threads_per_block = 256;
		int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		zero_W_gradient<<<kernel,threads_per_block ,0,ih_layer_info.s8>>>(d_W_grad,d_input_vocab_indicies_Wgrad,LSTM_size,w_grad_len);
	}


	cudaMemsetAsync(d_M_i_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s9);
	cudaMemsetAsync(d_M_f_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s10);
	cudaMemsetAsync(d_M_o_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s11);
	cudaMemsetAsync(d_M_c_grad,0,LSTM_size*LSTM_size*sizeof(dType),ih_layer_info.s12);
	cudaDeviceSynchronize();
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::update_weights() {
	#ifdef CPU_DEBUG
	update_weights_CPU();
	#endif
	update_weights_GPU();
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::update_weights_GPU() {

	scale_functor unary_op(minibatch_size);

	thrust::for_each(thrust_d_W_hi_grad,thrust_d_W_hi_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_i_grad,thrust_d_b_i_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_hf_grad,thrust_d_W_hf_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_f_grad,thrust_d_b_f_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_hc_grad,thrust_d_W_hc_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_c_grad,thrust_d_b_c_grad + LSTM_size*1,unary_op);

	thrust::for_each(thrust_d_W_ho_grad,thrust_d_W_ho_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_b_o_grad,thrust_d_b_o_grad + LSTM_size*1,unary_op);


	// dType *d_W_grad_DEBUG;
	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_W_grad_DEBUG, LSTM_size*input_vocab_size*sizeof(dType)),"GPU memory allocation failed\n");
	// cudaMemcpy(d_W_grad_DEBUG,d_W_grad,LSTM_size*input_vocab_size*sizeof(dType),cudaMemcpyDeviceToDevice);
	// CUDA_GET_LAST_ERROR();

	int threads_per_block = 256;
	int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
	dim3 kernel(num_block,256,1);
	scale_W_gradient<<<kernel,threads_per_block>>>(d_W_grad,d_input_vocab_indicies_Wgrad,LSTM_size,((dType)1.0)/minibatch_size ,w_grad_len);
	CUDA_GET_LAST_ERROR();

	//thrust::for_each(thrust_d_W_grad,thrust_d_W_grad + LSTM_size*input_vocab_size,unary_op);

	//COMAPRE THE TWO WGRADS
	// cudaDeviceSynchronize();
	// check_GPU_GPU(d_W_grad,d_W_grad_DEBUG,(dType)0.00000000001,LSTM_size,input_vocab_size,"W GRAD SCALE");
	// cudaFree(d_W_grad_DEBUG);

	// thrust::device_ptr<int> debug_ptr2 = thrust::device_pointer_cast(d_input_vocab_indicies_Wgrad);
	// for(int i=0; i < w_grad_len; i++) {
	// 	std::cout << debug_ptr2[i] << " ";
	// }

	// std::cout << "\n";


	thrust::for_each(thrust_d_M_i_grad,thrust_d_M_i_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_f_grad,thrust_d_M_f_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_o_grad,thrust_d_M_o_grad + LSTM_size*LSTM_size,unary_op);
	thrust::for_each(thrust_d_M_c_grad,thrust_d_M_c_grad + LSTM_size*LSTM_size,unary_op);

	if(clip_gradients) {
		// norm_clip_GPU(thrust_d_W_hi_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_b_i_grad,norm_clip,LSTM_size*1);

		// norm_clip_GPU(thrust_d_W_hf_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_b_f_grad,norm_clip,LSTM_size*1);

		// norm_clip_GPU(thrust_d_W_hc_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_b_c_grad,norm_clip,LSTM_size*1);

		// norm_clip_GPU(thrust_d_W_ho_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_b_o_grad,norm_clip,LSTM_size*1);

		norm_clip_GPU_v2(thrust_d_W_hi_grad,d_W_hi_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_W_hf_grad,d_W_hf_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_W_hc_grad,d_W_hc_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_W_ho_grad,d_W_ho_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

		norm_clip_GPU_v2(thrust_d_b_i_grad,d_b_i_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_f_grad,d_b_f_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_c_grad,d_b_c_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_b_o_grad,d_b_o_grad,norm_clip,LSTM_size*1,d_temp_result,d_result);

		//norm_clip_GPU(thrust_d_W_grad,norm_clip,LSTM_size*input_vocab_size);
		// norm_clip_W_GPU(thrust_d_W_grad,d_W_grad,
		// 	d_input_vocab_indicies_Wgrad,norm_clip,w_grad_len,LSTM_size,input_vocab_size*LSTM_size);

		norm_clip_W_GPU_v2(d_temp_result,d_W_grad,
			d_input_vocab_indicies_Wgrad,norm_clip,w_grad_len,LSTM_size); 

		// norm_clip_GPU(thrust_d_M_i_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_M_f_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_M_o_grad,norm_clip,LSTM_size*LSTM_size);
		// norm_clip_GPU(thrust_d_M_c_grad,norm_clip,LSTM_size*LSTM_size);

		norm_clip_GPU_v2(thrust_d_M_i_grad,d_M_i_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_f_grad,d_M_f_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_o_grad,d_M_o_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);
		norm_clip_GPU_v2(thrust_d_M_c_grad,d_M_c_grad,norm_clip,LSTM_size*LSTM_size,d_temp_result,d_result);

	}

	dType alpha = learning_rate;
	dType beta = 1;

	cudaDeviceSynchronize();

	//normal matrices
	cublasSetStream(ih_layer_info.handle,ih_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_W_hi_grad, LSTM_size, &beta, d_W_hi, LSTM_size, d_W_hi, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s2);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_W_hf_grad, LSTM_size, &beta, d_W_hf, LSTM_size, d_W_hf, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s4);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_W_hc_grad, LSTM_size, &beta, d_W_hc, LSTM_size, d_W_hc, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s6);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_W_ho_grad, LSTM_size, &beta, d_W_ho, LSTM_size, d_W_ho, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s9);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_M_i_grad, LSTM_size, &beta, d_M_i, LSTM_size, d_M_i, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s10);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_M_f_grad, LSTM_size, &beta, d_M_f, LSTM_size, d_M_f, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s12);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_M_c_grad, LSTM_size, &beta, d_M_c, LSTM_size, d_M_c, LSTM_size),"CUBLAS addition update parameter failed\n");

	cublasSetStream(ih_layer_info.handle,ih_layer_info.s11);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, LSTM_size, &alpha, 
		d_M_o_grad, LSTM_size, &beta, d_M_o, LSTM_size, d_M_o, LSTM_size),"CUBLAS addition update parameter failed\n");



	add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s1>>>(d_b_i,d_b_i_grad,learning_rate,LSTM_size*1);
	CUDA_GET_LAST_ERROR();
	add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s3>>>(d_b_f,d_b_f_grad,learning_rate,LSTM_size*1);
	CUDA_GET_LAST_ERROR();
	add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s5>>>(d_b_c,d_b_c_grad,learning_rate,LSTM_size*1);
	CUDA_GET_LAST_ERROR();
	add_grad_vecs<<<(LSTM_size+256-1)/256,256,0,ih_layer_info.s7>>>(d_b_o,d_b_o_grad,learning_rate,LSTM_size*1);
	CUDA_GET_LAST_ERROR();

	// CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, 1, &alpha, 
	// 	d_b_i_grad, LSTM_size, &beta, d_b_i, LSTM_size, d_b_i, LSTM_size),"CUBLAS addition update parameter failed\n");

	// CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, 1, &alpha, 
	// 	d_b_f_grad, LSTM_size, &beta, d_b_f, LSTM_size, d_b_f, LSTM_size),"CUBLAS addition update parameter failed\n");

	// CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, 1, &alpha, 
	// 	d_b_c_grad, LSTM_size, &beta, d_b_c, LSTM_size, d_b_c, LSTM_size),"CUBLAS addition update parameter failed\n");

	// CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, 1, &alpha, 
	// 	d_b_o_grad, LSTM_size, &beta, d_b_o, LSTM_size, d_b_o, LSTM_size),"CUBLAS addition update parameter failed\n");



	//special W 
	update_W_gradient<<<kernel,threads_per_block,0,ih_layer_info.s8>>>(d_W,d_W_grad,d_input_vocab_indicies_Wgrad,learning_rate,LSTM_size,w_grad_len);
	CUDA_GET_LAST_ERROR();

	//THIS IS FOR DEBUGGING
	// CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(ih_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,LSTM_size, input_vocab_size, &alpha, 
	// 	d_W_grad, LSTM_size, &beta, d_W, LSTM_size, d_W, LSTM_size),"CUBLAS addition update parameter failed\n");

	#ifdef CPU_DEBUG
	cudaDeviceSynchronize();
	eigen_check_thrust_ptr(W_hi,d_W_hi,"W_hi parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(W_hf,d_W_hf,"W_hf parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(W_hc,d_W_hc,"W_hc parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(W_ho,d_W_ho,"W_ho parameter updates in Input to Hidden layer",(dType)0.000000001);

	eigen_check_thrust_ptr(M_i,d_M_i,"M_i parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(M_f,d_M_f,"M_f parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(M_c,d_M_c,"M_c parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(M_o,d_M_o,"M_o parameter updates in Input to Hidden layer",(dType)0.000000001);

	eigen_check_thrust_ptr(b_i,d_b_i,"b_i parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(b_f,d_b_f,"b_f parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(b_c,d_b_c,"b_c parameter updates in Input to Hidden layer",(dType)0.000000001);
	eigen_check_thrust_ptr(b_o,d_b_o,"b_o parameter updates in Input to Hidden layer",(dType)0.000000001);

	eigen_check_thrust_ptr(W,d_W,"W parameter updates in Input to Hidden layer",(dType)0.000000001);
	#endif

}



//Update the model parameters
template<typename dType>
void Input_To_Hidden_Layer<dType>::update_weights_CPU() {

	W_hi_grad = ( ((dType)1.0)/minibatch_size)*W_hi_grad;
	b_i_grad = ( ((dType)1.0)/minibatch_size)*b_i_grad;

	W_hf_grad = ( ((dType)1.0)/minibatch_size)*W_hf_grad;
	b_f_grad = ( ((dType)1.0)/minibatch_size)*b_f_grad;

	W_hc_grad = ( ((dType)1.0)/minibatch_size)*W_hc_grad;
	b_c_grad = ( ((dType)1.0)/minibatch_size)*b_c_grad;

	W_ho_grad = ( ((dType)1.0)/minibatch_size)*W_ho_grad;
	b_o_grad = ( ((dType)1.0)/minibatch_size)*b_o_grad;

	W_grad = ( ((dType)1.0)/minibatch_size)*W_grad;
	M_i_grad = ( ((dType)1.0)/minibatch_size)*M_i_grad;
	M_f_grad = ( ((dType)1.0)/minibatch_size)*M_f_grad;
	M_o_grad = ( ((dType)1.0)/minibatch_size)*M_o_grad;
	M_c_grad = ( ((dType)1.0)/minibatch_size)*M_c_grad;

	// std::cout << "------------------W grad CPU printouts----------------------\n";
	// for(int i=0; i<LSTM_size; i++) {
	// 	for(int j=0; j<input_vocab_size; j++) {
	// 		std::cout << W_grad(i,j) << " ";
	// 	}
	// 	std::cout << "\n";
	// }

	//For gradient clipping
	if(clip_gradients) {
		computeNorm(W_hi_grad,norm_clip);
		computeNorm(b_i_grad,norm_clip);

		computeNorm(W_hf_grad,norm_clip);
		computeNorm(b_f_grad,norm_clip);

		computeNorm(W_hc_grad,norm_clip);
		computeNorm(b_c_grad,norm_clip);

		computeNorm(W_ho_grad,norm_clip);
		computeNorm(b_o_grad,norm_clip);

		computeNorm(W_grad,norm_clip);
		computeNorm(M_i_grad,norm_clip);
		computeNorm(M_f_grad,norm_clip);
		computeNorm(M_o_grad,norm_clip);
		computeNorm(M_c_grad,norm_clip);
	}

	W_hi.noalias() += (learning_rate)*W_hi_grad;
	b_i.noalias() += (learning_rate)*b_i_grad;

	W_hf.noalias() += (learning_rate)*W_hf_grad;
	b_f.noalias() += (learning_rate)*b_f_grad;

	W_hc.noalias() += (learning_rate)*W_hc_grad;
	b_c.noalias() += (learning_rate)*b_c_grad;

	W_ho.noalias() += (learning_rate)*W_ho_grad;
	b_o.noalias() += (learning_rate)*b_o_grad;

	W.noalias() += (learning_rate)*W_grad;
	M_i.noalias() += (learning_rate)*M_i_grad;
	M_f.noalias() += (learning_rate)*M_f_grad;
	M_o.noalias() += (learning_rate)*M_o_grad;
	M_c.noalias() += (learning_rate)*M_c_grad;
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_all_gradients(dType epsilon) {
	#ifdef CPU_DEBUG
	check_all_gradients_CPU(epsilon);
	#endif
	check_all_gradients_GPU(epsilon);
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_all_gradients_CPU(dType epsilon) {
		std::cout << "--------------------GRADIENT CHECKING FOR INPUT LAYER CPU-------------------------\n";
		std::cout << "GRADIENT CHECKING FOR W_hi\n";
		check_gradient(epsilon,W_hi,W_hi_grad);
		
		std::cout << "GRADIENT CHECKING FOR W_hf\n";
		check_gradient(epsilon,W_hf,W_hf_grad);

		std::cout << "GRADIENT CHECKING FOR W_ho\n";
		check_gradient(epsilon,W_ho,W_ho_grad);

		std::cout << "GRADIENT CHECKING FOR W_hc\n";
		check_gradient(epsilon,W_hc,W_hc_grad);

		std::cout << "GRADIENT CHECKING FOR b_i\n";
		check_gradient(epsilon,b_i,b_i_grad);

		std::cout << "GRADIENT CHECKING FOR b_f\n";
		check_gradient(epsilon,b_f,b_f_grad);

		std::cout << "GRADIENT CHECKING FOR b_c\n";
		check_gradient(epsilon,b_c,b_c_grad);

		std::cout << "GRADIENT CHECKING FOR b_o\n";
		check_gradient(epsilon,b_o,b_o_grad);

		std::cout << "GRADIENT CHECKING FOR M_i\n";
		check_gradient(epsilon,M_i,M_i_grad);
		
		std::cout << "GRADIENT CHECKING FOR M_f\n";
		check_gradient(epsilon,M_f,M_f_grad);

		std::cout << "GRADIENT CHECKING FOR M_o\n";
		check_gradient(epsilon,M_o,M_o_grad);
		
		std::cout << "GRADIENT CHECKING FOR M_c\n";
		check_gradient(epsilon,M_c,M_c_grad);

		std::cout << "GRADIENT CHECKING FOR W\n";
		check_gradient(epsilon,W,W_grad);
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_all_gradients_GPU(dType epsilon) {
		std::cout << "--------------------GRADIENT CHECKING FOR INPUT LAYER GPU-------------------------\n";
		std::cout << "GRADIENT CHECKING FOR W_hi\n";
		check_gradient_GPU(epsilon,d_W_hi,d_W_hi_grad,LSTM_size,LSTM_size);
		
		std::cout << "GRADIENT CHECKING FOR W_hf\n";
		check_gradient_GPU(epsilon,d_W_hf,d_W_hf_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR W_ho\n";
		check_gradient_GPU(epsilon,d_W_ho,d_W_ho_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR W_hc\n";
		check_gradient_GPU(epsilon,d_W_hc,d_W_hc_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR b_i\n";
		check_gradient_GPU(epsilon,d_b_i,d_b_i_grad,LSTM_size,1);

		std::cout << "GRADIENT CHECKING FOR b_f\n";
		check_gradient_GPU(epsilon,d_b_f,d_b_f_grad,LSTM_size,1);

		std::cout << "GRADIENT CHECKING FOR b_c\n";
		check_gradient_GPU(epsilon,d_b_c,d_b_c_grad,LSTM_size,1);

		std::cout << "GRADIENT CHECKING FOR b_o\n";
		check_gradient_GPU(epsilon,d_b_o,d_b_o_grad,LSTM_size,1);

		std::cout << "GRADIENT CHECKING FOR M_i\n";
		check_gradient_GPU(epsilon,d_M_i,d_M_i_grad,LSTM_size,LSTM_size);
		
		std::cout << "GRADIENT CHECKING FOR M_f\n";
		check_gradient_GPU(epsilon,d_M_f,d_M_f_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR M_o\n";
		check_gradient_GPU(epsilon,d_M_o,d_M_o_grad,LSTM_size,LSTM_size);
		
		std::cout << "GRADIENT CHECKING FOR M_c\n";
		check_gradient_GPU(epsilon,d_M_c,d_M_c_grad,LSTM_size,LSTM_size);

		std::cout << "GRADIENT CHECKING FOR W\n";
		check_gradient_GPU(epsilon,d_W,d_W_grad,LSTM_size,input_vocab_size);
}



template<typename dType>
template<typename Derived,typename Derived3>
void Input_To_Hidden_Layer<dType>::check_gradient(dType epsilon,const Eigen::MatrixBase<Derived3> &parameter_const,const Eigen::MatrixBase<Derived> &grad) {
	UNCONST(Derived3, parameter_const, parameter);
	for(int i=0; i<grad.rows(); i++) {
		for(int j=0; j<grad.cols(); j++) {
			dType loss = 0;
			parameter(i,j) += epsilon;
			loss = model->getError(false);
			parameter(i,j) += -2*epsilon;
			loss-= model->getError(false);
			parameter(i,j)+=epsilon;
			if( (std::abs(grad(i,j) - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(grad(i,j) - loss/(2*epsilon))/(std::abs(grad(i,j)) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << grad(i,j) << "\n";
				std::cout << "Gradient difference: " << std::abs(grad(i,j) - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(grad(i,j) - loss/(2*epsilon))/(std::abs(grad(i,j)) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			//std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}



template<typename dType>
void Input_To_Hidden_Layer<dType>::dump_weights_CPU(std::ofstream &output) {

	writeMatrix(W_hi,output);
	writeMatrix(b_i,output);

	writeMatrix(W_hf,output);
	writeMatrix(b_f,output);

	writeMatrix(W_hc,output);
	writeMatrix(b_c,output);

	writeMatrix(W_ho,output);
	writeMatrix(b_o,output);

	writeMatrix(W,output);
	writeMatrix(M_i,output);
	writeMatrix(M_f,output);
	writeMatrix(M_o,output);
	writeMatrix(M_c,output);
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::dump_weights_GPU(std::ofstream &output) {

	write_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_i,LSTM_size,1,output);

	write_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_f,LSTM_size,1,output);

	write_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_c,LSTM_size,1,output);

	write_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_b_o,LSTM_size,1,output);

	write_matrix_GPU(d_W,LSTM_size,input_vocab_size,output);
	write_matrix_GPU(d_M_i,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_f,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_o,LSTM_size,LSTM_size,output);
	write_matrix_GPU(d_M_c,LSTM_size,LSTM_size,output);

	cudaMemset(d_W_hi,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_b_i,0,LSTM_size*1*sizeof(dType));

	cudaMemset(d_W_hf,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_b_f,0,LSTM_size*1*sizeof(dType));

	cudaMemset(d_W_hc,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_b_c,0,LSTM_size*1*sizeof(dType));

	cudaMemset(d_W_ho,0,LSTM_size*LSTM_size*sizeof(dType));
	cudaMemset(d_b_o,0,LSTM_size*1*sizeof(dType));
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::dump_weights(std::ofstream &output) {
	//dump_weights_CPU(output);
	dump_weights_GPU(output);
}



template<typename dType>
void Input_To_Hidden_Layer<dType>::load_weights_CPU(std::ifstream &input) {
	
	readMatrix(W_hi,input);
	readMatrix(b_i,input);

	readMatrix(W_hf,input);
	readMatrix(b_f,input);

	readMatrix(W_hc,input);
	readMatrix(b_c,input);

	readMatrix(W_ho,input);
	readMatrix(b_o,input);

	readMatrix(W,input);
	readMatrix(M_i,input);
	readMatrix(M_f,input);
	readMatrix(M_o,input);
	readMatrix(M_c,input);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::load_weights_GPU(std::ifstream &input) {
	read_matrix_GPU(d_W_hi,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_i,LSTM_size,1,input);

	read_matrix_GPU(d_W_hf,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_f,LSTM_size,1,input);

	read_matrix_GPU(d_W_hc,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_c,LSTM_size,1,input);

	read_matrix_GPU(d_W_ho,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_b_o,LSTM_size,1,input);

	read_matrix_GPU(d_W,LSTM_size,input_vocab_size,input);
	read_matrix_GPU(d_M_i,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_f,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_o,LSTM_size,LSTM_size,input);
	read_matrix_GPU(d_M_c,LSTM_size,LSTM_size,input);
}


template<typename dType>
void Input_To_Hidden_Layer<dType>::load_weights(std::ifstream &input) {
	load_weights_GPU(input);
}


template<typename dType>
template<typename Derived>
void Input_To_Hidden_Layer<dType>::initMatrix(const Eigen::MatrixBase<Derived> &input_const) {
	UNCONST(Derived,input_const,input);
	dType lower = -1.0; //Lower bound for uniform dist
	dType upper = 1.0; //Upper bound for uniform dist
	boost::uniform_real<> distribution(lower,upper);
	for(int j=0; j<input.cols(); j++) {
		for(int i=0; i<input.rows(); i++) {
			input(i,j) =  distribution(gen);
		}
	}
}



template<typename dType>
void Input_To_Hidden_Layer<dType>::prep_GPU_vocab_indices(int *h_input_vocab_indicies,int *h_input_vocab_indicies_Wgrad,int current_length,int len_W) {
	this->h_input_vocab_indicies = h_input_vocab_indicies;
	this->current_length = current_length;

	//transfer to the GPU
	cudaMemcpy(d_input_vocab_indicies, h_input_vocab_indicies, minibatch_size*current_length*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer");
	cudaMemcpy(d_input_vocab_indicies_Wgrad, h_input_vocab_indicies_Wgrad, len_W*sizeof(int), cudaMemcpyHostToDevice);
	CUDA_GET_LAST_ERROR("d_vocab indicies prep LSTM layer W_grad");

	w_grad_len = len_W;
	//std::cout << w_grad_len << "\n";

	//Launch kernel to turn into 0/1's and indicies with no -1's
	int threads_per_block = 128;
	//int blocks_per_grid = std::min(current_length,128);
	int blocks_per_grid = 128;
	vocab_to_01<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_01_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 1");

	vocab_to_nonM1<<<blocks_per_grid,threads_per_block>>>(d_input_vocab_indices_full,d_input_vocab_indicies,current_length*minibatch_size);
	CUDA_GET_LAST_ERROR("Prep vocab indicies kernel 2");

	//cudaDeviceSynchronize();

	//thrust::device_ptr<int> debug_ptr = thrust::device_pointer_cast(d_input_vocab_indicies);
	// thrust::device_ptr<int> debug_ptr_2 = thrust::device_pointer_cast(d_input_vocab_indices_full);
	// thrust::device_ptr<int> debug_ptr_3 = thrust::device_pointer_cast(d_input_vocab_indices_01_full);
	// for(int i=0; i<minibatch_size*current_length; i++) {
	// 	std::cout << h_input_vocab_indicies[i] << " | " << debug_ptr[i] << " | " << debug_ptr_2[i] << " | " << debug_ptr_3[i] <<"\n";
	// }
	// std::cout << "\n\n";
}



template<typename dType>
template<typename Derived>
void Input_To_Hidden_Layer<dType>::swap_states_decoding(const Eigen::MatrixBase<Derived> &indicies,int index,dType *d_temp_swap_vals) {

	for(int i=0; i<indicies.rows(); i++) {
		#ifdef CPU_DEBUG
		temp_swap_vals.col(i)=nodes[index].h_t.col(indicies(i));
		#endif
		cudaMemcpy(d_temp_swap_vals+i*LSTM_size,nodes[index].d_h_t+indicies(i)*LSTM_size,LSTM_size*sizeof(dType),cudaMemcpyDeviceToDevice);
	}
	#ifdef CPU_DEBUG
	nodes[index].h_t = temp_swap_vals;
	#endif
	cudaMemcpy(nodes[index].d_h_t,d_temp_swap_vals,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDeviceToDevice);

	for(int i=0; i<indicies.rows(); i++) {
		#ifdef CPU_DEBUG
		temp_swap_vals.col(i)=nodes[index].c_t.col(indicies(i));
		#endif
		cudaMemcpy(d_temp_swap_vals+i*LSTM_size,nodes[index].d_c_t+indicies(i)*LSTM_size,LSTM_size*sizeof(dType),cudaMemcpyDeviceToDevice);
	}
	#ifdef CPU_DEBUG
	nodes[index].c_t = temp_swap_vals;
	#endif
	cudaMemcpy(nodes[index].d_c_t,d_temp_swap_vals,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDeviceToDevice);
}


template<typename dType>
template<typename Derived>
void Input_To_Hidden_Layer<dType>::transfer_decoding_states(const Eigen::MatrixBase<Derived> &s_h_t,const Eigen::MatrixBase<Derived> &s_c_t) {
	#ifdef CPU_DEBUG
	for(int i=0; i<nodes[0].h_t.cols(); i++) {
		nodes[0].h_t_prev.col(i) = s_h_t;
		nodes[0].c_t_prev.col(i) = s_c_t;
	}
	#endif
}

template<typename dType>
void Input_To_Hidden_Layer<dType>::transfer_decoding_states_GPU(dType *d_h_t,dType *d_c_t) {

	for(int i=0; i<minibatch_size; i++) {
		int step = i*LSTM_size;
		CUDA_ERROR_WRAPPER(cudaMemcpy(d_init_hidden_vector+step,d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),\
			"transfer decoding states h_t memcpy failed");
		CUDA_ERROR_WRAPPER(cudaMemcpy(d_init_cell_vector+step,d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),\
			"transfer decoding states c_t memcpy failed");
	}

	nodes[0].d_h_t_prev = d_init_hidden_vector;
	nodes[0].d_c_t_prev = d_init_cell_vector;

	#ifdef CPU_DEBUG
	eigen_check_thrust_ptr(nodes[0].h_t_prev,nodes[0].d_h_t_prev,"transfer decoding states fail h_t",(dType)0.00000001);
	eigen_check_thrust_ptr(nodes[0].c_t_prev,nodes[0].d_c_t_prev,"transfer decoding states fail c_t",(dType)0.00000001);
	#endif
}




