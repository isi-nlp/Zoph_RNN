
//Constructor
template<typename dType>
LSTM_IH_Node<dType>::LSTM_IH_Node(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m,
	int index,dType *d_zeros,bool dropout, dType dropout_rate) 
{
	model = m;
	this->dropout = dropout;
	this->dropout_rate = dropout_rate;

	init_LSTM_GPU(LSTM_size,minibatch_size,vocab_size,m);

	//model = m;
	this->minibatch_size = minibatch_size;
	this->LSTM_size = LSTM_size;
	this->index = index;
	this->d_zeros = d_zeros;
}



template<typename dType>
void LSTM_IH_Node<dType>::init_LSTM_GPU(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m) {

	cudaSetDevice(model->ih_layer_info.device_number);

	full_matrix_setup(&h_o_t,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_t,&d_c_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_f_t,&d_f_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_prime_t_tanh,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_i_t,&d_i_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_sparse_lookup,&d_sparse_lookup,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t,&d_h_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,LSTM_size,minibatch_size);
	cudaMemset(d_d_ERRt_ht,0,LSTM_size*minibatch_size*sizeof(dType));


	//allocate a matric that will have values between zero and one
	if(dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	}
}


template<typename dType>
void LSTM_IH_Node<dType>::update_vectors_forward_GPU(int *d_input_vocab_indices,int *d_input_vocab_indices_01,
	dType *d_h_t_prev,dType *d_c_t_prev) 
{
	this->d_h_t_prev = d_h_t_prev;
	this->d_c_t_prev = d_c_t_prev;
	this->d_input_vocab_indices = d_input_vocab_indices;
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
}



//Update the hidden state and cell state vectors for first column in target model
template<typename dType>
void LSTM_IH_Node<dType>::update_vectors_forward_decoder(int *d_input_vocab_indices,int *d_input_vocab_indices_01) 
{
	//GPU stuff
	this->d_input_vocab_indices = d_input_vocab_indices;
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
}



template<typename dType>
void LSTM_IH_Node<dType>::forward_prop() {
	forward_prop_GPU();
}



template<typename dType>
void LSTM_IH_Node<dType>::forward_prop_GPU() {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(model->ih_layer_info.device_number);
	//cudaDeviceSynchronize();
	//cudaDeviceSynchronize();
	//OPERATION
	//USING STREAM 0
	//compute_temp_mat(model->W);
	//std::cout << "f prop node starting\n";
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	CUDA_GET_LAST_ERROR("PRE SPARSE");

	if(!model->char_cnn) {
		sparse_lookup_kernel<<< kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_sparse_lookup,model->d_W,d_input_vocab_indices,minibatch_size,LSTM_size);
	}
	else {
		cudaEventRecord(model->ih_layer_info.char_cnn_ready,model->ih_layer_info.s0);
		model->char_cnn_layer->forward(index);
		cudaSetDevice(model->ih_layer_info.device_number);
		cudaStreamWaitEvent(model->ih_layer_info.s0,model->char_cnn_layer->forward_prop_done,0);
		if(model->model->decode) {
			devSynchAll();
			cudaMemcpyAsync(d_sparse_lookup,model->char_cnn_layer->top_highway_layer->nodes[0]->d_z,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->ih_layer_info.s0);
		}
		else {
			cudaMemcpyAsync(d_sparse_lookup,model->char_cnn_layer->top_highway_layer->nodes[index]->d_z,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->ih_layer_info.s0);
		}
	}
	CUDA_GET_LAST_ERROR("SPARSE");

	if(dropout && model->model->train) {

		if(!model->model->grad_check_flag) {
			curandSetStream(model->rand_gen, model->ih_layer_info.s0);
			curandGenerateUniform_wrapper(d_dropout_mask,LSTM_size*minibatch_size,model->rand_gen);
		}
		dropout_kernel<<<256,256,0,model->ih_layer_info.s0>>>(d_dropout_mask,model->dropout_rate,d_sparse_lookup,LSTM_size*minibatch_size);
	}

	cudaEventRecord(model->ih_layer_info.sparse_forward_start,model->ih_layer_info.s0);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//debug
	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(temp_mat,d_sparse_lookup,"sparse lookup kernel in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();
	//OPERATION
	//USING STREAMS 1 and 2
	//i_t = ((model->M_i*temp_mat + model->W_hi*h_t_prev).colwise() + model->b_i).array().unaryExpr(sigmoid_functor());
	dType alpha =1;
	dType beta = 0;

	//std::cout << "i_t start\n";
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s1);
	cudaStreamWaitEvent(model->ih_layer_info.s1,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_i,LSTM_size,
		d_sparse_lookup,LSTM_size,&beta,model->d_temp1,LSTM_size),"Forward prop i_t temp1 failed\n");
	cudaEventRecord(model->ih_layer_info.i_t_part1,model->ih_layer_info.s1);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s2);
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hi,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp2,LSTM_size),"Forward prop i_t temp2 failed\n");

	#ifdef REMOVE_STREAMS
	devSynchAll(); 
	CUDA_GET_LAST_ERROR("Check PPPP");
	#endif

	if(feed_input && index!=0) {

		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif

		//std::cout << "FEED FORWARD 1\n";
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s2);
		cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_Q_i,LSTM_size,
			d_h_tild,LSTM_size,&beta,model->d_temp9,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	#ifdef REMOVE_STREAMS
	devSynchAll();
	CUDA_GET_LAST_ERROR("Check 1234");
	#endif

	CUDA_GET_LAST_ERROR("i_t for lower level LSTM P1");
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.i_t_part1,0);
	if(!feed_input || index==0) {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_b_i,LSTM_size);
	}
	else {
		//std::cout << "FEED FORWARD 2\n";
		forward_sigmoid_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_temp9,model->d_b_i,LSTM_size);
		//std::cout << "HERE\n";
	}
	CUDA_GET_LAST_ERROR("i_t for lower level LSTM");
	cudaEventRecord(model->ih_layer_info.i_t_full,model->ih_layer_info.s2);
	//std::cout << "i_t end\n";
	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(i_t,d_i_t,"i_t in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();

	#ifdef REMOVE_STREAMS
	devSynchAll();
	CUDA_GET_LAST_ERROR("i_t for lower level LSTM P2");
	#endif

	//OPERATION
	//f_t = ((model->M_f*temp_mat + model->W_hf*h_t_prev).colwise() + model->b_f).array().unaryExpr(sigmoid_functor());
	//USING STREAMS 3 and 4
	alpha =1;
	beta = 0;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s3);
	cudaStreamWaitEvent(model->ih_layer_info.s3,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_f,LSTM_size,
		d_sparse_lookup,LSTM_size,&beta,model->d_temp3,LSTM_size),"Forward prop f_t temp3 failed\n");
	cudaEventRecord(model->ih_layer_info.f_t_part1,model->ih_layer_info.s3);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s4);
	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hf,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp4,LSTM_size),"Forward prop f_t temp4 failed\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	if(feed_input && index!=0) {
		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif
		//std::cout << "FEED FORWARD 1\n";
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s4);
		cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_Q_f,LSTM_size,
			d_h_tild,LSTM_size,&beta,model->d_temp10,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.f_t_part1,0);
	if(!feed_input || index==0) {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_b_f,LSTM_size);
	}
	else {
		//std::cout << "FEED FORWARD 2\n";
		forward_sigmoid_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_temp10,model->d_b_f,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("f_t");
	cudaEventRecord(model->ih_layer_info.f_t_full,model->ih_layer_info.s4);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(f_t,d_f_t,"f_t in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();
	//OPERATION
	//USING STREAMS 5 and 6
	//c_prime_t_tanh = ((model->M_c*temp_mat + model->W_hc*h_t_prev).colwise() + model->b_c).array().unaryExpr(tanh_functor());
	alpha =1;
	beta = 0;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s5);
	cudaStreamWaitEvent(model->ih_layer_info.s5,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_c,LSTM_size,
		d_sparse_lookup,LSTM_size,&beta,model->d_temp5,LSTM_size),"Forward prop c_prime_t_tanh temp5 failed\n");
	cudaEventRecord(model->ih_layer_info.c_prime_t_tanh_part1,model->ih_layer_info.s5);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hc,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp6,LSTM_size),"Forward prop c_prime_t_tanh temp6 failed\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	if(feed_input && index!=0) {
		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif
		//std::cout << "FEED FORWARD 1\n";
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
		cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_Q_c,LSTM_size,
			d_h_tild,LSTM_size,&beta,model->d_temp11,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.c_prime_t_tanh_part1,0);
	if(!feed_input || index==0) {
		forward_tanh_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_b_c,LSTM_size);
	}
	else {
		//std::cout << "FEED FORWARD 2\n";
		forward_tanh_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_temp11,model->d_b_c,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("c_prime_t_tanh");
	cudaEventRecord(model->ih_layer_info.c_prime_t_tanh_full,model->ih_layer_info.s6);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(c_prime_t_tanh,d_c_prime_t_tanh,"c_prime_t_tanh in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();
	//OPERATION
	//USING STREAMS 7 and 8
	//o_t = ((model->M_o*temp_mat + model->W_ho*h_t_prev).colwise() + model->b_o).unaryExpr(sigmoid_functor());
	alpha = 1;
	beta = 0;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
	cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_o,LSTM_size,
		d_sparse_lookup,LSTM_size,&beta,model->d_temp7,LSTM_size),"Forward prop o_t temp1 failed\n");
	cudaEventRecord(model->ih_layer_info.o_t_part1,model->ih_layer_info.s7);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_ho,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp8,LSTM_size),"Forward prop o_t temp2 failed\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	if(feed_input && index!=0) {
		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif
		//std::cout << "FEED FORWARD 1\n";
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
		cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.sparse_forward_start,0);
		cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.attention_forward,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_Q_o,LSTM_size,
			d_h_tild,LSTM_size,&beta,model->d_temp12,LSTM_size),"Forward prop i_t temp2 failed\n");
	}

	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.o_t_part1,0);
	if(!feed_input || index==0) {
		forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_b_o,LSTM_size);
	}
	else {
		//std::cout << "FEED FORWARD 2\n";
		forward_sigmoid_kernel_feed<<<kernel,threads_per_block,0,model->ih_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_temp12,model->d_b_o,LSTM_size);
	}
	CUDA_GET_LAST_ERROR("o_t");
	cudaEventRecord(model->ih_layer_info.o_t_full,model->ih_layer_info.s8);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//FOR NOW THE REST ARE USING THE DEFAULT STREAM
	//c_t = ((f_t.array())*(c_t_prev.array())).matrix() + (i_t.array()*(c_prime_t_tanh.array())).matrix();
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.i_t_full,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.f_t_full,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.c_prime_t_tanh_full,0);
	//cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.o_t_full,0);
	forward_c_t_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_c_t,d_f_t,d_c_t_prev,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("c_t");
	//cudaDeviceSynchronize();

	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->ih_layer_info.s0>>>(d_c_t,BZ_CUDA::cell_clip_threshold,LSTM_size*minibatch_size);
	}

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.o_t_full,0);
	forward_h_t_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_h_t,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("h_t");


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	//OPERATION
	// for(int i=0; i< vocab_indices_input.rows(); i++) {
	// 	if(vocab_indices_input(i)==-1) {
	// 		h_t.col(i).setZero();
	// 		c_t.col(i).setZero();
	// 	}
	// }
	//cudaDeviceSynchronize();
	// std::cout << "MASK FOR LSTM\n";
	// devSynchAll();
	// print_GPU_Matrix(d_input_vocab_indices_01,1,minibatch_size);

	zero_c_t_and_h_t<<< kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_h_t,d_c_t,d_input_vocab_indices_01,LSTM_size);
	CUDA_GET_LAST_ERROR("zero");
	//cudaDeviceSynchronize();


	// devSynchAll();
	// get_cell_states(d_c_t,LSTM_size,minibatch_size);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	send_h_t_above();

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(0);
	//cudaDeviceSynchronize();
	
}

template<typename dType>
void LSTM_IH_Node<dType>::send_h_t_above() {


	if(model->model->decode) {
		index = 0;
	}

	//run forward prop for attention model
	if(attention_model) {
		cudaEventRecord(model->attent_layer->layer_info.start_forward,model->ih_layer_info.s0);
		model->attent_layer->nodes[index].forward_prop();
		cudaStreamWaitEvent(model->ih_layer_info.s0,model->attent_layer->layer_info.forward_prop_done,0);

		//run the second attention model
		if(multi_attention) {
			cudaEventRecord(model->attent_layer_bi->layer_info.start_forward,model->ih_layer_info.s0);
			model->attent_layer_bi->nodes[index].forward_prop();
			cudaStreamWaitEvent(model->ih_layer_info.s0,model->attent_layer_bi->layer_info.forward_prop_done,0);
		}

		//now run it through the combiner layer
		if(multi_attention) {
			cudaEventRecord(model->att_comb_layer->start_forward,model->ih_layer_info.s0);
			model->att_comb_layer->nodes[index]->forward();
			cudaStreamWaitEvent(model->ih_layer_info.s0,model->att_comb_layer->forward_prop_done,0);
		}
	}

	//send the finished h_t to the above layer
	//the multigpu synchronization structure
	if(model->upper_layer.copy_h_t) {
		//transfer h_t to the layer above
		if(model->upper_layer.upper_softmax) {
			if(!model->upper_layer.source_side) {
				if(!attention_model) {
					//cudaMemcpyAsync(model->upper_layer.softmax->nodes[index].d_h_t, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
					cudaMemcpyAsync(model->upper_layer.softmax->get_ht_ptr(index), d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
				}
				else {
					if(multi_attention) {
						//std::cout << "HERE TRANS 1\n";
						cudaMemcpyAsync(model->upper_layer.softmax->get_ht_ptr(index), model->att_comb_layer->nodes[index]->d_ht_final,LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
					}
					else {
						cudaMemcpyAsync(model->upper_layer.softmax->get_ht_ptr(index), model->attent_layer->nodes[index].d_final_temp_2,LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
					}
				}
			}

			//the above check wont send anything in the bidirectional case, do this here
			if(model->bi_dir) {
				cudaMemcpyAsync(d_bi_dir_ht, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
			}
		}
		else {
			if(!attention_model) {
				cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
			}
			else {
				if(multi_attention) {
					cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, model->att_comb_layer->nodes[index]->d_ht_final, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
				}
				else {
					cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, model->attent_layer->nodes[index].d_final_temp_2, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s0);
				}
			}
		}
	}
	else {
		if(model->upper_layer.upper_softmax) {
			//upper layer is softmax
			if(!model->upper_layer.source_side) {
				if(!attention_model) {
					//model->upper_layer.softmax->nodes[index].d_h_t = d_h_t;
					model->upper_layer.softmax->set_ht_ptr(index,d_h_t);
				}
				else {
					model->upper_layer.softmax->set_ht_ptr(index,model->attent_layer->nodes[index].d_final_temp_2);
					//model->upper_layer.softmax->nodes[index].d_h_t = model->attent_layer->nodes[index].d_final_temp_2;
				}
			}
		}
		else {
			//upper layer is hidden layer
			if(!attention_model) {
				model->upper_layer.hidden_layer->nodes[index].d_h_t_below = d_h_t;
			}
			else {
				model->upper_layer.hidden_layer->nodes[index].d_h_t_below = model->attent_layer->nodes[index].d_final_temp_2;
			}
		}
	}

	cudaEventRecord(model->ih_layer_info.h_t_below_transfer,model->ih_layer_info.s0);

	//if feed input have the h_t go back to the lowest layer input
}


template<typename dType>
void LSTM_IH_Node<dType>::backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct)//,dType *d_d_ERRt_ht) 
{
	this->d_d_ERRnTOtp1_ht = d_d_ERRnTOtp1_ht;
	this->d_d_ERRnTOtp1_ct = d_d_ERRnTOtp1_ct;
	//this->d_d_ERRt_ht = d_d_ERRt_ht;
}

//this is to be called if feed input is true
template<typename dType>
void LSTM_IH_Node<dType>::attention_extra() {
	cudaSetDevice(model->ih_layer_info.device_number);
	dType *h_temp;
	full_matrix_setup(&h_temp,&d_ERRnTOt_h_tild,LSTM_size,minibatch_size);
	full_matrix_setup(&h_temp,&d_h_tild,LSTM_size,minibatch_size);
	feed_input = true;
}

template<typename dType>
void LSTM_IH_Node<dType>::back_prop_GPU(int index) {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	cudaSetDevice(model->ih_layer_info.device_number);

	//std::cout << "back prop node starting\n";
	if(model->upper_layer.upper_softmax) {
		//cudaStreamWaitEvent(model->ih_layer_info.s0,model->upper_layer.softmax->s_layer_info.d_ERR_ht_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s0,model->upper_layer.softmax->get_ERR_ht_event(),0);
	}
	else {
		cudaStreamWaitEvent(model->ih_layer_info.s0,model->upper_layer.hidden_layer->hh_layer_info.d_ERR_ht_done,0);
	}
	//cudaStreamWaitEvent(model->ih_layer_info.s0,model->model->s_layer_info.d_ERR_ht_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.htm1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.ctm1_done,0);

	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_grad_full_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_hi_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_hf_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_ho_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.W_hc_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_i_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_f_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_o_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.M_c_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_i_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_f_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_o_grad_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.b_c_grad_done,0);



	//now pass the error to the attention node if the attention model is in place
	//deal with the feed input here
	if(attention_model) {

		//run back_prop for combiner
		if(multi_attention) {
			cudaEventRecord(model->att_comb_layer->start_backward,model->ih_layer_info.s0);
			model->att_comb_layer->nodes[index]->backward();
			cudaStreamWaitEvent(model->ih_layer_info.s0,model->att_comb_layer->backward_prop_done,0);
		}

		//run backprop for each attention layer
		if(multi_attention) {
			cudaEventRecord(model->attent_layer_bi->layer_info.start_backward,model->ih_layer_info.s0);
			model->attent_layer_bi->nodes[index].back_prop();
			cudaStreamWaitEvent(model->ih_layer_info.s0,model->attent_layer_bi->layer_info.backward_prop_done,0);
		}

		cudaEventRecord(model->attent_layer->layer_info.start_backward,model->ih_layer_info.s0);
		model->attent_layer->nodes[index].back_prop();
		cudaStreamWaitEvent(model->ih_layer_info.s0,model->attent_layer->layer_info.backward_prop_done,0);

		//combine the errors before being use in the LSTM below
		if(multi_attention) {
			add_two_mats_into_third_kernel<<<256,256,0,model->ih_layer_info.s0>>>(model->att_comb_layer->nodes[index]->d_ERR_ht_top_loss,model->attent_layer->nodes[index].d_d_ERRt_ht_tild,model->attent_layer_bi->nodes[index].d_d_ERRt_ht_tild,LSTM_size*minibatch_size); 
		}
	}


	// std::cout << "Printing error with respect to h_t in LSTM node:\n";
	// devSynchAll();
	// print_GPU_Matrix(d_d_ERRt_ht,LSTM_size,minibatch_size);


	// if(model->upper_layer.upper_softmax && model->upper_layer.source_side) {
	// 	d_d_ERRt_ht = d_zeros;
	// }

	//USING STREAM ZERO
	//OPERATION
	//d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;
	dType alpha = 1;
	dType beta = 1;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ht,LSTM_size,
		&beta,d_d_ERRt_ht,LSTM_size,model->d_d_ERRnTOt_ht,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	if(model->nonrev_bi_dir) {
		for(int i=0; i<minibatch_size; i++) {
			if(model->model->bi_dir_source.final_index_hs[i]==index) {
				add_to_errors<<<256,256,0,model->ih_layer_info.s0>>>(model->d_d_ERRnTOt_ht,model->model->bi_dir_source.d_hs_nonrev_error_horiz[0],LSTM_size,i);
			}
		}
	}

	//OPERATION
	//d_ERRt_ct.transpose() = d_ERRnTOt_ht.transpose().array() * (o_t.array()*(1-(c_t).array().unaryExpr(tanh_sq_functor())));
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	d_ERRt_ct_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(model->d_d_ERRt_ct,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP c_t");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ct,LSTM_size,
		&beta,model->d_d_ERRt_ct,LSTM_size,model->d_d_ERRnTOt_ct,LSTM_size),"backprop addition failed, d_ERRnTOt_ct \n");


	//now potentially add in errors if using the bi-directional model
	if(model->nonrev_bi_dir) {
		for(int i=0; i<minibatch_size; i++) {
			if(model->model->bi_dir_source.final_index_hs[i]==index) {
				add_to_errors<<<256,256,0,model->ih_layer_info.s0>>>(model->d_d_ERRnTOt_ct,model->model->bi_dir_source.d_ct_nonrev_error_horiz[0],LSTM_size,i);
			}
		}
	}
	

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	//OPERATION
	//zero out columns of d_ERRnTOt_ht and d_ERRnTOt_ct
	zero_columns_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ht,d_input_vocab_indices_01,model->d_d_ERRnTOt_ht);
	CUDA_GET_LAST_ERROR("BP h_tn");
	zero_columns_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ct,d_input_vocab_indices_01,model->d_d_ERRnTOt_ct);
	CUDA_GET_LAST_ERROR("BP c_tn");

	//EVENT FOR FINISHING THE FIRST STUFF
	cudaEventRecord(model->ih_layer_info.backprop_init,model->ih_layer_info.s0);

	//STARTING FROM THIS POINT STREAMS WILL BE USED
	//OPERATION
	//USING STREAM 1
	//d_ERRnTOt_ot.transpose() = d_ERRnTOt_ht.transpose().array()*( c_t.array().unaryExpr(tanh_functor()) )*o_t*(1-o_t);
	cudaStreamWaitEvent(model->ih_layer_info.s1,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ot_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s1>>>(model->d_d_ERRnTOt_ot,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP o_tn");
	cudaEventRecord(model->ih_layer_info.err_ot_done,model->ih_layer_info.s1);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 2
	//d_ERRnTOt_ft.transpose() = d_ERRnTOt_ct.transpose().array()*(c_t_prev.array())*f_t*(1-f_t);
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(model->d_d_ERRnTOt_ft,model->d_d_ERRnTOt_ct,d_c_t_prev,d_f_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");
	cudaEventRecord(model->ih_layer_info.err_ft_done,model->ih_layer_info.s2);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 3
	//d_ERRnTOt_tanhcpt.transpose() = d_ERRnTOt_ct.transpose().array()*(i_t.array());
	cudaStreamWaitEvent(model->ih_layer_info.s3,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_tanhcpt_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s3>>>(model->d_d_ERRnTOt_tanhcpt,model->d_d_ERRnTOt_ct,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("BP tanh_tn");
	cudaEventRecord(model->ih_layer_info.err_tanhcpt_done,model->ih_layer_info.s3);
		
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 4
	//d_ERRnTOt_it.transpose() = d_ERRnTOt_ct.transpose().array()*(c_prime_t_tanh.array());
	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(model->d_d_ERRnTOt_it,model->d_d_ERRnTOt_ct,d_c_prime_t_tanh,d_i_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP it_tn");
	cudaEventRecord(model->ih_layer_info.err_it_done,model->ih_layer_info.s4);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	//OPERATION
	//USING STREAM 5,6,7,8,9
	// d_ERRnTOt_htM1.transpose() = (W_ho.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() )) \
	// + (W_hf.transpose()*((d_ERRnTOt_ft.transpose().array() * f_t.array() *(1-f_t.array())).matrix())) \
	// + (W_hi.transpose()*((d_ERRnTOt_it.transpose().array()*i_t.array()*(1-i_t.array())).matrix())) \
	// + (W_hc.transpose()*((d_ERRnTOt_tanhcpt.transpose().array()*(1-c_prime_t_tanh.array().square())).matrix()));
	dType alpha2 = 1;
	dType beta2 = 0;

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s5);
	cudaStreamWaitEvent(model->ih_layer_info.s5,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_ho,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p1_done,model->ih_layer_info.s5);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hf,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p2_done,model->ih_layer_info.s6);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
	cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hi,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p3_done,model->ih_layer_info.s7);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hc,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p4_done,model->ih_layer_info.s8);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif



	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p4_done,0);
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->ih_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1");


	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->ih_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}

	cudaEventRecord(model->ih_layer_info.htm1_done_temp,model->ih_layer_info.s9);


	// std::cout << "devSynchAll LSTM 1st layer backprop check 1\n";
	// devSynchAll();


	//OPERATION
	//send error to the attention model
	if(feed_input && index!=0) {

		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s5);
		cudaStreamWaitEvent(model->ih_layer_info.s5,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p1_done,model->ih_layer_info.s5);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
		cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p2_done,model->ih_layer_info.s6);


		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
		cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p3_done,model->ih_layer_info.s7);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
		cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.htm1_done_temp,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_Q_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p4_done,model->ih_layer_info.s8);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif


		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p1_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p2_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p3_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p4_done,0);
		add_four_matrices_kernel<<< kernel,threads_per_block,0,model->ih_layer_info.s9>>>(d_ERRnTOt_h_tild,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
		CUDA_GET_LAST_ERROR("BP h tilda");


		if(BZ_CUDA::clip_cell) {
			clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->ih_layer_info.s9>>>(d_ERRnTOt_h_tild,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
		}

		cudaMemcpyAsync(d_ERRnTOt_h_tild_cpy,d_ERRnTOt_h_tild,LSTM_size*minibatch_size*sizeof(dType),cudaMemcpyDefault,model->ih_layer_info.s9);
		cudaEventRecord(model->ih_layer_info.error_htild_below,model->ih_layer_info.s9);
		
		// devSynchAll();
		// std::cout << "PRINTING ERROR OF HTILD FROM LOWER LSTM: \n";
		// print_GPU_Matrix(d_ERRnTOt_h_tild,LSTM_size,minibatch_size);
	}

	//dont record this event until after the feed input
	cudaEventRecord(model->ih_layer_info.htm1_done,model->ih_layer_info.s9);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 10
	//d_ERRnTOt_ctM1.transpose() = (d_ERRnTOt_ct.transpose().array()*f_t.array());
	cudaStreamWaitEvent(model->ih_layer_info.s10,model->ih_layer_info.backprop_init,0);
	elementwise_mult_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s10>>>(model->d_d_ERRnTOt_ct,d_f_t,model->d_d_ERRnTOt_ctM1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");


	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->ih_layer_info.s10>>>(model->d_d_ERRnTOt_ct,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}
	cudaEventRecord(model->ih_layer_info.ctm1_done,model->ih_layer_info.s10);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//eigen_check_thrust_ptr(d_ERRnTOt_ctM1.transpose(),d_d_ERRnTOt_ctM1,"d_ERRnTOt_ctM1 in back prop",(dType)0.0001);

	compute_gradients_GPU();

}


template<typename dType>
void LSTM_IH_Node<dType>::compute_gradients_GPU() {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAMS 11,12,13,14
	//model->W_hi_grad.noalias() += (h_t_prev*(d_ERRnTOt_it.array() * i_t.transpose().array()*(1-i_t.transpose().array())).matrix()).transpose();
	//model->W_hf_grad.noalias() += (h_t_prev*(d_ERRnTOt_ft.array()*f_t.transpose().array()*(1-f_t.transpose().array())).matrix()).transpose();
	//model->W_hc_grad.noalias() += (h_t_prev*(d_ERRnTOt_ct.array()*(i_t.transpose().array())*(1-c_prime_t_tanh.transpose().array().square())).matrix()).transpose();
	//model->W_ho_grad.noalias() += (h_t_prev*(d_ERRnTOt_ot.array()*o_t.transpose().array()*(1-o_t.transpose().array())).matrix()).transpose();
	dType alpha = 1;
	dType beta = 1;

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s11);
	cudaStreamWaitEvent(model->ih_layer_info.s11,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hi_grad,LSTM_size),"Backprop W_hi grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hi_grad_done,model->ih_layer_info.s11);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s12);
	cudaStreamWaitEvent(model->ih_layer_info.s12,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hf_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hf_grad_done,model->ih_layer_info.s12);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s13);
	cudaStreamWaitEvent(model->ih_layer_info.s13,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hc_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hc_grad_done,model->ih_layer_info.s13);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s14);
	cudaStreamWaitEvent(model->ih_layer_info.s14,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_ho_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_ho_grad_done,model->ih_layer_info.s14);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	//OPERATION
	//USING STREAMS 15,16,17,18
	//compute_temp_mat(model->W);
	//model->M_i_grad.noalias() += (d_ERRnTOt_it.transpose().array() * i_t.array() * (1-i_t.array())).matrix() * temp_mat.transpose();
	//model->M_f_grad.noalias() += (d_ERRnTOt_ft.transpose().array() * f_t.array() * (1-f_t.array())).matrix() * temp_mat.transpose();
	//model->M_o_grad.noalias() += (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1-o_t.array())).matrix() * temp_mat.transpose();
	//model->M_c_grad.noalias() += (d_ERRnTOt_tanhcpt.transpose().array() * (1-c_prime_t_tanh.array().square())).matrix() * temp_mat.transpose();
	alpha = 1;
	beta = 1;

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s15);
	cudaStreamWaitEvent(model->ih_layer_info.s15,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_i_grad,LSTM_size),"Backprop M_i grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_i_grad_done,model->ih_layer_info.s15);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s16);
	cudaStreamWaitEvent(model->ih_layer_info.s16,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_f_grad,LSTM_size),"Backprop M_f grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_f_grad_done,model->ih_layer_info.s16);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s17);
	cudaStreamWaitEvent(model->ih_layer_info.s17,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_o_grad,LSTM_size),"Backprop M_o grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_o_grad_done,model->ih_layer_info.s17);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s18);
	cudaStreamWaitEvent(model->ih_layer_info.s18,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_c_grad,LSTM_size),"Backprop M_c grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_c_grad_done,model->ih_layer_info.s18);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//now do all of the Q's
	//resuse all events from M_i
	if(feed_input && index!=0) {
		alpha = 1;
		beta = 1;
		//std::cout << "HERE in feed_input backprop\n";
		#ifdef REMOVE_STREAMS_FEED_INPUT
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s15);
		cudaStreamWaitEvent(model->ih_layer_info.s15,model->ih_layer_info.err_it_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_it,LSTM_size,d_h_tild,LSTM_size,&beta,model->d_Q_i_grad,LSTM_size),"Backprop Q_i grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_i_grad_done,model->ih_layer_info.s15);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s16);
		cudaStreamWaitEvent(model->ih_layer_info.s16,model->ih_layer_info.err_ft_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_ft,LSTM_size,d_h_tild,LSTM_size,&beta,model->d_Q_f_grad,LSTM_size),"Backprop Q_f grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_f_grad_done,model->ih_layer_info.s16);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s17);
		cudaStreamWaitEvent(model->ih_layer_info.s17,model->ih_layer_info.err_ot_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_ot,LSTM_size,d_h_tild,LSTM_size,&beta,model->d_Q_o_grad,LSTM_size),"Backprop Q_o grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_o_grad_done,model->ih_layer_info.s17);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s18);
		cudaStreamWaitEvent(model->ih_layer_info.s18,model->ih_layer_info.err_tanhcpt_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
			model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_tild,LSTM_size,&beta,model->d_Q_c_grad,LSTM_size),"Backprop Q_c grad cublas gemm failed\n");
		cudaEventRecord(model->ih_layer_info.M_c_grad_done,model->ih_layer_info.s18);
	}

	//OPERATION
	//USING STREAMS 19,20,21,22
	//b_i_grad.noalias() += ((d_ERRnTOt_it.array() * (i_t.array() * (1-i_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	//b_f_grad.noalias() += ((d_ERRnTOt_ft.array() * (f_t.array() * (1-f_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	//b_c_grad.noalias() += (d_ERRnTOt_tanhcpt.array() * (1-c_prime_t_tanh.array().square()).matrix().transpose().array()).colwise().sum().matrix().transpose();
	//b_o_grad.noalias() += ((d_ERRnTOt_ot.array() * (o_t.array() * (1-o_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s19);
	cudaStreamWaitEvent(model->ih_layer_info.s19,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_it,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_i_grad,1),"backprop b_i_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_i_grad_done,model->ih_layer_info.s19);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s20);
	cudaStreamWaitEvent(model->ih_layer_info.s20,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ft,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_f_grad,1),"backprop b_f_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_f_grad_done,model->ih_layer_info.s20);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s21);
	cudaStreamWaitEvent(model->ih_layer_info.s21,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ot,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_o_grad,1),"backprop b_o_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_o_grad_done,model->ih_layer_info.s21);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s22);
	cudaStreamWaitEvent(model->ih_layer_info.s22,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_tanhcpt,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_c_grad,1),"backprop b_c_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_c_grad_done,model->ih_layer_info.s22);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAMS 23,24,25,26
	// Z_i = d_ERRnTOt_it.array()*(i_t.array() * (1-i_t.array())).matrix().transpose().array();
	// Z_f = d_ERRnTOt_ft.array()*(f_t.array() * (1-f_t.array())).matrix().transpose().array();
	// Z_o = d_ERRnTOt_ot.array()*(o_t.array() * (1-o_t.array())).matrix().transpose().array();
	// Z_c = d_ERRnTOt_tanhcpt.array()*(1-c_prime_t_tanh.array().square()).matrix().transpose().array();

	// for(int i=0; i<vocab_indices_input.rows(); i++) {
	// 	if(vocab_indices_input(i)!=-1) {
	// 		for(int j=0; j<model->W_grad.rows(); j++) {
	// 			double sumtemp = Z_i.row(i) * model->M_i.col(j);
	// 			sumtemp += Z_f.row(i) * model->M_f.col(j);
	// 			sumtemp += Z_o.row(i) * model->M_o.col(j);
	// 			sumtemp += Z_c.row(i) * model->M_c.col(j);
	// 			model->W_grad(j,vocab_indices_input(i)) += sumtemp;
	// 		}
	// 	}
	// }

	if(!model->char_cnn) {
		alpha = 1;
		beta = 0;
		//cudaStreamWaitEvent(model->ih_layer_info.s23,model->ih_layer_info.W_grad_full_done,0);
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s23);
		cudaStreamWaitEvent(model->ih_layer_info.s23,model->ih_layer_info.err_it_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
			LSTM_size,&alpha,model->d_M_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta,
			model->d_temp5,LSTM_size),"cublas W gradient failed temp5\n");
		cudaEventRecord(model->ih_layer_info.W_grad_p1_done,model->ih_layer_info.s23);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		//cudaStreamWaitEvent(model->ih_layer_info.s24,model->ih_layer_info.W_grad_full_done,0);
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s24);
		cudaStreamWaitEvent(model->ih_layer_info.s24,model->ih_layer_info.err_ft_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
			LSTM_size,&alpha,model->d_M_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta,
			model->d_temp6,LSTM_size),"cublas W gradient failed temp6\n");
		cudaEventRecord(model->ih_layer_info.W_grad_p2_done,model->ih_layer_info.s24);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		//cudaStreamWaitEvent(model->ih_layer_info.s25,model->ih_layer_info.W_grad_full_done,0);
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s25);
		cudaStreamWaitEvent(model->ih_layer_info.s25,model->ih_layer_info.err_ot_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
			LSTM_size,&alpha,model->d_M_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta,
			model->d_temp7,LSTM_size),"cublas W gradient failed temp7\n");
		cudaEventRecord(model->ih_layer_info.W_grad_p3_done,model->ih_layer_info.s25);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		//cudaStreamWaitEvent(model->ih_layer_info.s26,model->ih_layer_info.W_grad_full_done,0);
		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s26);
		cudaStreamWaitEvent(model->ih_layer_info.s26,model->ih_layer_info.err_tanhcpt_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
			LSTM_size,&alpha,model->d_M_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta,
			model->d_temp8,LSTM_size),"cublas W gradient failed temp8\n");
		cudaEventRecord(model->ih_layer_info.W_grad_p4_done,model->ih_layer_info.s26);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		//cudaDeviceSynchronize();
		//cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_full_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p1_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p2_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p3_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p4_done,0);
		int threads_per_block = 128;
		int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
		dim3 kernel(minibatch_size,num_block,1);

		if(!dropout) {
			// W_gradient_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s27>>>(model->d_W_grad,d_input_vocab_indices,model->d_temp5,
			// 	model->d_temp6,model->d_temp7,model->d_temp8,LSTM_size);

			W_small_gradient_kernel<<<256,256,0,model->ih_layer_info.s27>>>(model->d_small_W_grad,model->d_reverse_unique_indicies,model->d_temp5,
				model->d_temp6,model->d_temp7,model->d_temp8,d_input_vocab_indices,LSTM_size,minibatch_size);
		}
		else {
			// W_gradient_kernel_dropout<<<kernel,threads_per_block,0,model->ih_layer_info.s27>>>(model->d_W_grad,d_input_vocab_indices,model->d_temp5,
			// 	model->d_temp6,model->d_temp7,model->d_temp8,LSTM_size,d_dropout_mask,dropout_rate);

			W_small_dropout_gradient_kernel<<<256,256,0,model->ih_layer_info.s27>>>(model->d_small_W_grad,model->d_reverse_unique_indicies,model->d_temp5,
				model->d_temp6,model->d_temp7,model->d_temp8,d_input_vocab_indices,LSTM_size,minibatch_size,d_dropout_mask,dropout_rate);
		}
		CUDA_GET_LAST_ERROR("BP w_grad");
	}
	else {
		dType alpha2 = 1;
		dType beta2 = 0;

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s23);
		cudaStreamWaitEvent(model->ih_layer_info.s23,model->ih_layer_info.err_ot_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_M_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp5,LSTM_size),"Error backprop temp1 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p1_done,model->ih_layer_info.s3);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s24);
		cudaStreamWaitEvent(model->ih_layer_info.s24,model->ih_layer_info.err_ft_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_M_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp6,LSTM_size),"Error backprop temp2 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p2_done,model->ih_layer_info.s24);


		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s25);
		cudaStreamWaitEvent(model->ih_layer_info.s25,model->ih_layer_info.err_it_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_M_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp7,LSTM_size),"Error backprop temp3 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p3_done,model->ih_layer_info.s25);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s26);
		cudaStreamWaitEvent(model->ih_layer_info.s26,model->ih_layer_info.err_tanhcpt_done,0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
			&alpha2,model->d_M_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp8,LSTM_size),"Error backprop temp4 htM1\n");
		cudaEventRecord(model->ih_layer_info.htm1_p4_done,model->ih_layer_info.s26);

		#ifdef REMOVE_STREAMS
		devSynchAll();
		#endif

		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.htm1_p1_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.htm1_p2_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.htm1_p3_done,0);
		cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.htm1_p4_done,0);
		add_four_matrices_kernel_stride<<< 256,256,0,model->ih_layer_info.s27>>>(model->d_conv_char_error,model->d_temp5,model->d_temp6,model->d_temp7,model->d_temp8,LSTM_size*minibatch_size);
		CUDA_GET_LAST_ERROR("BP htm1");

		if(dropout) {
			dropout_kernel<<<256,256,0,model->ih_layer_info.s27>>>(d_dropout_mask,model->dropout_rate,model->d_conv_char_error,LSTM_size*minibatch_size);
		}

		cudaMemcpyAsync(model->char_cnn_layer->top_highway_layer->nodes[index]->d_Err_z, model->d_conv_char_error, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->ih_layer_info.s27);
		cudaEventRecord(model->char_cnn_layer->back_prop_start,model->ih_layer_info.s27);
		model->char_cnn_layer->backward(index);

	}
	cudaEventRecord(model->ih_layer_info.W_grad_full_done,model->ih_layer_info.s27);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	
	//cudaDeviceSynchronize();
}




template<typename dType>
void LSTM_IH_Node<dType>::dump_LSTM(std::ofstream &LSTM_dump_stream,std::string intro) {
	//cudaSetDevice(model->ih_layer_info.device_number);
	devSynchAll();
	LSTM_dump_stream << intro;
	int vocab_index;
	cudaMemcpy(&vocab_index,d_input_vocab_indices,1*sizeof(int),cudaMemcpyDeviceToHost);
	LSTM_dump_stream << "Vocab index:"<<vocab_index << "\n";

	//forget gate
	thrust::device_ptr<dType> output_ptr = thrust::device_pointer_cast(d_f_t);
	LSTM_dump_stream << "Forget gate:";
	for(int i=0; i<LSTM_size; i++) {
		LSTM_dump_stream << output_ptr[i];
		if(i!= LSTM_size-1) {
		 	LSTM_dump_stream << " ";
		}
	}
	LSTM_dump_stream << "\n";

	//input gate
	LSTM_dump_stream << "Input gate:";
	output_ptr = thrust::device_pointer_cast(d_i_t);
	for(int i=0; i<LSTM_size; i++) {
		LSTM_dump_stream << output_ptr[i];
		if(i!= LSTM_size-1) {
		 	LSTM_dump_stream << " ";
		}
	}
	LSTM_dump_stream << "\n";

	//c_t
	LSTM_dump_stream << "c_t:";
	output_ptr = thrust::device_pointer_cast(d_c_t);
	for(int i=0; i<LSTM_size; i++) {
		LSTM_dump_stream << output_ptr[i];
		if(i!= LSTM_size-1) {
		 	LSTM_dump_stream << " ";
		}
	}
	LSTM_dump_stream << "\n";


	//output gate
	LSTM_dump_stream << "Output gate:";
	output_ptr = thrust::device_pointer_cast(d_o_t);
	for(int i=0; i<LSTM_size; i++) {
		LSTM_dump_stream << output_ptr[i];
		if(i!= LSTM_size-1) {
		 	LSTM_dump_stream << " ";
		}
	}
	LSTM_dump_stream << "\n";


	//h_t
	LSTM_dump_stream << "h_t:";
	output_ptr = thrust::device_pointer_cast(d_h_t);
	for(int i=0; i<LSTM_size; i++) {
		LSTM_dump_stream << output_ptr[i];
		if(i!= LSTM_size-1) {
		 	LSTM_dump_stream << " ";
		}
	}
	LSTM_dump_stream << "\n";

}





