
//Constructor
template<typename dType>
LSTM_HH_Node<dType>::LSTM_HH_Node(int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m,int index,
	dType *d_zeros,bool dropout, dType dropout_rate) {

	model = m;
	this->dropout = dropout;
	this->dropout_rate = dropout_rate;

	init_LSTM_GPU(LSTM_size,minibatch_size,m);

	//model = m;
	this->minibatch_size = minibatch_size;
	this->LSTM_size = LSTM_size;
	this->index = index;
	this->d_zeros = d_zeros;
}

template<typename dType>
void LSTM_HH_Node<dType>::init_LSTM_GPU(int LSTM_size,int minibatch_size,struct Hidden_To_Hidden_Layer<dType> *m) {

	cudaSetDevice(model->hh_layer_info.device_number);

	full_matrix_setup(&h_o_t,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_t,&d_c_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_f_t,&d_f_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_prime_t_tanh,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_i_t,&d_i_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t,&d_h_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t_below,&d_h_t_below,LSTM_size,minibatch_size);
	full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,LSTM_size,minibatch_size);
	cudaMemset(d_d_ERRt_ht,0,LSTM_size*minibatch_size*sizeof(dType));

	if(dropout) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dropout_mask, LSTM_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	}

}


//Update the hidden state and cell state vectors
// template<typename dType>
// template<typename Derived,typename Derived2>
// void LSTM_HH_Node<dType>::update_vectors_forward(const Eigen::MatrixBase<Derived> &h_prev,
// 	const Eigen::MatrixBase<Derived> &c_prev,
// 	const Eigen::MatrixBase<Derived2> &vocab,
// 	int index,int *d_input_vocab_indices_01,
// 	dType *d_h_t_prev,dType *d_c_t_prev) 
// {
	
// 	update_vectors_forward_GPU(d_input_vocab_indices_01,d_h_t_prev,d_c_t_prev);

// }


template<typename dType>
void LSTM_HH_Node<dType>::update_vectors_forward_GPU(int *d_input_vocab_indices_01,
	dType *d_h_t_prev,dType *d_c_t_prev) 
{
	this->d_h_t_prev = d_h_t_prev;
	this->d_c_t_prev = d_c_t_prev;
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
}


//Update the hidden state and cell state vectors for first column in target model
template<typename dType>
void LSTM_HH_Node<dType>::update_vectors_forward_decoder(int *d_input_vocab_indices_01) 
{
	//GPU stuff
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
}




template<typename dType>
void LSTM_HH_Node<dType>::forward_prop() {
	
	forward_prop_GPU();
}


template<typename dType>
void LSTM_HH_Node<dType>::forward_prop_sync(cudaStream_t &my_s) {

	if(model->lower_layer.copy_d_Err_ht) {
		if(model->lower_layer.lower_input) {
			cudaStreamWaitEvent(my_s,model->lower_layer.input_layer->ih_layer_info.h_t_below_transfer,0);
		}
		else {
			cudaStreamWaitEvent(my_s,model->lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);
		}
	}
	cudaStreamWaitEvent(my_s,model->hh_layer_info.h_t_below_transfer,0);
	cudaStreamWaitEvent(my_s,model->hh_layer_info.dropout_done,0);
}

template<typename dType>
void LSTM_HH_Node<dType>::forward_prop_GPU() {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	
	cudaSetDevice(model->hh_layer_info.device_number);

	if(dropout && model->model->train) {
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.h_t_below_transfer,0);
		if(model->lower_layer.copy_d_Err_ht && model->lower_layer.lower_input) {
			cudaStreamWaitEvent(model->hh_layer_info.s0,model->lower_layer.input_layer->ih_layer_info.h_t_below_transfer,0);
		}
		else {
			cudaStreamWaitEvent(model->hh_layer_info.s0,model->lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);
		}

		if(!model->model->grad_check_flag) {
			curandSetStream(model->rand_gen,model->hh_layer_info.s0);
			curandGenerateUniform_wrapper(d_dropout_mask,LSTM_size*minibatch_size,model->rand_gen); 
		}
		dropout_kernel<<<256,256,0,model->hh_layer_info.s0>>>(d_dropout_mask,dropout_rate,d_h_t_below,LSTM_size*minibatch_size);
		cudaEventRecord(model->hh_layer_info.dropout_done,model->hh_layer_info.s0);
	}

	//OPERATION
	//USING STREAMS 1 and 2
	//i_t = ((model->M_i*h_t_below + model->W_hi*h_t_prev).colwise() + model->b_i).array().unaryExpr(sigmoid_functor());
	dType alpha =1;
	dType beta = 0;

	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	CUDA_GET_LAST_ERROR("CHECKPOINT 0");

	//std::cout << "i_t start\n";
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s1);
	forward_prop_sync(model->hh_layer_info.s1);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_i,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp1,LSTM_size),"Forward prop i_t temp1 failed\n");
	cudaEventRecord(model->hh_layer_info.i_t_part1,model->hh_layer_info.s1);


	CUDA_GET_LAST_ERROR("CHECKPOINT 1");
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s2);
	forward_prop_sync(model->hh_layer_info.s2);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hi,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp2,LSTM_size),"Forward prop i_t temp2 failed\n");


	CUDA_GET_LAST_ERROR("CHECKPOINT 2");
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(model->hh_layer_info.s2,model->hh_layer_info.i_t_part1,0);
	forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_b_i,LSTM_size);
	CUDA_GET_LAST_ERROR("i_t LSTM HH");
	cudaEventRecord(model->hh_layer_info.i_t_full,model->hh_layer_info.s2);
	//std::cout << "i_t end\n";
	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(i_t,d_i_t,"i_t in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();
	//OPERATION
	//f_t = ((model->M_f*temp_mat + model->W_hf*h_t_prev).colwise() + model->b_f).array().unaryExpr(sigmoid_functor());
	//USING STREAMS 3 and 4
	alpha =1;
	beta = 0;
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s3);
	forward_prop_sync(model->hh_layer_info.s3);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_f,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp3,LSTM_size),"Forward prop f_t temp3 failed\n");
	cudaEventRecord(model->hh_layer_info.f_t_part1,model->hh_layer_info.s3);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s4);
	forward_prop_sync(model->hh_layer_info.s4);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hf,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp4,LSTM_size),"Forward prop f_t temp4 failed\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(model->hh_layer_info.s4,model->hh_layer_info.f_t_part1,0);
	forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_b_f,LSTM_size);
	CUDA_GET_LAST_ERROR("f_t");
	cudaEventRecord(model->hh_layer_info.f_t_full,model->hh_layer_info.s4);

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
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s5);
	forward_prop_sync(model->hh_layer_info.s5);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_c,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp5,LSTM_size),"Forward prop c_prime_t_tanh temp5 failed\n");
	cudaEventRecord(model->hh_layer_info.c_prime_t_tanh_part1,model->hh_layer_info.s5);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s6);
	forward_prop_sync(model->hh_layer_info.s6);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hc,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp6,LSTM_size),"Forward prop c_prime_t_tanh temp6 failed\n");


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(model->hh_layer_info.s6,model->hh_layer_info.c_prime_t_tanh_part1,0);
	forward_tanh_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_b_c,LSTM_size);
	CUDA_GET_LAST_ERROR("c_prime_t_tanh");
	cudaEventRecord(model->hh_layer_info.c_prime_t_tanh_full,model->hh_layer_info.s6);


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
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s7);
	forward_prop_sync(model->hh_layer_info.s7);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_M_o,LSTM_size,
		d_h_t_below,LSTM_size,&beta,model->d_temp7,LSTM_size),"Forward prop o_t temp1 failed\n");
	cudaEventRecord(model->hh_layer_info.o_t_part1,model->hh_layer_info.s7);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s8);
	forward_prop_sync(model->hh_layer_info.s8);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_ho,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp8,LSTM_size),"Forward prop o_t temp2 failed\n");


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(model->hh_layer_info.s8,model->hh_layer_info.o_t_part1,0);
	forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_b_o,LSTM_size);
	CUDA_GET_LAST_ERROR("o_t");
	cudaEventRecord(model->hh_layer_info.o_t_full,model->hh_layer_info.s8);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//FOR NOW THE REST ARE USING THE DEFAULT STREAM
	//c_t = ((f_t.array())*(c_t_prev.array())).matrix() + (i_t.array()*(c_prime_t_tanh.array())).matrix();
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.i_t_full,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.f_t_full,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.c_prime_t_tanh_full,0);
	//cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.o_t_full,0);
	forward_c_t_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(d_c_t,d_f_t,d_c_t_prev,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("c_t");


	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->hh_layer_info.s0>>>(d_c_t,BZ_CUDA::cell_clip_threshold,LSTM_size*minibatch_size);
	}

	//cudaDeviceSynchronize();
	//OPERATION
	//h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.o_t_full,0);
	forward_h_t_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(d_h_t,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("h_t");

	//OPERATION
	// for(int i=0; i< vocab_indices_input.rows(); i++) {
	// 	if(vocab_indices_input(i)==-1) {
	// 		h_t.col(i).setZero();
	// 		c_t.col(i).setZero();
	// 	}
	// }
	//cudaDeviceSynchronize();
	zero_c_t_and_h_t<<< kernel,threads_per_block,0,model->hh_layer_info.s0>>>(d_h_t,d_c_t,d_input_vocab_indices_01,LSTM_size);
	CUDA_GET_LAST_ERROR("zero");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//this is for the attention model forward prop testing
	send_h_t_above();

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif


	//cudaDeviceSynchronize();
	cudaSetDevice(0);
}


template<typename dType>
void LSTM_HH_Node<dType>::send_h_t_above() {

	//run forward prop for attention model
	if(attention_model) {
		cudaEventRecord(model->attent_layer->layer_info.start_forward,model->hh_layer_info.s0);
		model->attent_layer->nodes[index].forward_prop();
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->attent_layer->layer_info.forward_prop_done,0);
	}

	//send the finished h_t to the above layer
	//the multigpu synchronization structure
	if(model->upper_layer.copy_h_t) {
		//transfer h_t to the layer above
		if(model->upper_layer.upper_softmax) {
			if(!model->upper_layer.source_side) {
				if(!attention_model) {
					cudaMemcpyAsync(model->upper_layer.softmax->get_ht_ptr(index), d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s0);
				}
				else {
					cudaMemcpyAsync(model->upper_layer.softmax->get_ht_ptr(index), model->attent_layer->nodes[index].d_final_temp_2,LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s0);
				}
			}
		}
		else {
			if(!attention_model) {
				cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, d_h_t, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s0);
			}
			else {
				cudaMemcpyAsync(model->upper_layer.hidden_layer->nodes[index].d_h_t_below, model->attent_layer->nodes[index].d_final_temp_2, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s0);
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
					//model->upper_layer.softmax->nodes[index].d_h_t = model->attent_layer->nodes[index].d_final_temp_2;
					model->upper_layer.softmax->set_ht_ptr(index,model->attent_layer->nodes[index].d_final_temp_2);
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

	cudaEventRecord(model->hh_layer_info.h_t_below_transfer,model->hh_layer_info.s0);
}

template<typename dType>
void LSTM_HH_Node<dType>::backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct) {
	this->d_d_ERRnTOtp1_ht = d_d_ERRnTOtp1_ht;
	this->d_d_ERRnTOtp1_ct = d_d_ERRnTOtp1_ct;
}

template<typename dType>
void LSTM_HH_Node<dType>::back_prop_GPU(int index) {

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaSetDevice(model->hh_layer_info.device_number);

	//std::cout << "back prop node starting\n";
	if(model->upper_layer.upper_softmax) {
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->upper_layer.softmax->get_ERR_ht_event(),0);
	}
	else {
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->upper_layer.hidden_layer->hh_layer_info.htm1_done,0);
	}

	//cudaStreamWaitEvent(model->hh_layer_info.s0,model->model->s_layer_info.d_ERR_ht_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_grad_full_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.htm1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.ctm1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_hi_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_hf_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_ho_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.W_hc_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_i_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_f_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_o_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.M_c_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_i_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_f_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_o_grad_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s0,model->hh_layer_info.b_c_grad_done,0);


	if(attention_model) {
		cudaEventRecord(model->attent_layer->layer_info.start_backward,model->hh_layer_info.s0);
		model->attent_layer->nodes[index].back_prop();
		cudaStreamWaitEvent(model->hh_layer_info.s0,model->attent_layer->layer_info.backward_prop_done,0);
	}


	// if(model->upper_layer.upper_softmax && model->upper_layer.source_side) {
	// 	d_d_ERRt_ht = d_zeros;
	// }

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//USING STREAM ZERO
	//OPERATION
	//d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;
	dType alpha = 1;
	dType beta = 1;
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ht,LSTM_size,
		&beta,d_d_ERRt_ht,LSTM_size,model->d_d_ERRnTOt_ht,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//d_ERRt_ct.transpose() = d_ERRnTOt_ht.transpose().array() * (o_t.array()*(1-(c_t).array().unaryExpr(tanh_sq_functor())));
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	d_ERRt_ct_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(model->d_d_ERRt_ct,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP c_t");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ct,LSTM_size,
		&beta,model->d_d_ERRt_ct,LSTM_size,model->d_d_ERRnTOt_ct,LSTM_size),"backprop addition failed, d_ERRnTOt_ct \n");

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//zero out columns of d_ERRnTOt_ht and d_ERRnTOt_ct
	zero_columns_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ht,d_input_vocab_indices_01,model->d_d_ERRnTOt_ht);
	CUDA_GET_LAST_ERROR("BP h_tn");
	zero_columns_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s0>>>(LSTM_size, model->d_d_ERRnTOt_ct,d_input_vocab_indices_01,model->d_d_ERRnTOt_ct);
	CUDA_GET_LAST_ERROR("BP c_tn");

	//EVENT FOR FINISHING THE FIRST STUFF
	cudaEventRecord(model->hh_layer_info.backprop_init,model->hh_layer_info.s0);

	//STARTING FROM THIS POINT STREAMS WILL BE USED
	//OPERATION
	//USING STREAM 1
	//d_ERRnTOt_ot.transpose() = d_ERRnTOt_ht.transpose().array()*( c_t.array().unaryExpr(tanh_functor()) )*o_t*(1-o_t);
	cudaStreamWaitEvent(model->hh_layer_info.s1,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_ot_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s1>>>(model->d_d_ERRnTOt_ot,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP o_tn");
	cudaEventRecord(model->hh_layer_info.err_ot_done,model->hh_layer_info.s1);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 2
	//d_ERRnTOt_ft.transpose() = d_ERRnTOt_ct.transpose().array()*(c_t_prev.array())*f_t*(1-f_t);
	cudaStreamWaitEvent(model->hh_layer_info.s2,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s2>>>(model->d_d_ERRnTOt_ft,model->d_d_ERRnTOt_ct,d_c_t_prev,d_f_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");
	cudaEventRecord(model->hh_layer_info.err_ft_done,model->hh_layer_info.s2);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 3
	//d_ERRnTOt_tanhcpt.transpose() = d_ERRnTOt_ct.transpose().array()*(i_t.array());
	cudaStreamWaitEvent(model->hh_layer_info.s3,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_tanhcpt_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s3>>>(model->d_d_ERRnTOt_tanhcpt,model->d_d_ERRnTOt_ct,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("BP tanh_tn");
	cudaEventRecord(model->hh_layer_info.err_tanhcpt_done,model->hh_layer_info.s3);
		
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 4
	//d_ERRnTOt_it.transpose() = d_ERRnTOt_ct.transpose().array()*(c_prime_t_tanh.array());
	cudaStreamWaitEvent(model->hh_layer_info.s4,model->hh_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s4>>>(model->d_d_ERRnTOt_it,model->d_d_ERRnTOt_ct,d_c_prime_t_tanh,d_i_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP it_tn");
	cudaEventRecord(model->hh_layer_info.err_it_done,model->hh_layer_info.s4);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	dType alpha2 = 1;
	dType beta2 = 0;
	//OPERATION
	//USING STREAM 5,6,7,8,9
	//this is for the error being passed to the lower LSTM layer

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s5);
	cudaStreamWaitEvent(model->hh_layer_info.s5,model->hh_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p1_done,model->hh_layer_info.s5);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s6);
	cudaStreamWaitEvent(model->hh_layer_info.s6,model->hh_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p2_done,model->hh_layer_info.s6);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s7);
	cudaStreamWaitEvent(model->hh_layer_info.s7,model->hh_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p3_done,model->hh_layer_info.s7);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s8);
	cudaStreamWaitEvent(model->hh_layer_info.s8,model->hh_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_M_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p4_done,model->hh_layer_info.s8);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p4_done,0);
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_h_Below,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1 below");

	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_h_Below,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	if(dropout) {
		dropout_kernel<<<256,256,0,model->hh_layer_info.s9>>>(d_dropout_mask,dropout_rate,model->d_d_ERRnTOt_h_Below,LSTM_size*minibatch_size);
	}

	if(model->lower_layer.copy_d_Err_ht) {
		if(model->lower_layer.lower_input) {
			cudaMemcpyAsync(model->lower_layer.input_layer->nodes[index].d_d_ERRt_ht, model->d_d_ERRnTOt_h_Below, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s9);
		}
		else {
			cudaMemcpyAsync(model->lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht, model->d_d_ERRnTOt_h_Below, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,model->hh_layer_info.s9);
		}
	}
	else {
		if(model->lower_layer.lower_input) {
			model->lower_layer.input_layer->nodes[index].d_d_ERRt_ht = model->d_d_ERRnTOt_h_Below;
		}
		else {
			model->lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht = model->d_d_ERRnTOt_h_Below;
		}
	}
	cudaEventRecord(model->hh_layer_info.d_ERR_ht_done,model->hh_layer_info.s9);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	//OPERATION
	//USING STREAM 5,6,7,8,9
	// d_ERRnTOt_htM1.transpose() = (W_ho.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() )) \
	// + (W_hf.transpose()*((d_ERRnTOt_ft.transpose().array() * f_t.array() *(1-f_t.array())).matrix())) \
	// + (W_hi.transpose()*((d_ERRnTOt_it.transpose().array()*i_t.array()*(1-i_t.array())).matrix())) \
	// + (W_hc.transpose()*((d_ERRnTOt_tanhcpt.transpose().array()*(1-c_prime_t_tanh.array().square())).matrix()));

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s5);
	cudaStreamWaitEvent(model->hh_layer_info.s5,model->hh_layer_info.d_ERR_ht_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_ho,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta2,model->d_temp1,LSTM_size),"Error backprop temp1 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p1_done,model->hh_layer_info.s5);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s6);
	cudaStreamWaitEvent(model->hh_layer_info.s6,model->hh_layer_info.d_ERR_ht_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hf,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p2_done,model->hh_layer_info.s6);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s7);
	cudaStreamWaitEvent(model->hh_layer_info.s7,model->hh_layer_info.d_ERR_ht_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hi,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p3_done,model->hh_layer_info.s7);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s8);
	cudaStreamWaitEvent(model->hh_layer_info.s8,model->hh_layer_info.d_ERR_ht_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hc,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	cudaEventRecord(model->hh_layer_info.htm1_p4_done,model->hh_layer_info.s8);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.htm1_p4_done,0);
	cudaStreamWaitEvent(model->hh_layer_info.s9,model->hh_layer_info.d_ERR_ht_done,0);
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1");


	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->hh_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}


	cudaEventRecord(model->hh_layer_info.htm1_done,model->hh_layer_info.s9);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAM 10
	//d_ERRnTOt_ctM1.transpose() = (d_ERRnTOt_ct.transpose().array()*f_t.array());
	cudaStreamWaitEvent(model->hh_layer_info.s10,model->hh_layer_info.backprop_init,0);
	elementwise_mult_kernel<<<kernel,threads_per_block,0,model->hh_layer_info.s10>>>(model->d_d_ERRnTOt_ct,d_f_t,model->d_d_ERRnTOt_ctM1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");

	if(BZ_CUDA::clip_cell) {
		clip_mat_kernel<<<std::min(256,(LSTM_size*minibatch_size + 256 - 1)/256),256,0,model->hh_layer_info.s10>>>(model->d_d_ERRnTOt_ct,BZ_CUDA::error_clip_threshold,LSTM_size*minibatch_size);
	}

	cudaEventRecord(model->hh_layer_info.ctm1_done,model->hh_layer_info.s10);

	
	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	compute_gradients_GPU();

	cudaSetDevice(0);
}


template<typename dType>
void LSTM_HH_Node<dType>::compute_gradients_GPU() {


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

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s11);
	cudaStreamWaitEvent(model->hh_layer_info.s11,model->hh_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hi_grad,LSTM_size),"Backprop W_hi grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_hi_grad_done,model->hh_layer_info.s11);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s12);
	cudaStreamWaitEvent(model->hh_layer_info.s12,model->hh_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hf_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_hf_grad_done,model->hh_layer_info.s12);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s13);
	cudaStreamWaitEvent(model->hh_layer_info.s13,model->hh_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hc_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_hc_grad_done,model->hh_layer_info.s13);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s14);
	cudaStreamWaitEvent(model->hh_layer_info.s14,model->hh_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_ho_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.W_ho_grad_done,model->hh_layer_info.s14);

	//OPERATION
	//USING STREAMS 15,16,17,18
	//compute_temp_mat(model->W);
	//model->M_i_grad.noalias() += (d_ERRnTOt_it.transpose().array() * i_t.array() * (1-i_t.array())).matrix() * temp_mat.transpose();
	//model->M_f_grad.noalias() += (d_ERRnTOt_ft.transpose().array() * f_t.array() * (1-f_t.array())).matrix() * temp_mat.transpose();
	//model->M_o_grad.noalias() += (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1-o_t.array())).matrix() * temp_mat.transpose();
	//model->M_c_grad.noalias() += (d_ERRnTOt_tanhcpt.transpose().array() * (1-c_prime_t_tanh.array().square())).matrix() * temp_mat.transpose();
	alpha = 1;
	beta = 1;

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s15);
	cudaStreamWaitEvent(model->hh_layer_info.s15,model->hh_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_it,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_i_grad,LSTM_size),"Backprop M_i grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.M_i_grad_done,model->hh_layer_info.s15);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s16);
	cudaStreamWaitEvent(model->hh_layer_info.s16,model->hh_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_f_grad,LSTM_size),"Backprop M_f grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.M_f_grad_done,model->hh_layer_info.s16);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s17);
	cudaStreamWaitEvent(model->hh_layer_info.s17,model->hh_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_o_grad,LSTM_size),"Backprop M_o grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.M_o_grad_done,model->hh_layer_info.s17);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s18);
	cudaStreamWaitEvent(model->hh_layer_info.s18,model->hh_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_below,LSTM_size,&beta,model->d_M_c_grad,LSTM_size),"Backprop M_c grad cublas gemm failed\n");
	cudaEventRecord(model->hh_layer_info.M_c_grad_done,model->hh_layer_info.s18);


	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	//OPERATION
	//USING STREAMS 19,20,21,22
	//b_i_grad.noalias() += ((d_ERRnTOt_it.array() * (i_t.array() * (1-i_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	//b_f_grad.noalias() += ((d_ERRnTOt_ft.array() * (f_t.array() * (1-f_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	//b_c_grad.noalias() += (d_ERRnTOt_tanhcpt.array() * (1-c_prime_t_tanh.array().square()).matrix().transpose().array()).colwise().sum().matrix().transpose();
	//b_o_grad.noalias() += ((d_ERRnTOt_ot.array() * (o_t.array() * (1-o_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s19);
	cudaStreamWaitEvent(model->hh_layer_info.s19,model->hh_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_it,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_i_grad,1),"backprop b_i_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_i_grad_done,model->hh_layer_info.s19);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s20);
	cudaStreamWaitEvent(model->hh_layer_info.s20,model->hh_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ft,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_f_grad,1),"backprop b_f_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_f_grad_done,model->hh_layer_info.s20);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s21);
	cudaStreamWaitEvent(model->hh_layer_info.s21,model->hh_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ot,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_o_grad,1),"backprop b_o_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_o_grad_done,model->hh_layer_info.s21);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif

	cublasSetStream(model->hh_layer_info.handle,model->hh_layer_info.s22);
	cudaStreamWaitEvent(model->hh_layer_info.s22,model->hh_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->hh_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_tanhcpt,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_c_grad,1),"backprop b_c_grad failed\n");
	cudaEventRecord(model->hh_layer_info.b_c_grad_done,model->hh_layer_info.s22);

	#ifdef REMOVE_STREAMS
	devSynchAll();
	#endif
	//cudaDeviceSynchronize();

	#ifdef CPU_DEBUG
	cudaDeviceSynchronize();
	#endif

}




template<typename dType>
void LSTM_HH_Node<dType>::dump_LSTM(std::ofstream &LSTM_dump_stream,std::string intro) {

	//cudaSetDevice(model->hh_layer_info.device_number);

	cudaDeviceSynchronize();
	LSTM_dump_stream << intro;

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





