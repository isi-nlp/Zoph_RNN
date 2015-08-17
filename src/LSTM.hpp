
//Constructor
template<typename dType>
LSTM_IH_Node<dType>::LSTM_IH_Node(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m) {

	#ifdef CPU_DEBUG
	init_LSTM_CPU(LSTM_size,minibatch_size,vocab_size,m);
	#endif
	init_LSTM_GPU(LSTM_size,minibatch_size,vocab_size,m);

	model = m;
	this->minibatch_size = minibatch_size;
	this->LSTM_size = LSTM_size;
}

template<typename dType>
void LSTM_IH_Node<dType>::init_LSTM_CPU(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m) {
	i_t.resize(LSTM_size,minibatch_size);
	f_t.resize(LSTM_size,minibatch_size);
	c_t.resize(LSTM_size,minibatch_size);
	c_prime_t_tanh.resize(LSTM_size,minibatch_size); //This is the tanh of c
	o_t.resize(LSTM_size,minibatch_size);
	h_t.resize(LSTM_size,minibatch_size);

	h_t.setZero();
	c_t.setZero();

	//Needed for forward propagation
	//These have dimension (hidden state size)x(size of minibatch)
	h_t_prev.resize(LSTM_size,minibatch_size); //h_(t-1)
	c_t_prev.resize(LSTM_size,minibatch_size); //c_(t-1)

	//Temp matrix for computing embedding layer weight matrix multiplied by one-hot matrix
	//Dim (hidden state size)x(minibatch size)
	temp_mat.resize(LSTM_size,minibatch_size);

	//This is the length of the minibatch
	//Each index ranges from 0 to (input vocab size-1), in order to select matrix column from embedding layer
	//This is for x_t, since it is a one-hot vector
	vocab_indices_input.resize(minibatch_size,1);

	//This is the length of the minibatch
	//Each index ranges from 0 to (output vocab size-1), in order to select matrix column from embedding layer
	//This is for softmax layer
	vocab_indices_output.resize(minibatch_size,1);

	//This is the derivative of the error from time n to time t with respect to h_t
	//Has size (minibatch size)x(hidden state size)
	d_ERRnTOt_ht.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_ot.resize(minibatch_size,LSTM_size);

	d_ERRt_ct.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_ft.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_tanhcpt.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_it.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_htM1.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_ctM1.resize(minibatch_size,LSTM_size);

	Z_i.resize(minibatch_size,LSTM_size);
	Z_f.resize(minibatch_size,LSTM_size);
	Z_c.resize(minibatch_size,LSTM_size);
	Z_o.resize(minibatch_size,LSTM_size);

	// model = m;
	
	// //-----------------------------------------------GPU stuff-----------------------------------------------
	// this->minibatch_size = minibatch_size;
}

template<typename dType>
void LSTM_IH_Node<dType>::init_LSTM_GPU(int LSTM_size,int minibatch_size,int vocab_size,struct Input_To_Hidden_Layer<dType> *m) {

	//full_matrix_setup(&h_d_ERRt_ht,&d_d_ERRt_ht,LSTM_size,minibatch_size);
	//full_matrix_setup(&h_d_ERRnTOt_ht,&d_d_ERRnTOt_ht,LSTM_size,minibatch_size);
	full_matrix_setup(&h_o_t,&d_o_t,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_t,&d_c_t,LSTM_size,minibatch_size);
	//full_matrix_setup(&h_d_ERRt_ct,&d_d_ERRt_ct,LSTM_size,minibatch_size);
	//full_matrix_setup(&h_d_ERRnTOt_ct,&d_d_ERRnTOt_ct,LSTM_size,minibatch_size);
	//full_matrix_setup(&h_d_ERRnTOt_ot,&d_d_ERRnTOt_ot,LSTM_size,minibatch_size);
	full_matrix_setup(&h_f_t,&d_f_t,LSTM_size,minibatch_size);
	//full_matrix_setup(&h_d_ERRnTOt_ft,&d_d_ERRnTOt_ft,LSTM_size,minibatch_size);
	full_matrix_setup(&h_c_prime_t_tanh,&d_c_prime_t_tanh,LSTM_size,minibatch_size);
	full_matrix_setup(&h_i_t,&d_i_t,LSTM_size,minibatch_size);
	// full_matrix_setup(&h_d_ERRnTOt_tanhcpt,&d_d_ERRnTOt_tanhcpt,LSTM_size,minibatch_size);
	// full_matrix_setup(&h_d_ERRnTOt_it,&d_d_ERRnTOt_it,LSTM_size,minibatch_size);
	// full_matrix_setup(&h_d_ERRnTOt_htM1,&d_d_ERRnTOt_htM1,LSTM_size,minibatch_size);
	// full_matrix_setup(&h_d_ERRnTOt_ctM1,&d_d_ERRnTOt_ctM1,LSTM_size,minibatch_size);
	full_matrix_setup(&h_sparse_lookup,&d_sparse_lookup,LSTM_size,minibatch_size);
	full_matrix_setup(&h_h_t,&d_h_t,LSTM_size,minibatch_size);
}


//Update the hidden state and cell state vectors
template<typename dType>
template<typename Derived,typename Derived2>
void LSTM_IH_Node<dType>::update_vectors_forward(const Eigen::MatrixBase<Derived> &h_prev,
	const Eigen::MatrixBase<Derived> &c_prev,
	const Eigen::MatrixBase<Derived2> &vocab,
	int index,
	int *d_input_vocab_indices,int *d_input_vocab_indices_01,
	dType *d_h_t_prev,dType *d_c_t_prev) 
{
	#ifdef CPU_DEBUG
	update_vectors_forward_CPU(h_prev,c_prev,vocab,index);
	#endif
	update_vectors_forward_GPU(d_input_vocab_indices,d_input_vocab_indices_01,d_h_t_prev,d_c_t_prev);
	#ifdef CPU_DEBUG
	update_vectors_forward_DEBUG();
	#endif
}

template<typename dType>
template<typename Derived,typename Derived2>
void LSTM_IH_Node<dType>::update_vectors_forward_CPU(const Eigen::MatrixBase<Derived> &h_prev,
	const Eigen::MatrixBase<Derived> &c_prev,
	const Eigen::MatrixBase<Derived2> &vocab,int index) 
{
	h_t_prev = h_prev;
	c_t_prev = c_prev;
	vocab_indices_input = vocab.col(index);
	temp_mat.setZero(h_t_prev.rows(),h_t_prev.cols());
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

template<typename dType>
void LSTM_IH_Node<dType>::update_vectors_forward_DEBUG() {
	bool pass = true;
	if(minibatch_size!=vocab_indices_input.size()) {
		std::cout << "ERROR: size of vocab vectors being passed in for update_vectors_forward() is different\n";
	}

	//debug vocab indicies
	thrust::device_ptr<int> debug_ptr = thrust::device_pointer_cast(d_input_vocab_indices);
	thrust::device_ptr<int> debug_ptr_2 = thrust::device_pointer_cast(d_input_vocab_indices_01);
	for(int i=0; i<minibatch_size; i++) {
		//std::cout << vocab_indices_input(i) << " | " << debug_ptr[i] << "\n";
		if(vocab_indices_input(i)==-1) {
			if(debug_ptr_2[i]!=0) {
				std::cout << "ERROR in update forward vectors, 01 vec is not zero at -1\n";
			}
			if(debug_ptr[i]!=0) {
				std::cout << "ERROR in update forward vectors, full vec is not zero at -1\n";
			}
		}
		else {
			if(debug_ptr[i]!=vocab_indices_input(i)) {
				std::cout << "ERROR in update forward vectors, full vec does not agree with CPU vec\n";
			}

			if(debug_ptr_2[i]!=1) {
				std::cout << "ERROR in update forward vectors, full vec does not agree with CPU vec\n";
			}
		}
	}
}


//Update the hidden state and cell state vectors for first column in target model
template<typename dType>
template<typename Derived>
void LSTM_IH_Node<dType>::update_vectors_forward_decoder(
	const Eigen::MatrixBase<Derived> &vocab,
	int index,
	int *d_input_vocab_indices,int *d_input_vocab_indices_01) 
{
	#ifdef CPU_DEBUG
	vocab_indices_input = vocab.col(index);
	temp_mat.setZero(h_t_prev.rows(),h_t_prev.cols());
	#endif

	//GPU stuff
	this->d_input_vocab_indices = d_input_vocab_indices;
	this->d_input_vocab_indices_01 = d_input_vocab_indices_01;
}

//Update the output vocab indicies
template<typename dType>
template<typename Derived>
void LSTM_IH_Node<dType>::update_vectors_backward(const Eigen::MatrixBase<Derived> &vocab,int index) {
	#ifdef CPU_DEBUG
	vocab_indices_output = vocab.col(index);
	#endif
}

//For input embedding layer
//Pass in the weight matrix for the embedding layer 
//Need to multithread later
template<typename dType>
template<typename Derived>
void LSTM_IH_Node<dType>::compute_temp_mat(const Eigen::MatrixBase<Derived> &W_mat) {

	for(int i=0; i<vocab_indices_input.rows(); i++) {
		if(vocab_indices_input(i)!=-1) {
			temp_mat.col(i) = W_mat.col(vocab_indices_input(i));
		}
		else {
			//Just assign it to a vector, since it does not matter
			temp_mat.col(i) = W_mat.col(0);
		}
	}
}


template<typename dType>
void LSTM_IH_Node<dType>::forward_prop() {
	#ifdef CPU_DEBUG
	forward_prop_CPU();
	#endif
	forward_prop_GPU();
}

//Compute the forward values for the LSTM node
//This is after the node has recieved the previous hidden and cell state values
template<typename dType>
void LSTM_IH_Node<dType>::forward_prop_CPU() {
	//Input gate
	compute_temp_mat(model->W);

	//input gate
	i_t = ((model->M_i*temp_mat + model->W_hi*h_t_prev).colwise() + model->b_i).array().unaryExpr(sigmoid_functor());

	//Forget gate
	f_t = ((model->M_f*temp_mat + model->W_hf*h_t_prev).colwise() + model->b_f).array().unaryExpr(sigmoid_functor());

	//Cell gate
	c_prime_t_tanh = ((model->M_c*temp_mat + model->W_hc*h_t_prev).colwise() + model->b_c).array().unaryExpr(tanh_functor());

	c_t = ((f_t.array())*(c_t_prev.array())).matrix() + (i_t.array()*(c_prime_t_tanh.array())).matrix();

	//Output gate
	o_t = ((model->M_o*temp_mat + model->W_ho*h_t_prev).colwise() + model->b_o).unaryExpr(sigmoid_functor());

	//Output hidden state
	h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));

	//Now do a check to see if h_t or c_t should be zeroed out
	for(int i=0; i< vocab_indices_input.rows(); i++) {
		if(vocab_indices_input(i)==-1) {
			h_t.col(i).setZero();
			c_t.col(i).setZero();
		}
	}
}

template<typename dType>
void LSTM_IH_Node<dType>::forward_prop_GPU() {

	//cudaDeviceSynchronize();
	//cudaDeviceSynchronize();
	//OPERATION
	//USING STREAM 0
	//compute_temp_mat(model->W);
	//std::cout << "f prop node starting\n";
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	sparse_lookup_kernel<<< kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_sparse_lookup,model->d_W,d_input_vocab_indices,minibatch_size,LSTM_size);
	CUDA_GET_LAST_ERROR("SPARSE");
	cudaEventRecord(model->ih_layer_info.sparse_forward_start,model->ih_layer_info.s0);

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

	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.i_t_part1,0);
	forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(d_i_t,model->d_temp1,model->d_temp2,model->d_b_i,LSTM_size);
	CUDA_GET_LAST_ERROR("i_t");
	cudaEventRecord(model->ih_layer_info.i_t_full,model->ih_layer_info.s2);
	//std::cout << "i_t end\n";
	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(i_t,d_i_t,"i_t in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();
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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s4);
	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hf,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp4,LSTM_size),"Forward prop f_t temp4 failed\n");

	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.f_t_part1,0);
	forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(d_f_t,model->d_temp3,model->d_temp4,model->d_b_f,LSTM_size);
	CUDA_GET_LAST_ERROR("f_t");
	cudaEventRecord(model->ih_layer_info.f_t_full,model->ih_layer_info.s4);

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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_hc,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp6,LSTM_size),"Forward prop c_prime_t_tanh temp6 failed\n");

	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.c_prime_t_tanh_part1,0);
	forward_tanh_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s6>>>(d_c_prime_t_tanh,model->d_temp5,model->d_temp6,model->d_b_c,LSTM_size);
	CUDA_GET_LAST_ERROR("c_prime_t_tanh");
	cudaEventRecord(model->ih_layer_info.c_prime_t_tanh_full,model->ih_layer_info.s6);


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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.sparse_forward_start,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,&alpha,model->d_W_ho,LSTM_size,
		d_h_t_prev,LSTM_size,&beta,model->d_temp8,LSTM_size),"Forward prop o_t temp2 failed\n");

	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.o_t_part1,0);
	forward_sigmoid_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s8>>>(d_o_t,model->d_temp7,model->d_temp8,model->d_b_o,LSTM_size);
	CUDA_GET_LAST_ERROR("o_t");
	cudaEventRecord(model->ih_layer_info.o_t_full,model->ih_layer_info.s8);


	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(o_t,d_o_t,"o_t in forward prop",(dType)0.0001);

	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(c_prime_t_tanh,d_c_prime_t_tanh,"c_prime_t_tanh in forward prop",(dType)0.0001);
	//cudaDeviceSynchronize();
	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(f_t,d_f_t,"f_t in forward prop",(dType)0.0001);

	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(i_t,d_i_t,"i_t in forward prop",(dType)0.0001);

	// cudaDeviceSynchronize();
	// eigen_check_thrust_ptr(c_t_prev,d_c_t_prev,"c_t_prev in forward prop",(dType)0.0001);



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
	//OPERATION
	//h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));
	cudaStreamWaitEvent(model->ih_layer_info.s0,model->ih_layer_info.o_t_full,0);
	forward_h_t_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_h_t,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("h_t");

	//OPERATION
	// for(int i=0; i< vocab_indices_input.rows(); i++) {
	// 	if(vocab_indices_input(i)==-1) {
	// 		h_t.col(i).setZero();
	// 		c_t.col(i).setZero();
	// 	}
	// }
	//cudaDeviceSynchronize();
	zero_c_t_and_h_t<<< kernel,threads_per_block,0,model->ih_layer_info.s0>>>(d_h_t,d_c_t,d_input_vocab_indices_01,LSTM_size);
	CUDA_GET_LAST_ERROR("zero");
	//cudaDeviceSynchronize();
	#ifdef CPU_DEBUG
	cudaDeviceSynchronize();
	eigen_check_thrust_ptr(o_t,d_o_t,"o_t in forward prop",(dType)0.00000001);

	eigen_check_thrust_ptr(i_t,d_i_t,"i_t in forward prop",(dType)0.00000001);
	eigen_check_thrust_ptr(f_t,d_f_t,"f_t in forward prop",(dType)0.00000001);
	eigen_check_thrust_ptr(h_t,d_h_t,"h_t in forward prop",(dType)0.00000001);
	eigen_check_thrust_ptr(c_t,d_c_t,"c_t in forward prop",(dType)0.00000001);
	#endif

	//cudaDeviceSynchronize();
	
}

template<typename dType>
template<typename Derived>
void LSTM_IH_Node<dType>::back_prop(const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ht,const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ct,
	const Eigen::MatrixBase<Derived> &d_ERRt_ht) {

	#ifdef CPU_DEBUG
	back_prop_CPU(d_ERRnTOtp1_ht,d_ERRnTOtp1_ct,d_ERRt_ht);
	#endif
	back_prop_GPU();
}

template<typename dType>
void LSTM_IH_Node<dType>::backprop_prep_GPU(dType *d_d_ERRnTOtp1_ht,dType *d_d_ERRnTOtp1_ct,dType *d_d_ERRt_ht) {
	this->d_d_ERRnTOtp1_ht = d_d_ERRnTOtp1_ht;
	this->d_d_ERRnTOtp1_ct = d_d_ERRnTOtp1_ct;
	this->d_d_ERRt_ht = d_d_ERRt_ht;
}

template<typename dType>
void LSTM_IH_Node<dType>::back_prop_GPU() {

	//std::cout << "back prop node starting\n";

	cudaStreamWaitEvent(model->ih_layer_info.s0,model->model->s_layer_info.d_ERR_ht_done,0);
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
	//USING STREAM ZERO
	//OPERATION
	//d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;
	dType alpha = 1;
	dType beta = 1;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ht,LSTM_size,
		&beta,d_d_ERRt_ht,LSTM_size,model->d_d_ERRnTOt_ht,LSTM_size),"backprop addition failed d_ERRnTOt_ht\n");

	//OPERATION
	//d_ERRt_ct.transpose() = d_ERRnTOt_ht.transpose().array() * (o_t.array()*(1-(c_t).array().unaryExpr(tanh_sq_functor())));
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	d_ERRt_ct_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s0>>>(model->d_d_ERRt_ct,model->d_d_ERRnTOt_ht,d_o_t,d_c_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP c_t");

	//OPERATION
	//d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,d_d_ERRnTOtp1_ct,LSTM_size,
		&beta,model->d_d_ERRt_ct,LSTM_size,model->d_d_ERRnTOt_ct,LSTM_size),"backprop addition failed, d_ERRnTOt_ct \n");

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

	//OPERATION
	//USING STREAM 2
	//d_ERRnTOt_ft.transpose() = d_ERRnTOt_ct.transpose().array()*(c_t_prev.array())*f_t*(1-f_t);
	cudaStreamWaitEvent(model->ih_layer_info.s2,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s2>>>(model->d_d_ERRnTOt_ft,model->d_d_ERRnTOt_ct,d_c_t_prev,d_f_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP f_tn");
	cudaEventRecord(model->ih_layer_info.err_ft_done,model->ih_layer_info.s2);

	//OPERATION
	//USING STREAM 3
	//d_ERRnTOt_tanhcpt.transpose() = d_ERRnTOt_ct.transpose().array()*(i_t.array());
	cudaStreamWaitEvent(model->ih_layer_info.s3,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_tanhcpt_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s3>>>(model->d_d_ERRnTOt_tanhcpt,model->d_d_ERRnTOt_ct,d_i_t,d_c_prime_t_tanh,LSTM_size);
	CUDA_GET_LAST_ERROR("BP tanh_tn");
	cudaEventRecord(model->ih_layer_info.err_tanhcpt_done,model->ih_layer_info.s3);
	
	//OPERATION
	//USING STREAM 4
	//d_ERRnTOt_it.transpose() = d_ERRnTOt_ct.transpose().array()*(c_prime_t_tanh.array());
	cudaStreamWaitEvent(model->ih_layer_info.s4,model->ih_layer_info.backprop_init,0);
	d_ERRnTOt_ft_it_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s4>>>(model->d_d_ERRnTOt_it,model->d_d_ERRnTOt_ct,d_c_prime_t_tanh,d_i_t,LSTM_size);
	CUDA_GET_LAST_ERROR("BP it_tn");
	cudaEventRecord(model->ih_layer_info.err_it_done,model->ih_layer_info.s4);

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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s6);
	cudaStreamWaitEvent(model->ih_layer_info.s6,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hf,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta2,model->d_temp2,LSTM_size),"Error backprop temp2 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p2_done,model->ih_layer_info.s6);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s7);
	cudaStreamWaitEvent(model->ih_layer_info.s7,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hi,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta2,model->d_temp3,LSTM_size),"Error backprop temp3 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p3_done,model->ih_layer_info.s7);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s8);
	cudaStreamWaitEvent(model->ih_layer_info.s8,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,LSTM_size,
		&alpha2,model->d_W_hc,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta2,model->d_temp4,LSTM_size),"Error backprop temp4 htM1\n");
	cudaEventRecord(model->ih_layer_info.htm1_p4_done,model->ih_layer_info.s8);

	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p2_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p3_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s9,model->ih_layer_info.htm1_p4_done,0);
	add_four_matrices_kernel<<< kernel,threads_per_block,0,model->ih_layer_info.s9>>>(model->d_d_ERRnTOt_htM1,model->d_temp1,model->d_temp2,model->d_temp3,model->d_temp4,LSTM_size);
	CUDA_GET_LAST_ERROR("BP htm1");
	cudaEventRecord(model->ih_layer_info.htm1_done,model->ih_layer_info.s9);

	//OPERATION
	//USING STREAM 10
	//d_ERRnTOt_ctM1.transpose() = (d_ERRnTOt_ct.transpose().array()*f_t.array());
	cudaStreamWaitEvent(model->ih_layer_info.s10,model->ih_layer_info.backprop_init,0);
	elementwise_mult_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s10>>>(model->d_d_ERRnTOt_ct,d_f_t,model->d_d_ERRnTOt_ctM1,LSTM_size);
	CUDA_GET_LAST_ERROR("BP ctm1");
	cudaEventRecord(model->ih_layer_info.ctm1_done,model->ih_layer_info.s10);

	#ifdef CPU_DEBUG
	cudaDeviceSynchronize();
	eigen_check_thrust_ptr(d_ERRnTOt_ctM1.transpose(),model->d_d_ERRnTOt_ctM1,"d_ERRnTOt_ctM1 in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(d_ERRnTOt_ctM1.transpose(),model->d_d_ERRnTOt_ctM1,"d_ERRnTOt_ctM1 in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(d_ERRnTOt_ctM1.transpose(),model->d_d_ERRnTOt_ctM1,"d_ERRnTOt_ctM1 in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(d_ERRnTOt_htM1.transpose(),model->d_d_ERRnTOt_htM1,"d_ERRnTOt_htM1 in back prop",(dType)0.00000001);
	#endif
	//eigen_check_thrust_ptr(d_ERRnTOt_ctM1.transpose(),d_d_ERRnTOt_ctM1,"d_ERRnTOt_ctM1 in back prop",(dType)0.0001);

	compute_gradients_GPU();
}


template<typename dType>
void LSTM_IH_Node<dType>::compute_gradients_GPU() {



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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s12);
	cudaStreamWaitEvent(model->ih_layer_info.s12,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hf_grad,LSTM_size),"Backprop W_hf grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hf_grad_done,model->ih_layer_info.s12);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s13);
	cudaStreamWaitEvent(model->ih_layer_info.s13,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_hc_grad,LSTM_size),"Backprop W_hc grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_hc_grad_done,model->ih_layer_info.s13);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s14);
	cudaStreamWaitEvent(model->ih_layer_info.s14,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_h_t_prev,LSTM_size,&beta,model->d_W_ho_grad,LSTM_size),"Backprop W_ho grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.W_ho_grad_done,model->ih_layer_info.s14);

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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s16);
	cudaStreamWaitEvent(model->ih_layer_info.s16,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ft,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_f_grad,LSTM_size),"Backprop M_f grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_f_grad_done,model->ih_layer_info.s16);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s17);
	cudaStreamWaitEvent(model->ih_layer_info.s17,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_ot,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_o_grad,LSTM_size),"Backprop M_o grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_o_grad_done,model->ih_layer_info.s17);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s18);
	cudaStreamWaitEvent(model->ih_layer_info.s18,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,CUBLAS_OP_T,LSTM_size,LSTM_size,minibatch_size,&alpha,
		model->d_d_ERRnTOt_tanhcpt,LSTM_size,d_sparse_lookup,LSTM_size,&beta,model->d_M_c_grad,LSTM_size),"Backprop M_c grad cublas gemm failed\n");
	cudaEventRecord(model->ih_layer_info.M_c_grad_done,model->ih_layer_info.s18);


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

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s20);
	cudaStreamWaitEvent(model->ih_layer_info.s20,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ft,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_f_grad,1),"backprop b_f_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_f_grad_done,model->ih_layer_info.s20);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s21);
	cudaStreamWaitEvent(model->ih_layer_info.s21,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_ot,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_o_grad,1),"backprop b_o_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_o_grad_done,model->ih_layer_info.s21);

	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s22);
	cudaStreamWaitEvent(model->ih_layer_info.s22,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(model->ih_layer_info.handle,CUBLAS_OP_N,LSTM_size,minibatch_size,&alpha,model->d_d_ERRnTOt_tanhcpt,LSTM_size,
		model->d_ones_minibatch,1,&beta,model->d_b_c_grad,1),"backprop b_c_grad failed\n");
	cudaEventRecord(model->ih_layer_info.b_c_grad_done,model->ih_layer_info.s22);

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
	alpha = 1;
	beta = 0;
	//cudaStreamWaitEvent(model->ih_layer_info.s23,model->ih_layer_info.W_grad_full_done,0);
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s23);
	cudaStreamWaitEvent(model->ih_layer_info.s23,model->ih_layer_info.err_it_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_i,LSTM_size,model->d_d_ERRnTOt_it,LSTM_size,&beta,
		model->d_temp5,LSTM_size),"cublas W gradient failed temp5\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p1_done,model->ih_layer_info.s23);

	//cudaStreamWaitEvent(model->ih_layer_info.s24,model->ih_layer_info.W_grad_full_done,0);
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s24);
	cudaStreamWaitEvent(model->ih_layer_info.s24,model->ih_layer_info.err_ft_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_f,LSTM_size,model->d_d_ERRnTOt_ft,LSTM_size,&beta,
		model->d_temp6,LSTM_size),"cublas W gradient failed temp6\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p2_done,model->ih_layer_info.s24);

	//cudaStreamWaitEvent(model->ih_layer_info.s25,model->ih_layer_info.W_grad_full_done,0);
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s25);
	cudaStreamWaitEvent(model->ih_layer_info.s25,model->ih_layer_info.err_ot_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_o,LSTM_size,model->d_d_ERRnTOt_ot,LSTM_size,&beta,
		model->d_temp7,LSTM_size),"cublas W gradient failed temp7\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p3_done,model->ih_layer_info.s25);

	//cudaStreamWaitEvent(model->ih_layer_info.s26,model->ih_layer_info.W_grad_full_done,0);
	cublasSetStream(model->ih_layer_info.handle,model->ih_layer_info.s26);
	cudaStreamWaitEvent(model->ih_layer_info.s26,model->ih_layer_info.err_tanhcpt_done,0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(model->ih_layer_info.handle,CUBLAS_OP_T,CUBLAS_OP_N,LSTM_size,minibatch_size,
		LSTM_size,&alpha,model->d_M_c,LSTM_size,model->d_d_ERRnTOt_tanhcpt,LSTM_size,&beta,
		model->d_temp8,LSTM_size),"cublas W gradient failed temp8\n");
	cudaEventRecord(model->ih_layer_info.W_grad_p4_done,model->ih_layer_info.s26);

	//cudaDeviceSynchronize();
	//cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_full_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p1_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p2_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p3_done,0);
	cudaStreamWaitEvent(model->ih_layer_info.s27,model->ih_layer_info.W_grad_p4_done,0);
	int threads_per_block = 128;
	int num_block = (LSTM_size+threads_per_block-1)/threads_per_block;
	dim3 kernel(minibatch_size,num_block,1);
	W_gradient_kernel<<<kernel,threads_per_block,0,model->ih_layer_info.s27>>>(model->d_W_grad,d_input_vocab_indices,model->d_temp5,
		model->d_temp6,model->d_temp7,model->d_temp8,LSTM_size);
	CUDA_GET_LAST_ERROR("BP w_grad");
	cudaEventRecord(model->ih_layer_info.W_grad_full_done,model->ih_layer_info.s27);

	//cudaDeviceSynchronize();

	#ifdef CPU_DEBUG
	cudaDeviceSynchronize();
	eigen_check_thrust_ptr(model->W_grad,model->d_W_grad,"d_W_grad in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(model->W_hi_grad,model->d_W_hi_grad,"d_W_hi_grad in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(model->W_hf_grad,model->d_W_hf_grad,"d_W_hf_grad in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(model->W_hc_grad,model->d_W_hc_grad,"d_W_hc_grad in back prop",(dType)0.00000001);
	eigen_check_thrust_ptr(model->W_ho_grad,model->d_W_ho_grad,"d_W_ho_grad in back prop",(dType)0.00000001);
	#endif

}






//Computes errors for this LSTM node
template<typename dType>
template<typename Derived>
void LSTM_IH_Node<dType>::back_prop_CPU(const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ht,const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ct,
	const Eigen::MatrixBase<Derived> &d_ERRt_ht) {

	//Now get the derivative of h_t with respect to this error and all after it (t-n)
	d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;

	//Derivative of error at time t with respect to c_t
	//d_ERRt_ct = d_ERRnTOt_ht.array() * (o_t.array()*(1- (c_t).array().unaryExpr(tanh_sq_functor()))).matrix().transpose().array();

	//NON-TRANSPOSE
	d_ERRt_ct.transpose() = d_ERRnTOt_ht.transpose().array() * (o_t.array()*(1-(c_t).array().unaryExpr(tanh_sq_functor())));

	d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;

	//Check to see if we should zero out derivatives
	//Now do a check to see if h_t or c_t should be zeroed out
	for(int i=0; i< vocab_indices_input.rows(); i++) {
		if(vocab_indices_input(i)==-1) {
			d_ERRnTOt_ht.row(i).setZero();
			d_ERRnTOt_ct.row(i).setZero();
		}
	}

	//Derivative of error from time t to n with respect to o_t
	//d_ERRnTOt_ot = d_ERRnTOt_ht.array()*( (c_t.array().unaryExpr(tanh_functor())).matrix().transpose().array() );

	//NON-TRANSPOSE
	d_ERRnTOt_ot.transpose() = d_ERRnTOt_ht.transpose().array()*( c_t.array().unaryExpr(tanh_functor()) );

	//Derivative of Error from t to n with respect to f_t
	//d_ERRnTOt_ft = d_ERRnTOt_ct.array()*(c_t_prev.transpose().array());

	//NON-TRANSPOSE
	d_ERRnTOt_ft.transpose() = d_ERRnTOt_ct.transpose().array()*(c_t_prev.array());

	//This is the derivative of the error from time n to time t with respect to tanhc'_t
	//d_ERRnTOt_tanhcpt = d_ERRnTOt_ct.array()*(i_t.transpose().array());

	//NON-TRANSPOSE
	d_ERRnTOt_tanhcpt.transpose() = d_ERRnTOt_ct.transpose().array()*(i_t.array());

	//This is the derivative of the error from time n to time t with respect to i_t
	//d_ERRnTOt_it = d_ERRnTOt_ct.array()*(c_prime_t_tanh.transpose().array());

	//NON-TRANSPOSE
	d_ERRnTOt_it.transpose() = d_ERRnTOt_ct.transpose().array()*(c_prime_t_tanh.array());


	///////////////////From this point on we can precompute the errors, might be able to do before//////////////////////
	//std::cout <<"-------------TEST1------------\n";
	//std::cout << model->W_ho.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() ) << "\n";

	//std::cout << "-------------TEST2-------------\n";
	//std::cout << 


	//This is the derivative of the error from time n to t with respect to h_(t-1)
	// d_ERRnTOt_htM1 = (model->W_ho.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() )).transpose() \
	// + (model->W_hf.transpose()*((d_ERRnTOt_ft.transpose().array() * f_t.array() *(1-f_t.array())).matrix())).transpose() \
	// + (model->W_hi.transpose()*((d_ERRnTOt_it.transpose().array()*i_t.array()*(1-i_t.array())).matrix())).transpose() \
	// + (model->W_hc.transpose()*((d_ERRnTOt_tanhcpt.transpose().array()*(1-c_prime_t_tanh.array().square())).matrix())).transpose();

	//NON-TRANSPOSE
	d_ERRnTOt_htM1.transpose() = (model->W_ho.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() )) \
	+ (model->W_hf.transpose()*((d_ERRnTOt_ft.transpose().array() * f_t.array() *(1-f_t.array())).matrix())) \
	+ (model->W_hi.transpose()*((d_ERRnTOt_it.transpose().array()*i_t.array()*(1-i_t.array())).matrix())) \
	+ (model->W_hc.transpose()*((d_ERRnTOt_tanhcpt.transpose().array()*(1-c_prime_t_tanh.array().square())).matrix()));

	//Derivative from error from time t to n with respect to ctM1
	//d_ERRnTOt_ctM1 = (d_ERRnTOt_ct.array()*f_t.transpose().array());

	//NON-TRANSPOSE
	d_ERRnTOt_ctM1.transpose() = (d_ERRnTOt_ct.transpose().array()*f_t.array());

	//Update the gradients
	compute_gradients_CPU();
}

//Called for the gradient update for input embedding layer
//Can parallelize later
template<typename dType>
template<typename Derived, typename Derived2>
void LSTM_IH_Node<dType>::sparseGradUpdate(const Eigen::MatrixBase<Derived> &grad_const, const Eigen::MatrixBase<Derived2> &d_Err) {
	UNCONST(Derived,grad_const,grad);
	for(int i=0; i< vocab_indices_input.rows(); i++) {
		if(vocab_indices_input(i)!=-1) {
			for(int j=0; j<grad.rows(); j++) {
				grad(j,vocab_indices_input(i)) += d_Err(i,j);
			}
		}
	}
}

template<typename dType>
void LSTM_IH_Node<dType>::compute_gradients_CPU() {


	//Hiden state matrices
	model->W_hi_grad.noalias() += (h_t_prev*(d_ERRnTOt_it.array() * i_t.transpose().array()*(1-i_t.transpose().array())).matrix()).transpose();
	model->W_hf_grad.noalias() += (h_t_prev*(d_ERRnTOt_ft.array()*f_t.transpose().array()*(1-f_t.transpose().array())).matrix()).transpose();
	model->W_hc_grad.noalias() += (h_t_prev*(d_ERRnTOt_ct.array()*(i_t.transpose().array())*(1-c_prime_t_tanh.transpose().array().square())).matrix()).transpose();
	model->W_ho_grad.noalias() += (h_t_prev*(d_ERRnTOt_ot.array()*o_t.transpose().array()*(1-o_t.transpose().array())).matrix()).transpose();

	//embedded matrices
	compute_temp_mat(model->W);
	model->M_i_grad.noalias() += (d_ERRnTOt_it.transpose().array() * i_t.array() * (1-i_t.array())).matrix() * temp_mat.transpose();
	model->M_f_grad.noalias() += (d_ERRnTOt_ft.transpose().array() * f_t.array() * (1-f_t.array())).matrix() * temp_mat.transpose();
	model->M_o_grad.noalias() += (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1-o_t.array())).matrix() * temp_mat.transpose();
	model->M_c_grad.noalias() += (d_ERRnTOt_tanhcpt.transpose().array() * (1-c_prime_t_tanh.array().square())).matrix() * temp_mat.transpose();

	//Update the bias gradients
	model->b_i_grad.noalias() += ((d_ERRnTOt_it.array() * (i_t.array() * (1-i_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	model->b_f_grad.noalias() += ((d_ERRnTOt_ft.array() * (f_t.array() * (1-f_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	model->b_c_grad.noalias() += (d_ERRnTOt_tanhcpt.array() * (1-c_prime_t_tanh.array().square()).matrix().transpose().array()).colwise().sum().matrix().transpose();
	model->b_o_grad.noalias() += ((d_ERRnTOt_ot.array() * (o_t.array() * (1-o_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();

	compute_W_gradient_CPU();
}

//Seperate function because it is messy
template<typename dType>
void LSTM_IH_Node<dType>::compute_W_gradient_CPU() {

	Z_i = d_ERRnTOt_it.array()*(i_t.array() * (1-i_t.array())).matrix().transpose().array();
	Z_f = d_ERRnTOt_ft.array()*(f_t.array() * (1-f_t.array())).matrix().transpose().array();
	Z_o = d_ERRnTOt_ot.array()*(o_t.array() * (1-o_t.array())).matrix().transpose().array();
	Z_c = d_ERRnTOt_tanhcpt.array()*(1-c_prime_t_tanh.array().square()).matrix().transpose().array();

	for(int i=0; i<vocab_indices_input.rows(); i++) {
		if(vocab_indices_input(i)!=-1) {
			for(int j=0; j<model->W_grad.rows(); j++) {
				dType sumtemp = Z_i.row(i) * model->M_i.col(j);
				sumtemp += Z_f.row(i) * model->M_f.col(j);
				sumtemp += Z_o.row(i) * model->M_o.col(j);
				sumtemp += Z_c.row(i) * model->M_c.col(j);
				model->W_grad(j,vocab_indices_input(i)) += sumtemp;
			}
		}
	}
}


template<typename dType>
void LSTM_IH_Node<dType>::get_d_ERRt_ht_ONE(dType *d_d_ERRt_ht_softmax) {

}

template<typename dType>
void LSTM_IH_Node<dType>::get_d_ERRt_ht_CPU(dType *d_d_ERRt_ht_softmax) {

}

template<typename dType>
void LSTM_IH_Node<dType>::get_d_ERRt_ht_DMA(dType *d_d_ERRt_ht_softmax) {
	
}





