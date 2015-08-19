//Model.hpp file that contains implementations for the model class
template<typename dType>
void neuralMT_model<dType>::initModel(int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
 int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
 std::string input_weight_file,std::string output_weight_file,bool scaled,bool train_perplexity,
 bool truncated_softmax,int shortlist_size,int sampled_size,bool LM) {

	//Initialize the softmax layer
	softmax.init_softmax_layer(target_vocab_size,minibatch_size,this,norm_clip,LSTM_size,clip_gradients,
		learning_rate,longest_sent,scaled,train_perplexity,truncated_softmax,shortlist_size,sampled_size);

	//Now print gpu info
	std::cout << "----------Memory status after softmax layer is made-----------\n";
	print_GPU_Info();

	if(!LM) {
		//Initialize the input layer
		input_layer_source.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101);

		//Now print gpu info
		std::cout << "---------Memory status after source side of input layer is initialized----------\n";
		print_GPU_Info();
	}

	input_layer_target.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102);

	//Initialize the hidden layer
	// hidden_layer.init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,input_vocab_size,output_vocab_size,
 // 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this);

	this->input_weight_file = input_weight_file;
	this->output_weight_file = output_weight_file;
	this->debug = debug;
	zero_error.setZero(minibatch_size,LSTM_size);
	train_perplexity_mode = train_perplexity;
	this->truncated_softmax = truncated_softmax;
	this->LM = LM;

	std::cout << "--------Memory status after target side of input layer is initialized--------\n";
	init_GPUs();

	//Now print gpu info
	print_GPU_Info();
}

template<typename dType>
void neuralMT_model<dType>::print_GPU_Info() {

	int num_devices = -1;
	cudaGetDeviceCount(&num_devices);
	size_t free_bytes, total_bytes = 0;
  	int selected = 0;
  	for (int i = 0; i < num_devices; i++) {
	    cudaDeviceProp prop;
	    cudaGetDeviceProperties(&prop, i);
	    std::cout << "Device Number: " << i << std::endl;
	    std::cout << "Device Name: " << prop.name << std::endl;
	   	cudaSetDevice(i);
	    cudaMemGetInfo( &free_bytes, &total_bytes);
	    std::cout << "Total Memory (MB): " << total_bytes/(1.0e6) << std::endl;
	    std::cout << "Memory Free (MB): " << free_bytes/(1.0e6) << std::endl << std::endl;
  }
  	cudaSetDevice(0);
}

//for this we need to initialze the source minibatch size to one
template<typename dType>
void neuralMT_model<dType>::initModel_decoding(int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
 int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
 std::string input_weight_file,std::string output_weight_file,bool scaled) {

	//Initialize the softmax layer
	softmax.init_softmax_layer(target_vocab_size,minibatch_size,this,norm_clip,LSTM_size,clip_gradients,learning_rate,
		longest_sent,scaled,false,false,0,0);

	//Initialize the input layer
	input_layer_source.init_Input_To_Hidden_Layer(LSTM_size,1,source_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101);

	input_layer_target.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102);

	//Initialize the hidden layer
	// hidden_layer.init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,input_vocab_size,output_vocab_size,
 // 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this);

	this->input_weight_file = input_weight_file;
	this->output_weight_file = output_weight_file;
	this->debug = debug;
	this->LM = false;
	zero_error.setZero(minibatch_size,LSTM_size);

	init_GPUs();
}

template<typename dType>
void neuralMT_model<dType>::init_GPUs() {

	//input layer
	cudaSetDevice(ih_layer_info.device_number);
	CUBLAS_ERROR_WRAPPER(cublasCreate(&ih_layer_info.handle),"CUBLAS handler initialization failed\n");
	cudaStreamCreate(&ih_layer_info.s0);
	cudaStreamCreate(&ih_layer_info.s1);
	cudaStreamCreate(&ih_layer_info.s2);
	cudaStreamCreate(&ih_layer_info.s3);
	cudaStreamCreate(&ih_layer_info.s4);
	cudaStreamCreate(&ih_layer_info.s5);
	cudaStreamCreate(&ih_layer_info.s6);
	cudaStreamCreate(&ih_layer_info.s7);
	cudaStreamCreate(&ih_layer_info.s8);
	cudaStreamCreate(&ih_layer_info.s9);
	cudaStreamCreate(&ih_layer_info.s10);
	cudaStreamCreate(&ih_layer_info.s11);
	cudaStreamCreate(&ih_layer_info.s12);
	cudaStreamCreate(&ih_layer_info.s13);
	cudaStreamCreate(&ih_layer_info.s14);
	cudaStreamCreate(&ih_layer_info.s15);
	cudaStreamCreate(&ih_layer_info.s16);
	cudaStreamCreate(&ih_layer_info.s17);
	cudaStreamCreate(&ih_layer_info.s18);
	cudaStreamCreate(&ih_layer_info.s19);
	cudaStreamCreate(&ih_layer_info.s20);
	cudaStreamCreate(&ih_layer_info.s21);
	cudaStreamCreate(&ih_layer_info.s22);
	cudaStreamCreate(&ih_layer_info.s23);
	cudaStreamCreate(&ih_layer_info.s24);
	cudaStreamCreate(&ih_layer_info.s25);
	cudaStreamCreate(&ih_layer_info.s26);
	cudaStreamCreate(&ih_layer_info.s27);

	cudaEventCreate(&ih_layer_info.sparse_forward_start);
	cudaEventCreate(&ih_layer_info.i_t_part1);
	cudaEventCreate(&ih_layer_info.i_t_full);
	cudaEventCreate(&ih_layer_info.f_t_part1);
	cudaEventCreate(&ih_layer_info.f_t_full);
	cudaEventCreate(&ih_layer_info.c_prime_t_tanh_part1);
	cudaEventCreate(&ih_layer_info.c_prime_t_tanh_full);
	cudaEventCreate(&ih_layer_info.o_t_part1);
	cudaEventCreate(&ih_layer_info.o_t_full);
	cudaEventCreate(&ih_layer_info.W_grad_full_done);

	cudaEventCreate(&ih_layer_info.backprop_init);
	cudaEventCreate(&ih_layer_info.err_ot_done);
	cudaEventCreate(&ih_layer_info.err_ft_done);
	cudaEventCreate(&ih_layer_info.err_tanhcpt_done);
	cudaEventCreate(&ih_layer_info.err_it_done);
	cudaEventCreate(&ih_layer_info.htm1_p1_done);
	cudaEventCreate(&ih_layer_info.htm1_p2_done);
	cudaEventCreate(&ih_layer_info.htm1_p3_done);
	cudaEventCreate(&ih_layer_info.htm1_p4_done);

	cudaEventCreate(&ih_layer_info.W_grad_p1_done);
	cudaEventCreate(&ih_layer_info.W_grad_p2_done);
	cudaEventCreate(&ih_layer_info.W_grad_p3_done);
	cudaEventCreate(&ih_layer_info.W_grad_p4_done);

	cudaEventCreate(&ih_layer_info.htm1_done);
	cudaEventCreate(&ih_layer_info.ctm1_done);

	cudaEventCreate(&ih_layer_info.W_hi_grad_done);
	cudaEventCreate(&ih_layer_info.W_hf_grad_done);
	cudaEventCreate(&ih_layer_info.W_ho_grad_done);
	cudaEventCreate(&ih_layer_info.W_hc_grad_done);
	cudaEventCreate(&ih_layer_info.M_i_grad_done);
	cudaEventCreate(&ih_layer_info.M_f_grad_done);
	cudaEventCreate(&ih_layer_info.M_o_grad_done);
	cudaEventCreate(&ih_layer_info.M_c_grad_done);
	cudaEventCreate(&ih_layer_info.b_i_grad_done);
	cudaEventCreate(&ih_layer_info.b_f_grad_done);
	cudaEventCreate(&ih_layer_info.b_o_grad_done);
	cudaEventCreate(&ih_layer_info.b_c_grad_done);


	//softmax layer
	cudaSetDevice(s_layer_info.device_number);
	CUBLAS_ERROR_WRAPPER(cublasCreate(&s_layer_info.handle),"CUBLAS handler initialization failed\n");
	cudaStreamCreate(&s_layer_info.s0);
	cudaStreamCreate(&s_layer_info.s1);
	cudaStreamCreate(&s_layer_info.s2);
	cudaStreamCreate(&s_layer_info.s3);

	cudaEventCreate(&s_layer_info.outputdist_done);
	cudaEventCreate(&s_layer_info.d_ERR_ht_done);
	cudaEventCreate(&s_layer_info.d_D_grad_done);
	cudaEventCreate(&s_layer_info.d_b_d_grad_done);

	//give the layers their struct info's
	input_layer_source.ih_layer_info = ih_layer_info;
	input_layer_target.ih_layer_info = ih_layer_info;
	softmax.s_layer_info = s_layer_info;

	cudaSetDevice(0);
}


template<typename dType>
template<typename Derived>
void neuralMT_model<dType>::compute_gradients(const Eigen::MatrixBase<Derived> &source_input_minibatch_const,
	const Eigen::MatrixBase<Derived> &source_output_minibatch_const,const Eigen::MatrixBase<Derived> &target_input_minibatch_const,
	const Eigen::MatrixBase<Derived> &target_output_minibatch_const,int *h_input_vocab_indicies_source,
	int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_output_vocab_indicies_target,
	int current_source_length,int current_target_length,int *h_input_vocab_indicies_source_Wgrad,int *h_input_vocab_indicies_target_Wgrad,
	int len_source_Wgrad,int len_target_Wgrad,int *h_sampled_indices,int len_unique_words_trunc_softmax) 
{
	//Clear the gradients before forward/backward pass
	//eventually clear gradients at the end
	//clear_gradients();

	//Send the CPU vocab input data to the GPU layers
	//For the input layer, 2 host vectors must be transfered since need special preprocessing for W gradient
	if(!LM){
		input_layer_source.prep_GPU_vocab_indices(h_input_vocab_indicies_source,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
	}
	input_layer_target.prep_GPU_vocab_indices(h_input_vocab_indicies_target,h_input_vocab_indicies_target_Wgrad,current_target_length,len_target_Wgrad);
	softmax.prep_GPU_vocab_indices(h_output_vocab_indicies_target,current_target_length);
	if(truncated_softmax) {
		softmax.prep_trunc(h_sampled_indices,len_unique_words_trunc_softmax);
	}
	cudaDeviceSynchronize();


	if(!LM) {
		//Do the source side forward pass
		input_layer_source.nodes[0].update_vectors_forward(input_layer_source.init_hidden_vector,input_layer_source.init_cell_vector,
			source_input_minibatch_const,0,input_layer_source.d_input_vocab_indices_full,input_layer_source.d_input_vocab_indices_01_full,
			input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
		input_layer_source.nodes[0].forward_prop();
		//cudaDeviceSynchronize();
		//for(int i=1; i<source_input_minibatch_const.cols(); i++) {
		for(int i=1; i<current_source_length; i++) {
			int step = i*input_layer_source.minibatch_size;
			input_layer_source.nodes[i].update_vectors_forward(input_layer_source.nodes[i-1].h_t,
				input_layer_source.nodes[i-1].c_t,source_input_minibatch_const,i,
				input_layer_source.d_input_vocab_indices_full+step,input_layer_source.d_input_vocab_indices_01_full+step,
				input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t);
			input_layer_source.nodes[i].forward_prop();
			//cudaDeviceSynchronize();
		}
	}
	//Send stuff to GPU at this point?? Could get speedups this way

	//Do the target side forward pass
	//int prev_source_index = source_input_minibatch_const.cols()-1;

	if(LM) {
		input_layer_target.nodes[0].update_vectors_forward(input_layer_target.init_hidden_vector,
			input_layer_target.init_cell_vector,target_input_minibatch_const,0,
			input_layer_target.d_input_vocab_indices_full,input_layer_target.d_input_vocab_indices_01_full,
			input_layer_target.d_init_hidden_vector,input_layer_target.d_init_cell_vector);
	}
	else {
		int prev_source_index = current_source_length-1;
		input_layer_target.nodes[0].update_vectors_forward(input_layer_source.nodes[prev_source_index].h_t,
			input_layer_source.nodes[prev_source_index].c_t,target_input_minibatch_const,0,
			input_layer_target.d_input_vocab_indices_full,input_layer_target.d_input_vocab_indices_01_full,
			input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t);
	}
	input_layer_target.nodes[0].forward_prop();
	//cudaDeviceSynchronize();
	//for(int i=1; i<target_input_minibatch_const.cols(); i++) {
	for(int i=1; i<current_target_length; i++) {
		int step = i*input_layer_target.minibatch_size;
		input_layer_target.nodes[i].update_vectors_forward(input_layer_target.nodes[i-1].h_t,input_layer_target.nodes[i-1].c_t,
			target_input_minibatch_const,i,input_layer_target.d_input_vocab_indices_full+step,
			input_layer_target.d_input_vocab_indices_01_full+step,
			input_layer_target.nodes[i-1].d_h_t,input_layer_target.nodes[i-1].d_c_t);
		input_layer_target.nodes[i].forward_prop();
		//cudaDeviceSynchronize();
	}

	cudaDeviceSynchronize();
	/////////////////////////////////////////backward pass/////////////////////////////////////////////////

	////////////////////////////Do the backward pass for the target first////////////////////////////
	//int last_index = target_output_minibatch_const.cols()-1;
	int last_index = current_target_length-1;
	input_layer_target.nodes[last_index].update_vectors_backward(target_output_minibatch_const,last_index);

	//transfer h_t to the softmax at this point
	//int step = (target_output_minibatch_const.cols()-1)*input_layer_target.minibatch_size;
	int step = (current_target_length-1)*input_layer_target.minibatch_size;
	softmax.backprop_prep_GPU(input_layer_target.nodes[last_index].d_h_t,softmax.d_output_vocab_indices+step,softmax.d_output_vocab_indices_01+step,
		softmax.d_output_vocab_indices_01_float+step);

	//record these two events to start for the GPU

	softmax.compute_gradient(input_layer_target.nodes[last_index].h_t,input_layer_target.nodes[last_index].vocab_indices_output); 
	//cudaDeviceSynchronize();
	input_layer_target.nodes[last_index].backprop_prep_GPU(input_layer_target.d_init_d_ERRnTOtp1_ht,input_layer_target.d_init_d_ERRnTOtp1_ct,
			softmax.d_d_ERRt_ht);
	//cudaDeviceSynchronize();
	input_layer_target.nodes[last_index].back_prop(input_layer_target.init_d_ERRnTOtp1_ht,input_layer_target.init_d_ERRnTOtp1_ct,softmax.d_ERRt_ht);
	//cudaDeviceSynchronize();
	//for(int i=target_output_minibatch_const.cols()-2; i>=0; i--) {
	for(int i=current_target_length-2; i>=0; i--) {

		step = i*input_layer_target.minibatch_size;
		input_layer_target.nodes[i].update_vectors_backward(target_output_minibatch_const,i);

		softmax.backprop_prep_GPU(input_layer_target.nodes[i].d_h_t,softmax.d_output_vocab_indices+step,softmax.d_output_vocab_indices_01+step,
			softmax.d_output_vocab_indices_01_float+step);

		softmax.compute_gradient(input_layer_target.nodes[i].h_t,input_layer_target.nodes[i].vocab_indices_output); 

		//cudaDeviceSynchronize();
		// input_layer_target.nodes[i].backprop_prep_GPU(input_layer_target.nodes[i+1].d_d_ERRnTOt_htM1,input_layer_target.nodes[i+1].d_d_ERRnTOt_ctM1,
		// 	softmax.d_d_ERRt_ht);
		input_layer_target.nodes[i].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,input_layer_target.d_d_ERRnTOt_ctM1,
			softmax.d_d_ERRt_ht);

		input_layer_target.nodes[i].back_prop(input_layer_target.nodes[i+1].d_ERRnTOt_htM1,input_layer_target.nodes[i+1].d_ERRnTOt_ctM1,softmax.d_ERRt_ht);
		//cudaDeviceSynchronize();
	}


	///////////////////////////Now do the backward pass for the source///////////////////////
	//prev_source_index = source_output_minibatch_const.cols()-1;
	if(!LM) {
		int prev_source_index = current_source_length-1;

		input_layer_source.nodes[prev_source_index].update_vectors_backward(source_output_minibatch_const,prev_source_index);

		// input_layer_source.nodes[prev_source_index].backprop_prep_GPU(input_layer_target.nodes[0].d_d_ERRnTOt_htM1,
		// 	input_layer_target.nodes[0].d_d_ERRnTOt_ctM1,input_layer_source.d_zeros);
		input_layer_source.nodes[prev_source_index].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,
		 	input_layer_target.d_d_ERRnTOt_ctM1,input_layer_source.d_zeros);

		input_layer_source.nodes[prev_source_index].back_prop(input_layer_target.nodes[0].d_ERRnTOt_htM1,input_layer_target.nodes[0].d_ERRnTOt_ctM1,zero_error);
		//cudaDeviceSynchronize();
		//for(int i=source_output_minibatch_const.cols()-2; i>=0; i--) {
		for(int i=current_source_length-2; i>=0; i--) {

			input_layer_source.nodes[i].update_vectors_backward(source_output_minibatch_const,i);

			// input_layer_source.nodes[i].backprop_prep_GPU(input_layer_source.nodes[i+1].d_d_ERRnTOt_htM1,input_layer_source.nodes[i+1].d_d_ERRnTOt_ctM1,
			// 	input_layer_source.d_zeros);

			input_layer_source.nodes[i].backprop_prep_GPU(input_layer_source.d_d_ERRnTOt_htM1,input_layer_source.d_d_ERRnTOt_ctM1,
				input_layer_source.d_zeros);

			input_layer_source.nodes[i].back_prop(input_layer_source.nodes[i+1].d_ERRnTOt_htM1,input_layer_source.nodes[i+1].d_ERRnTOt_ctM1,zero_error);
			//cudaDeviceSynchronize();
		}
	}

	if(debug) {
		dType epsilon =(dType)1e-5;
		cudaDeviceSynchronize();
		check_all_gradients(epsilon);
	}

	// //Update the model parameter weights
	update_weights();

	clear_gradients();

	cudaDeviceSynchronize();

	if(train_perplexity_mode) {
		double tmp_perp;
		cudaMemcpy(&tmp_perp,softmax.d_train_perplexity,1*sizeof(double),cudaMemcpyDeviceToHost);
		train_perplexity+=tmp_perp;
		cudaMemset(softmax.d_train_perplexity,0,1*sizeof(double));
	}
}

template<typename dType>
void neuralMT_model<dType>::clear_gradients() {
	if(!LM) {
		input_layer_source.clear_gradients(false);
	}
	input_layer_target.clear_gradients(false);
	softmax.clear_gradients();
}

template<typename dType>
double neuralMT_model<dType>::getError(bool GPU) 
{
	double loss=0;

	if(!LM) {
		input_layer_source.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_source,file_info->h_input_vocab_indicies_source_Wgrad,
			file_info->current_source_length,file_info->len_source_Wgrad);
	}
	input_layer_target.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_target,file_info->h_input_vocab_indicies_target_Wgrad,
		file_info->current_target_length,file_info->len_target_Wgrad);
	softmax.prep_GPU_vocab_indices(file_info->h_output_vocab_indicies_target,file_info->current_target_length);
	cudaDeviceSynchronize();

	if(!LM) {
		input_layer_source.nodes[0].update_vectors_forward(input_layer_source.init_hidden_vector,input_layer_source.init_cell_vector,
			file_info->minibatch_tokens_source_input,0,input_layer_source.d_input_vocab_indices_full,input_layer_source.d_input_vocab_indices_01_full,
			input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
		input_layer_source.nodes[0].forward_prop();
		//for(int i=1; i<file_info->minibatch_tokens_source_input.cols(); i++) {
		for(int i=1; i<file_info->current_source_length; i++) {
			int step = i*input_layer_source.minibatch_size;
			input_layer_source.nodes[i].update_vectors_forward(input_layer_source.nodes[i-1].h_t,input_layer_source.nodes[i-1].c_t,
				file_info->minibatch_tokens_source_input,i,input_layer_source.d_input_vocab_indices_full+step,
				input_layer_source.d_input_vocab_indices_01_full+step,
				input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t);
			input_layer_source.nodes[i].forward_prop();
			//cudaDeviceSynchronize();
		}
	}


	//std::cout << "----------------STARTING TARGET SIDE FOR GET ERROR----------------\n";
	//Do the target side forward pass
	//int prev_source_index = file_info->minibatch_tokens_source_input.cols()-1;
	if(LM) {
		input_layer_target.nodes[0].update_vectors_forward(input_layer_target.init_hidden_vector,
			input_layer_target.init_cell_vector,
			file_info->minibatch_tokens_target_input,0,input_layer_target.d_input_vocab_indices_full,
			input_layer_target.d_input_vocab_indices_01_full,
			input_layer_target.d_init_hidden_vector,input_layer_target.d_init_cell_vector);
	}
	else {
		int prev_source_index = file_info->current_source_length-1;
		input_layer_target.nodes[0].update_vectors_forward(input_layer_source.nodes[prev_source_index].h_t,input_layer_source.nodes[prev_source_index].c_t,
			file_info->minibatch_tokens_target_input,0,input_layer_target.d_input_vocab_indices_full,
			input_layer_target.d_input_vocab_indices_01_full,
			input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t);
	}	

	input_layer_target.nodes[0].forward_prop();
	cudaDeviceSynchronize();
	//note d_h_t can be null for these as all we need is the vocab pointers correct for getting the error
	softmax.backprop_prep_GPU(input_layer_target.nodes[0].d_h_t,softmax.d_output_vocab_indices,softmax.d_output_vocab_indices_01,
		softmax.d_output_vocab_indices_01_float);

	if(GPU) {
		#ifdef CPU_DEBUG
		//This is here to pass the CPU-GPU tests being called
		softmax.compute_loss(input_layer_target.nodes[0].h_t,file_info->minibatch_tokens_target_output.col(0));//for debug checking
		#endif
		loss += softmax.compute_loss_GPU();
	}
	else {
		loss += softmax.compute_loss(input_layer_target.nodes[0].h_t,file_info->minibatch_tokens_target_output.col(0));
	}
	cudaDeviceSynchronize();

	//for(int i=1; i<file_info->minibatch_tokens_target_input.cols(); i++) {
	for(int i=1; i<file_info->current_target_length; i++) {
		int step = i*input_layer_target.minibatch_size;

		input_layer_target.nodes[i].update_vectors_forward(input_layer_target.nodes[i-1].h_t,input_layer_target.nodes[i-1].c_t,
			file_info->minibatch_tokens_target_input,i,input_layer_target.d_input_vocab_indices_full+step,
			input_layer_target.d_input_vocab_indices_01_full+step,
			input_layer_target.nodes[i-1].d_h_t,input_layer_target.nodes[i-1].d_c_t);

		input_layer_target.nodes[i].forward_prop();
		cudaDeviceSynchronize();
		softmax.backprop_prep_GPU(input_layer_target.nodes[i].d_h_t,softmax.d_output_vocab_indices+step,softmax.d_output_vocab_indices_01+step,
			softmax.d_output_vocab_indices_01_float+step);

		if(GPU) {
			#ifdef CPU_DEBUG
			softmax.compute_loss(input_layer_target.nodes[i].h_t,file_info->minibatch_tokens_target_output.col(i));
			#endif
			loss += softmax.compute_loss_GPU();
		}
		else {
			loss += softmax.compute_loss(input_layer_target.nodes[i].h_t,file_info->minibatch_tokens_target_output.col(i));
		}
		cudaDeviceSynchronize();
	}

	return loss;
}



template<typename dType>
void neuralMT_model<dType>::check_all_gradients(dType epsilon) 
{
	if(!LM) {
		input_layer_source.check_all_gradients(epsilon);
	}
	input_layer_target.check_all_gradients(epsilon);
	softmax.check_all_gradients(epsilon);
	//hidden_layer.check_all_gradients(epsilon,input_minibatch_const,output_minibatch_const);
}


//Update the model parameters
template<typename dType>
void neuralMT_model<dType>::update_weights() {
	cudaDeviceSynchronize();
	softmax.update_weights();
	if(!LM) {
		input_layer_source.update_weights();
	}
	input_layer_target.update_weights();
	//hidden_layer.update_weights();
}


template<typename dType>
void neuralMT_model<dType>::dump_weights() {
	output.open(output_weight_file.c_str(),std::ios_base::app);

	output.precision(std::numeric_limits<dType>::digits10 + 2);
	//output.flush();
	if(!LM) {
		input_layer_source.dump_weights(output);
	}
	//output.flush();
	input_layer_target.dump_weights(output);
	//output.flush();
	softmax.dump_weights(output);
	//output.flush();
	output.close();
	//output.flush();
}


//Load in the weights from a file, so the model can be used
template<typename dType>
void neuralMT_model<dType>::load_weights() {
	//input.open("aaaaa");
	input.open(input_weight_file.c_str());

	//now load the weights by bypassing the intro stuff
	std::string str;
	std::string word;
	std::getline(input, str);
	std::getline(input, str);
	while(std::getline(input, str)) {
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
		}
	}

	if(!LM) {
		while(std::getline(input, str)) {
			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
					break; //done with source mapping
			}
		}
	}

	if(!LM) {
		input_layer_source.load_weights(input);
	}
	input_layer_target.load_weights(input);
	//input.sync();
	softmax.load_weights(input);

	input.close();
}

template<typename dType>
void neuralMT_model<dType>::initFileInfo(struct file_helper *file_info) {
	this->file_info = file_info;
}


template<typename dType>
void neuralMT_model<dType>::update_learning_rate(dType new_learning_rate) {

	input_layer_source.learning_rate = new_learning_rate;
	input_layer_target.learning_rate = new_learning_rate;
	softmax.learning_rate = new_learning_rate;
}


template<typename dType>
double neuralMT_model<dType>::get_perplexity(std::string test_file_name,int minibatch_size,int &test_num_lines_in_file, int longest_sent,
	int source_vocab_size,int target_vocab_size,std::ofstream &HPC_output,bool load_weights_val,int &test_total_words,bool HPC_output_flag,
	bool force_decode,std::string fd_filename) 
{

	if(load_weights_val) {
		load_weights();
	}
	//set trunc softmax to zero always for perplexity!
	file_helper file_info(test_file_name,minibatch_size,test_num_lines_in_file,longest_sent,
		source_vocab_size,target_vocab_size,test_total_words,false,0,0); //Initialize the file information
	initFileInfo(&file_info);

	std::ofstream fd_stream;
	if(force_decode) {
		fd_stream.open(fd_filename);
	}

	if(truncated_softmax && !load_weights_val) {
		//copy the d_subset_D and d_subset_b_d
		load_shortlist_D<<<256,256>>>(softmax.d_subset_D,softmax.d_D,softmax.LSTM_size,softmax.trunc_size,softmax.output_vocab_size,softmax.shortlist_size);
		load_shortlist_D<<<256,256>>>(softmax.d_subset_b_d,softmax.d_b_d,1,softmax.trunc_size,softmax.output_vocab_size,softmax.shortlist_size);
		cudaDeviceSynchronize();
	}

	int current_epoch = 1;
	std::cout << "Getting perplexity of dev set" << std::endl;
	if(HPC_output_flag) {
		HPC_output << "Getting perplexity of dev set" << std::endl;
	}
	//int total_words = 0; //For perplexity
	//double P_data = 0;
	double P_data_GPU = 0;
	while(current_epoch <= 1) {
		bool success = file_info.read_minibatch();
		//P_data += getError(false);
		double temp = getError(true);
		fd_stream << temp << "\n";
		P_data_GPU += temp;
		//total_words += file_info.words_in_minibatch;
		if(!success) {
			current_epoch+=1;
		}
	}
	//P_data = P_data/std::log(2.0); //Change to base 2 log
	P_data_GPU = P_data_GPU/std::log(2.0); 
	//double perplexity = std::pow(2,-1*P_data/file_info.num_target_words);
	double perplexity_GPU = std::pow(2,-1*P_data_GPU/file_info.total_target_words);
	std::cout << "Total target words: " << file_info.total_target_words << "\n";
	//std::cout << "Perplexity CPU : " << perplexity << std::endl;
	std::cout <<  std::setprecision(15) << "Perplexity dev set: " << perplexity_GPU << std::endl;
	std::cout <<  std::setprecision(15) << "P_data dev set: " << P_data_GPU << std::endl;
	//fd_stream << perplexity_GPU << "\n";
	if(HPC_output_flag) {
		HPC_output <<  std::setprecision(15) << "P_data: " << P_data_GPU << std::endl;
		HPC_output <<  std::setprecision(15) << "Perplexity dev set: " << perplexity_GPU << std::endl;
	}

	return perplexity_GPU;
}



template<typename dType>
template<typename Derived>
void neuralMT_model<dType>::decoder_forward_prop_source(const Eigen::MatrixBase<Derived> &source_vocab_indices,int *d_input_vocab_indicies_source,int *d_one) {

	cudaDeviceSynchronize();
	//std::cout << "------starting forward prop source -----------------\n";
	input_layer_source.nodes[0].update_vectors_forward(input_layer_source.init_hidden_vector,input_layer_source.init_cell_vector,
		source_vocab_indices,0,d_input_vocab_indicies_source,d_one,
		input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
	input_layer_source.nodes[0].forward_prop();

	for(int i=1; i<source_vocab_indices.cols(); i++) {
		int step = i; //since minibatch size is 1
		input_layer_source.nodes[i].update_vectors_forward(input_layer_source.nodes[i-1].h_t,input_layer_source.nodes[i-1].c_t,
			source_vocab_indices,i,d_input_vocab_indicies_source + step,d_one,
			input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t);
		input_layer_source.nodes[i].forward_prop();
	}

	//std::cout << "SOURCE LENGTH: " << source_vocab_indices.cols() << "\n";

	cudaDeviceSynchronize();
	// std::cout << "---------------FIRST h_t debug---------------\n";
	// std::cout << input_layer_source.nodes[0].h_t << "\n\n";
	// std::cout << "index being used for input: " << source_vocab_indices.col(0) << "\n";

	// std::cout << "---------------SECOND h_t debug---------------\n";
	// std::cout << input_layer_source.nodes[1].h_t << "\n\n";
	// std::cout << "index being used for input: " << source_vocab_indices.col(1) << "\n";

	// std::cout << "---------------LAST h_t debug---------------\n";
	// std::cout << input_layer_source.nodes[source_vocab_indices.cols()-1].h_t << "\n\n";
	// std::cout << "index being used for input: " << source_vocab_indices.col(source_vocab_indices.cols()-1) << "\n";

	// std::cout << "---------------LAST c_t debug---------------\n";
	// std::cout << input_layer_source.nodes[source_vocab_indices.cols()-1].c_t << "\n\n";
	// std::cout << "index being used for input: " << source_vocab_indices.col(source_vocab_indices.cols()-1) << "\n";
}


template<typename dType>
void neuralMT_model<dType>::decoder_forward_prop_target(struct file_helper_decoder *fh,struct decoder<dType> *d,int *d_ones,int curr_index,
	dType *h_outputdist) 
{
	//Do the target side forward pass
	//std::cout << "-----forward prop target index: " << curr_index << "\n";
	if(curr_index==0) {
		//std::cout << "FORWARD PROP TARGET DEBUG\n";
		int prev_source_index = fh->sentence_length-1;

		input_layer_target.transfer_decoding_states(input_layer_source.nodes[prev_source_index].h_t,input_layer_source.nodes[prev_source_index].c_t);
		input_layer_target.transfer_decoding_states_GPU(input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t);

		input_layer_target.nodes[curr_index].update_vectors_forward_decoder(d->current_indices,0,d->d_current_indices,d_ones);
		input_layer_target.nodes[curr_index].forward_prop();
		// std::cout << "-------------First target h_t after transfer---------\n";
		// std::cout << input_layer_target.nodes[0].h_t << "\n\n";
		// std::cout << input_layer_target.nodes[0].c_t << "\n\n";
		cudaDeviceSynchronize();
		softmax.backprop_prep_GPU(input_layer_target.nodes[curr_index].d_h_t,NULL,NULL,NULL);
		softmax.getDist(input_layer_target.nodes[curr_index].h_t);
		softmax.get_distribution_GPU(softmax.output_vocab_size,softmax.d_outputdist,softmax.d_D,softmax.d_b_d); //non-trunc
	}
	else {
		int step = curr_index*input_layer_target.minibatch_size;
		input_layer_target.nodes[curr_index].update_vectors_forward(input_layer_target.nodes[curr_index-1].h_t,input_layer_target.nodes[curr_index-1].c_t,
			d->current_indices,0,d->d_current_indices,d_ones,
			input_layer_target.nodes[curr_index-1].d_h_t,input_layer_target.nodes[curr_index-1].d_c_t);
		#ifdef CPU_DEBUG
		eigen_check_thrust_ptr(input_layer_target.nodes[curr_index-1].h_t,input_layer_target.nodes[curr_index-1].d_h_t,"decode forward prop target h_t_prev",(dType)0.00000001);
		eigen_check_thrust_ptr(input_layer_target.nodes[curr_index-1].c_t,input_layer_target.nodes[curr_index-1].d_c_t,"decode forward prop target c_t_prev",(dType)0.00000001);
		#endif
		input_layer_target.nodes[curr_index].forward_prop();
		cudaDeviceSynchronize();
		softmax.backprop_prep_GPU(input_layer_target.nodes[curr_index].d_h_t,NULL,NULL,NULL);
		softmax.getDist(input_layer_target.nodes[curr_index].h_t);
		#ifdef CPU_DEBUG
		eigen_check_thrust_ptr(input_layer_target.nodes[curr_index].h_t,input_layer_target.nodes[curr_index].d_h_t,"decode forward prop target h_t",(dType)0.00000001);
		eigen_check_thrust_ptr(softmax.D,softmax.d_D,"decode forward prop target D",(dType)0.00000001);
		eigen_check_thrust_ptr(softmax.b_d,softmax.d_b_d,"decode forward prop target b_d",(dType)0.00000001);
		#endif
		softmax.get_distribution_GPU(softmax.output_vocab_size,softmax.d_outputdist,softmax.d_D,softmax.d_b_d);
	}
	//copy the outputdist from the GPU to the output
	cudaDeviceSynchronize();
	cudaMemcpy(h_outputdist,softmax.d_outputdist,softmax.output_vocab_size*input_layer_target.minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost); //CHANGED
}



template<typename dType>
void neuralMT_model<dType>::beam_decoder(int beam_size,std::string input_file_name,
	std::string input_weight_file_name,int num_lines_in_file,int source_vocab_size,int target_vocab_size,
	int longest_sent,int LSTM_size,dType penalty,std::string decoder_output_file,dType min_decoding_ratio,
	dType max_decoding_ratio,bool scaled,int num_hypotheses,bool print_score) 
{

	//initialize stuff special to decoder
	input_layer_target.temp_swap_vals.resize(LSTM_size,beam_size); //used for changing hidden and cell state columns

	if(target_vocab_size<=20 || beam_size >= target_vocab_size) {
		std::cout << "Beam size reset to one because of small target vocab: " << beam_size << "\n";
		beam_size = 1;
	}

	file_helper_decoder fileh(input_file_name,num_lines_in_file,longest_sent);
	const int start_symbol =0;
	const int end_symbol =1;
	decoder<dType> d(beam_size,target_vocab_size,start_symbol,end_symbol,longest_sent,min_decoding_ratio,
		penalty,decoder_output_file,num_hypotheses,print_score);

	//Need to reinit the model for beam size to make sure the minibatch size is changed to the beam size
	initModel_decoding(LSTM_size,beam_size,source_vocab_size,target_vocab_size,
		longest_sent,false,0,0,0,input_weight_file_name,"",scaled);

	int *d_ones; //all ones size of beamsearch, used in forward passes
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_ones,beam_size*1*sizeof(int)),"GPU memory allocation failed\n");
	ones_mat<<<1,256>>>(d_ones,beam_size);
	cudaDeviceSynchronize();

	dType *h_outputdist = (dType *)malloc(target_vocab_size*beam_size*sizeof(dType));
	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> outputdist;
	outputdist.resize(target_vocab_size,beam_size);

	dType *d_temp_swap_vals;
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_swap_vals,LSTM_size*beam_size*sizeof(dType)),"GPU memory allocation failed\n");

	int *d_input_vocab_indicies_source;
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_input_vocab_indicies_source,longest_sent*sizeof(int)),"GPU memory allocation failed\n");

	//Now since the model has been reinitialized to random values, load the old values
	std::cout << "Starting to load weights from file (This can take some time ...)\n";
	load_weights();

	for(int i=0; i<num_lines_in_file; i++) {
		fileh.read_sentence(); //read sentence from file and put it on the GPU
		//std::cout << "Starting new sentence for decoding\n";
		std::cout << "Current sentence num: " << i+1 << " (out of) " << num_lines_in_file << "\n";
		//run forward prop or the source side
		// std::cout << "----------------minibatch tokens_source-----------------\n";
		// std::cout << "Current index: " << i << "\n";
		// std::cout << "Length of sentence: " << fileh.sentence_length << "\n";
		// std::cout << "Length of sentence: " << fileh.minibatch_tokens_source_input.cols() << "\n";
		// std::cout << fileh.minibatch_tokens_source_input << "\n";

		input_layer_source.prep_GPU_vocab_indices(fileh.h_input_vocab_indicies_source,NULL,
			fileh.words_in_sent,0);

		cudaMemcpy(d_input_vocab_indicies_source,fileh.h_input_vocab_indicies_source,fileh.sentence_length*sizeof(int),cudaMemcpyHostToDevice);
		decoder_forward_prop_source(fileh.minibatch_tokens_source_input,d_input_vocab_indicies_source,d_ones);
		//now run the decoder on the target side
		d.init_decoder();
		int last_index =0;
		for(int j=0; j<std::min((int)(max_decoding_ratio*fileh.sentence_length),longest_sent-2); j++) {
			decoder_forward_prop_target(&fileh,&d,d_ones,j,h_outputdist);
			copy_dist_to_eigen(h_outputdist,outputdist);
			d.expand_hypothesis(outputdist,j);
			input_layer_target.swap_states_decoding(d.new_indicies_changes,j,d_temp_swap_vals);
			last_index=j;
		}
		decoder_forward_prop_target(&fileh,&d,d_ones,last_index+1,h_outputdist);
		d.finish_current_hypotheses(outputdist);
		//d.print_current_hypotheses();
		d.output_k_best_hypotheses(fileh.sentence_length);
	}

	// cudaFree(d_ones);
	// cudaFree(d_input_vocab_indicies_source);
	// cudaFree(d_temp_swap_vals);
	// free(h_outputdist);
}

template<typename dType>
template<typename Derived>
void neuralMT_model<dType>::copy_dist_to_eigen(dType *h_outputdist,const Eigen::MatrixBase<Derived> &outputdist_const) {
	UNCONST(Derived,outputdist_const,outputdist);
	for(int i=0; i<outputdist.rows(); i++) {
		for(int j=0; j<outputdist.cols(); j++) {
			outputdist(i,j) = h_outputdist[IDX2C(i,j,outputdist.rows())];
		}
	}
}

template<typename dType>
void neuralMT_model<dType>::stoicastic_generation(int length,std::string output_file_name) {

	//load weights
	//always load for stoic generation
	load_weights();

	std::cout << "\n--------------Starting stochastic generation-------------\n";

	BZ_CUDA::gen.seed(static_cast<unsigned int>(std::time(0)));
	//file stuff
	std::ofstream ofs;
	ofs.open(output_file_name.c_str());;

	//the start index is zero, so feed it through
	int *h_current_index = (int *)malloc(1 *sizeof(int));
	h_current_index[0] = 0; //this is the start index, always start the generation with this
	int *d_current_index;

	ofs << h_current_index[0] << " ";

	int *h_one = (int *)malloc(1 *sizeof(int));
	h_one[0] = 1;
	int *d_one;

	dType *h_outputdist = (dType *)malloc(softmax.output_vocab_size*1*sizeof(dType));
	dType *d_outputdist;
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist,softmax.output_vocab_size*1*sizeof(dType)),"GPU memory allocation failed\n");

	dType *d_h_t_prev;
	dType *d_c_t_prev;
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_prev,softmax.LSTM_size*1*sizeof(dType)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_c_t_prev,softmax.LSTM_size*1*sizeof(dType)),"GPU memory allocation failed\n");

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_current_index, 1*sizeof(int)),"GPU memory allocation failed\n");
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_one, 1*sizeof(int)),"GPU memory allocation failed\n");
	cudaMemcpy(d_current_index, h_current_index, 1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_one, h_one, 1*sizeof(int), cudaMemcpyHostToDevice);

	LSTM_IH_Node<dType> sg_node(input_layer_target.LSTM_size,1,input_layer_target.input_vocab_size,&input_layer_target);
	//std::cout << "Current char being sent to softmax: " << h_current_index[0] << "\n";
	//now start the generation
	sg_node.update_vectors_forward_GPU(d_current_index,d_one,
		input_layer_target.d_init_hidden_vector,input_layer_target.d_init_cell_vector);

	sg_node.forward_prop();
	cudaDeviceSynchronize();

	softmax.backprop_prep_GPU(sg_node.d_h_t,NULL,NULL,NULL);
	h_current_index[0] = softmax.stoic_generation(h_outputdist,d_outputdist);
	ofs << h_current_index[0] << " ";
	cudaMemcpy(d_current_index, h_current_index, 1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_h_t_prev,sg_node.d_h_t,softmax.LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_c_t_prev,sg_node.d_c_t,softmax.LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);

	for(int i=1; i<length; i++) {

		//std::cout << "Current char being sent to softmax: " << h_current_index[0] << "\n";
		sg_node.update_vectors_forward_GPU(d_current_index,d_one,d_h_t_prev,d_c_t_prev);
		sg_node.forward_prop();
		cudaDeviceSynchronize();

		softmax.backprop_prep_GPU(sg_node.d_h_t,NULL,NULL,NULL);
		h_current_index[0] = softmax.stoic_generation(h_outputdist,d_outputdist);
		ofs << h_current_index[0] << " ";
		if(h_current_index[0]==1) {
			//clear hidden state because end of file
			cudaMemset(sg_node.d_h_t,0,softmax.LSTM_size*1*sizeof(dType));
			cudaMemset(sg_node.d_c_t,0,softmax.LSTM_size*1*sizeof(dType));
		}
		cudaMemcpy(d_current_index, h_current_index, 1*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_h_t_prev,sg_node.d_h_t,softmax.LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_c_t_prev,sg_node.d_c_t,softmax.LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);
	}

	ofs.close();
}



