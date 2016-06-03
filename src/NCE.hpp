
template<typename dType>
void NCE_layer<dType>::init_loss_layer(struct neuralMT_model<precision> *model,global_params &params) {

	this->LSTM_size = params.LSTM_size;
	this->minibatch_size = params.minibatch_size;
	this->output_vocab_size = params.target_vocab_size;
	this->num_negative_samples = params.num_negative_samples;
	this->longest_sent = params.longest_sent;
	this->model = model;
	this->learning_rate = params.learning_rate;
	this->dropout = params.dropout;
	this->dropout_rate = params.dropout_rate;
	this->clip_gradients = params.clip_gradient; //If true then clip gradients
	this->norm_clip = params.norm_clip; //For gradient clipping
	this->share_samples = params.share_samples;

	cudaSetDevice(s_layer_info.device_number);

	dType *h_temp;
	full_matrix_setup(&h_temp,&d_outputdist,output_vocab_size,minibatch_size); //for full softmax during perplexity, be careful since the storage order is reversed
	full_matrix_setup(&h_temp,&d_D,LSTM_size,output_vocab_size);
	full_matrix_setup(&h_temp,&d_b_d,output_vocab_size,1);
	//full_matrix_setup(&h_temp,&d_D_grad,LSTM_size,output_vocab_size);
	if(share_samples) {
		full_matrix_setup(&h_temp,&d_temp_D_grad,LSTM_size,num_negative_samples);
		full_matrix_setup(&h_temp,&d_dot_products,num_negative_samples + minibatch_size,minibatch_size);
	}
	else {
		//full_matrix_setup(&h_temp,&d_temp_D_grad,LSTM_size,num_negative_samples*minibatch_size);
		full_matrix_setup(&h_temp,&d_dot_products,num_negative_samples+1,minibatch_size);
	}
	full_vector_setup(&h_temp,&d_b_d_grad,output_vocab_size);
	full_vector_setup_ones(&h_temp,&d_ones,minibatch_size);

	//for mem saving
	if(share_samples) {
		full_matrix_setup(&h_temp,&d_small_D_grad,(num_negative_samples+minibatch_size)*LSTM_size,longest_sent);
	}
	else {
		full_matrix_setup(&h_temp,&d_small_D_grad,LSTM_size,output_vocab_size); //1.2G
		//reduction space for doing backprop
		full_matrix_setup(&h_temp,&d_reductuction_space,num_negative_samples+1,LSTM_size*minibatch_size);
	}
	
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_reverse_unique_indicies, output_vocab_size*sizeof(int)),"GPU memory allocation failed\n");
	// cudaMemset(d_reverse_unique_indicies,0,1*sizeof(int));
	// CUDA_GET_LAST_ERROR("CHECK 1");

	thrust_d_small_D_grad = thrust::device_pointer_cast(d_small_D_grad);

	//trick to set bias to minus log(vocab size)
	thrust::device_ptr<dType> bias_ptr = thrust::device_pointer_cast(d_b_d);
	for(int i=0; i<output_vocab_size; i++) {
		bias_ptr[i] = -1*std::log(output_vocab_size);
	}

	if(share_samples) {
		h_vocab_indicies = (int *)malloc( (longest_sent * (num_negative_samples + minibatch_size))*sizeof(int));
		h_vocab_indicies_01 = (int *)malloc( (longest_sent * (minibatch_size))*sizeof(int));
		h_sampling_probs = (dType *)malloc( (longest_sent * (num_negative_samples + minibatch_size))*sizeof(dType));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies, (longest_sent * (num_negative_samples + minibatch_size))*sizeof(int)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_sampling_probs, (longest_sent * (num_negative_samples + minibatch_size))*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies_01, (longest_sent * minibatch_size)*sizeof(int)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies_nonneg, (longest_sent * minibatch_size)*sizeof(int)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_OBJ_val_temp, NUM_NCE_THREADS*sizeof(double)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_final_NCE_OBJ, 1*sizeof(double)),"GPU memory allocation failed\n");
		cudaMemset(d_final_NCE_OBJ,0,1*sizeof(double));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_b_d_grad,(num_negative_samples)*sizeof(dType)),"GPU memory allocation failed\n");
		h_unique_indicies = (int *)malloc( (longest_sent * (num_negative_samples + minibatch_size))*sizeof(int));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_unique_indicies,longest_sent*(num_negative_samples+minibatch_size)*sizeof(int)),"GPU memory allocation failed\n");
		thrust_d_b_d_grad = thrust::device_pointer_cast(d_b_d_grad);
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");
	}
	else {
		h_vocab_indicies = (int *)malloc( (longest_sent * ((num_negative_samples+1)*minibatch_size))*sizeof(int));
		h_vocab_indicies_01 = (int *)malloc( (longest_sent * (minibatch_size))*sizeof(int));
		h_sampling_probs = (dType *)malloc( (longest_sent * ((num_negative_samples+1)*minibatch_size))*sizeof(dType));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies, (longest_sent * ((num_negative_samples+1)*minibatch_size))*sizeof(int)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_sampling_probs, (longest_sent * ((num_negative_samples+1)*minibatch_size))*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies_01, (longest_sent * minibatch_size)*sizeof(int)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_indicies_nonneg, (longest_sent * minibatch_size)*sizeof(int)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_OBJ_val_temp, NUM_NCE_THREADS*sizeof(double)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_final_NCE_OBJ, 1*sizeof(double)),"GPU memory allocation failed\n");
		cudaMemset(d_final_NCE_OBJ,0,1*sizeof(double));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_b_d_grad,(num_negative_samples+1)*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		h_unique_indicies = (int *)malloc( (longest_sent * ((num_negative_samples+1)*minibatch_size))*sizeof(int));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_unique_indicies,longest_sent*((num_negative_samples+1)*minibatch_size)*sizeof(int)),"GPU memory allocation failed\n");
		thrust_d_b_d_grad = thrust::device_pointer_cast(d_b_d_grad);
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_result, NORM_THREADS*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_result, 1*sizeof(dType)),"GPU memory allocation failed\n");
	}


	//zero gradients to being
	cudaMemset(d_b_d_grad,0,output_vocab_size*sizeof(dType));
	//cudaMemset(d_D_grad,0,output_vocab_size*LSTM_size*sizeof(dType));
	if(share_samples) {
		cudaMemset(d_small_D_grad,0,(num_negative_samples+minibatch_size)*LSTM_size*longest_sent*sizeof(dType));
	}
	else {
		cudaMemset(d_small_D_grad,0,LSTM_size*output_vocab_size*sizeof(dType));
	}

	//random cuda generator for dropout
	curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT);
	boost::uniform_int<> unif_boost( 1, 1000000 );
	curandSetPseudoRandomGeneratorSeed(rand_gen,BZ_CUDA::curr_seed);
	BZ_CUDA::curr_seed+=7;

	if(BZ_CUDA::print_partition_function) {
		h_partition_vals = (double *)malloc(minibatch_size*sizeof(double));
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_partition_vals, minibatch_size*sizeof(double)),"GPU memory allocation failed\n");
	}

	//get unigram counts from a file *only for target side
	std::vector<long long int> unigram_counts(output_vocab_size);
	std::fill(unigram_counts.begin(),unigram_counts.end(),0);
	get_unigram_counts(unigram_counts,params.train_file_name); //fill the unigram counts for sampling using the alias method
	unigram = multinomial<long long int,double> (unigram_counts);

	//allocate the nodes
	for(int i=0; i<longest_sent; i++) {
		nodes.push_back( NCE_Node<dType>(params.LSTM_size,params.minibatch_size,params.num_negative_samples,i,params.dropout,share_samples) );
	}
}




//this will compute all of the negative samples for a minibatch
template<typename dType>
void NCE_layer<dType>::get_unigram_counts(std::vector<long long int> &unigram_counts,std::string file_name) {

	std::ifstream data_file;
	data_file.open(file_name.c_str(),std::ifstream::in);
	std::string line;
	std::string word;

	while(std::getline(data_file, line)) { //source 1
		std::getline(data_file, line); //source 2
		std::getline(data_file, line); //target 1
		std::getline(data_file, line); //target 2
		std::istringstream iss_input_source(line, std::istringstream::in);

		while( iss_input_source >> word ) {
    		if(std::stoi(word) !=-1) {
    			unigram_counts[std::stoi(word)]++;
    		}
		}
	}

	// std::cout << "Printing unigram counts\n";
	// for(int i=0; i<10; i++) {
	// 	std::cout << unigram_counts[i] << " ";
	// }
	// std::cout << "\n";

	data_file.close();
}




//prep the indicies
template<typename dType>
void NCE_layer<dType>::prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length) {

	if(share_samples) {
		prep_GPU_vocab_indices_shared_samples(h_output_vocab_indicies_target,current_target_length);
	}
	else {
		prep_GPU_vocab_indices_nonshared_samples(h_output_vocab_indicies_target,current_target_length);
	}
}

template<typename dType>
void NCE_layer<dType>::prep_GPU_vocab_indices_shared_samples(int *h_output_vocab_indicies_target,int current_target_length) {
	
	cudaSetDevice(s_layer_info.device_number);
	//this is for gradient checking where you want to have the same samples
	if(model->grad_check_flag) {
		return;
	}

	int curr_index = 0;
	int curr_unique_index = 0;
	std::unordered_map<int,bool> unqiue_check;
	//current target length
	for(int i=0; i<current_target_length; i++) {

		//generate the samples
		for(int j=0; j<num_negative_samples; j++) {
			h_vocab_indicies[curr_index] = unigram.sample(BZ_CUDA::gen);
			h_sampling_probs[curr_index] = std::log(num_negative_samples*unigram.prob(h_vocab_indicies[curr_index]));

			if(unqiue_check.count(h_vocab_indicies[curr_index])==0) {
				unqiue_check[h_vocab_indicies[curr_index]] = true;
				h_unique_indicies[curr_unique_index] = h_vocab_indicies[curr_index];
				curr_unique_index++;
			}

			curr_index++;
		}
		for(int j=0; j<minibatch_size; j++) {
			if(h_output_vocab_indicies_target[j + minibatch_size*i] !=-1) {
				h_vocab_indicies[curr_index] = h_output_vocab_indicies_target[j + minibatch_size*i];
			}
			else {
				h_vocab_indicies[curr_index] = 1; //put index at 1 instead of zero
			}
			h_sampling_probs[curr_index] = std::log(num_negative_samples*unigram.prob(h_vocab_indicies[curr_index]));

			if(unqiue_check.count(h_vocab_indicies[curr_index])==0) {
				unqiue_check[h_vocab_indicies[curr_index]] = true;
				h_unique_indicies[curr_unique_index] = h_vocab_indicies[curr_index];
				curr_unique_index++;
			}

			curr_index++;
		}
	}


	for(int i=0; i<minibatch_size*current_target_length; i++) {
		if(h_output_vocab_indicies_target[i]==-1) {
			h_vocab_indicies_01[i] = 0;
		}
		else {
			h_vocab_indicies_01[i] = 1;
		}
	} 


	curr_num_unique = curr_unique_index;

	cudaMemcpy(d_vocab_indicies, h_vocab_indicies, current_target_length*(minibatch_size + num_negative_samples)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vocab_indicies_01, h_vocab_indicies_01, current_target_length*minibatch_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vocab_indicies_nonneg, h_output_vocab_indicies_target, current_target_length*minibatch_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_unique_indicies, h_unique_indicies, curr_num_unique*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sampling_probs, h_sampling_probs, current_target_length*(minibatch_size + num_negative_samples)*sizeof(dType), cudaMemcpyHostToDevice);


	
	devSynchAll();
	setup_reverse_indicies<<<256,256>>>(d_reverse_unique_indicies,d_unique_indicies,curr_num_unique);
	devSynchAll();
	CUDA_GET_LAST_ERROR("POST GPU 1");
	// std::cout << "\nSAMPLING PROBS\n";
	// for(int i=0; i<current_target_length*(minibatch_size+num_negative_samples); i++) {
	// 	std::cout << h_sampling_probs[i] << " ";
	// }
	// std::cout << "\n";

	// std::cout << "\nIndicies\n";
	// std::cout << "Current target length " << current_target_length << "\n";
	// for(int i=0; i<current_target_length*(minibatch_size); i++) {
	// 	if(i%minibatch_size==0) {
	// 		std::cout << " minibatch:" << i/minibatch_size << " ";
	// 	}
	// 	std::cout <<  h_output_vocab_indicies_target[i] << " ";
	// }
	// std::cout << "\n\n";

	// std::cout << "\nMask Indicies\n";
	// std::cout << "Current target length " << current_target_length << "\n";
	// for(int i=0; i<current_target_length*(minibatch_size); i++) {
	// 	if(i%minibatch_size==0) {
	// 		std::cout << " minibatch:" << i/minibatch_size << " ";
	// 	}
	// 	std::cout <<  h_vocab_indicies_01[i] << " ";
	// }
	// std::cout << "\n\n";
}




template<typename dType>
void NCE_layer<dType>::prep_GPU_vocab_indices_nonshared_samples(int *h_output_vocab_indicies_target,int current_target_length) {
	cudaSetDevice(s_layer_info.device_number);
	//this is for gradient checking where you want to have the same samples
	if(model->grad_check_flag) {
		return;
	}

	int curr_index = 0;
	int curr_unique_index = 0;
	std::unordered_map<int,bool> unqiue_check;
	//current target length
	for(int i=0; i<current_target_length; i++) {

		for(int k=0; k<minibatch_size; k++) {
			//generate the samples
			for(int j=0; j<num_negative_samples; j++) {
				h_vocab_indicies[curr_index] = unigram.sample(BZ_CUDA::gen);
				h_sampling_probs[curr_index] = std::log(num_negative_samples*unigram.prob(h_vocab_indicies[curr_index]));

				if(unqiue_check.count(h_vocab_indicies[curr_index])==0) {
					unqiue_check[h_vocab_indicies[curr_index]] = true;
					h_unique_indicies[curr_unique_index] = h_vocab_indicies[curr_index];
					curr_unique_index++;
				}

				curr_index++;
			}
			for(int j=0; j<1; j++) {
				if(h_output_vocab_indicies_target[j + minibatch_size*i] !=-1) {
					h_vocab_indicies[curr_index] = h_output_vocab_indicies_target[j + minibatch_size*i];
				}
				else {
					h_vocab_indicies[curr_index] = 1; //put index at 1 instead of zero
				}
				h_sampling_probs[curr_index] = std::log(num_negative_samples*unigram.prob(h_vocab_indicies[curr_index]));

				if(unqiue_check.count(h_vocab_indicies[curr_index])==0) {
					unqiue_check[h_vocab_indicies[curr_index]] = true;
					h_unique_indicies[curr_unique_index] = h_vocab_indicies[curr_index];
					curr_unique_index++;
				}

				curr_index++;
			}
		}
	}


	for(int i=0; i<minibatch_size*current_target_length; i++) {
		if(h_output_vocab_indicies_target[i]==-1) {
			h_vocab_indicies_01[i] = 0;
		}
		else {
			h_vocab_indicies_01[i] = 1;
		}
	} 


	curr_num_unique = curr_unique_index;

	cudaMemcpy(d_vocab_indicies, h_vocab_indicies, current_target_length*(minibatch_size * (num_negative_samples+1))*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vocab_indicies_01, h_vocab_indicies_01, current_target_length*minibatch_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vocab_indicies_nonneg, h_output_vocab_indicies_target, current_target_length*minibatch_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_unique_indicies, h_unique_indicies, curr_num_unique*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sampling_probs, h_sampling_probs, current_target_length*(minibatch_size * (num_negative_samples+1))*sizeof(dType), cudaMemcpyHostToDevice);


	
	devSynchAll();
	setup_reverse_indicies<<<256,256>>>(d_reverse_unique_indicies,d_unique_indicies,curr_num_unique);
	devSynchAll();
	CUDA_GET_LAST_ERROR("POST GPU 1");
	// std::cout << "\nSAMPLING PROBS\n";
	// for(int i=0; i<current_target_length*(minibatch_size+num_negative_samples); i++) {
	// 	std::cout << h_sampling_probs[i] << " ";
	// }
	// std::cout << "\n";

	// std::cout << "\nIndicies\n";
	// std::cout << "Current target length " << current_target_length << "\n";
	// for(int i=0; i<current_target_length*(minibatch_size); i++) {
	// 	if(i%minibatch_size==0) {
	// 		std::cout << " minibatch:" << i/minibatch_size << " ";
	// 	}
	// 	std::cout <<  h_output_vocab_indicies_target[i] << " ";
	// }
	// std::cout << "\n\n";

	// std::cout << "\nMask Indicies\n";
	// std::cout << "Current target length " << current_target_length << "\n";
	// for(int i=0; i<current_target_length*(minibatch_size); i++) {
	// 	if(i%minibatch_size==0) {
	// 		std::cout << " minibatch:" << i/minibatch_size << " ";
	// 	}
	// 	std::cout <<  h_vocab_indicies_01[i] << " ";
	// }
	// std::cout << "\n\n";
}


template<typename dType>
void NCE_layer<dType>::backprop_prep_GPU(dType *d_h_t,int step) 
{	
	int index = step/minibatch_size;
	this->d_h_t = d_h_t;
	if(share_samples) {
		d_vocab_indicies_single = d_vocab_indicies + index*(minibatch_size+num_negative_samples);
		d_sampling_probs_single = d_sampling_probs + index*(minibatch_size+num_negative_samples);
	}
	else {
		d_vocab_indicies_single = d_vocab_indicies + index*(minibatch_size*(1+num_negative_samples));
		d_sampling_probs_single = d_sampling_probs + index*(minibatch_size*(1+num_negative_samples));
	}
	d_vocab_indicies_01_single = d_vocab_indicies_01 + step;
	d_vocab_indicies_nonneg_single = d_vocab_indicies_nonneg + step;
}


template<typename dType>
void NCE_layer<dType>::backprop_prep_GPU_mgpu(int step) {

	int index = step/minibatch_size;
	if(share_samples) {
		d_vocab_indicies_single = d_vocab_indicies + index*(minibatch_size+num_negative_samples);
		d_sampling_probs_single = d_sampling_probs + index*(minibatch_size+num_negative_samples);
	}
	else {
		d_vocab_indicies_single = d_vocab_indicies + index*(minibatch_size+(1+num_negative_samples));
		d_sampling_probs_single = d_sampling_probs + index*(minibatch_size+(1+num_negative_samples));
	}
	d_vocab_indicies_01_single = d_vocab_indicies_01 + step;
	d_vocab_indicies_nonneg_single = d_vocab_indicies_nonneg + step;
}	


template<typename dType>
void NCE_layer<dType>::forward_prop(int index) {


	#ifdef REMOVE_STREAM
	devSynchAll();
	#endif

	// devSynchAll();
	//std::cout << "CURRENT NCE INDEX " << index << "\n";
	// print_GPU_Matrix(d_vocab_indicies_01_single,1,minibatch_size);
	// print_GPU_Matrix(d_vocab_indicies_nonneg_single,1,minibatch_size);

	cudaSetDevice(s_layer_info.device_number);

	if(!model->train) {
		cudaMemset(d_final_NCE_OBJ,0,1*sizeof(double));
	}

	//wait for the h_t transfer to start
	if(lower_layer.lower_input) {
		cudaStreamWaitEvent(s_layer_info.s0,lower_layer.input_layer->ih_layer_info.h_t_below_transfer,0);
	}
	else {
		cudaStreamWaitEvent(s_layer_info.s0,lower_layer.hidden_layer->hh_layer_info.h_t_below_transfer,0);
	}

	if(dropout && !model->attent_params.attention_model) {
		curandSetStream(rand_gen, s_layer_info.s0);
		if(!model->grad_check_flag) {
			curandSetStream(rand_gen,s_layer_info.s0);
			curandGenerateUniform_wrapper(nodes[index].d_dropout_mask,LSTM_size*minibatch_size,rand_gen); 
		}
		dropout_kernel<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_dropout_mask,dropout_rate,nodes[index].d_h_t,LSTM_size*minibatch_size);
	}

	// //load in the embeddings
	// load_in_embeddings<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_temp_embeddings,d_D,d_vocab_indicies_single,num_negative_samples+minibatch_size,LSTM_size);
	// CUDA_GET_LAST_ERROR("ERROR IN KERNEL LOAD embeddings");
	
	if(share_samples) {

		//load in the embeddings
		load_in_embeddings<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_temp_embeddings,d_D,d_vocab_indicies_single,num_negative_samples+minibatch_size,LSTM_size);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL LOAD embeddings");
		
		//multiply h_t by the embeddings
		dType alpha = 1;
		dType beta = 0;
		cublasSetStream(s_layer_info.handle,s_layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_T, CUBLAS_OP_N,
			num_negative_samples + minibatch_size, minibatch_size, LSTM_size, &alpha, nodes[index].d_temp_embeddings, LSTM_size,
			nodes[index].d_h_t, LSTM_size, &beta, d_dot_products, num_negative_samples + minibatch_size),"forward h_t with embeddings failed\n");

		//compute -P(true) for all of the elements, also add in the bias at this step
		calc_p_true_kernel<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_p_true,d_dot_products,d_sampling_probs_single,d_b_d,d_vocab_indicies_single,num_negative_samples+minibatch_size,minibatch_size,d_vocab_indicies_01_single);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL CALC P_TRUE");


		//get the objective function for NCE
		objective_val_p1_NCE_kernel<<<NUM_NCE_THREADS,NUM_NCE_THREADS,0,s_layer_info.s0>>>(nodes[index].d_p_true,d_OBJ_val_temp,num_negative_samples,minibatch_size,d_vocab_indicies_01_single);
		objective_val_p2_NCE_kernel<<<1,1,0,s_layer_info.s0>>>(nodes[index].d_p_true,d_final_NCE_OBJ,d_OBJ_val_temp,num_negative_samples,minibatch_size,d_vocab_indicies_01_single);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL NCE end forward_prop");
	}
	else {
		//todo forward prop when not sharing noise samples
		//multiply h_t by the embeddings
		nce_dot_product_SPARSE<<<256,NUM_NCE_THREADS,0,s_layer_info.s0>>>(d_dot_products,d_D,nodes[index].d_h_t,d_vocab_indicies_single,LSTM_size,minibatch_size,num_negative_samples+1,output_vocab_size);

		//compute -P(true) for all of the elements, also add in the bias at this step
		calc_p_true_kernel_nonshare<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_p_true,d_dot_products,d_sampling_probs_single,d_b_d,d_vocab_indicies_single,num_negative_samples,minibatch_size,d_vocab_indicies_01_single);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL CALC P_TRUE");


		//get the objective function for NCE
		objective_val_p1_NCE_kernel<<<NUM_NCE_THREADS,NUM_NCE_THREADS,0,s_layer_info.s0>>>(nodes[index].d_p_true,d_OBJ_val_temp,num_negative_samples,minibatch_size,d_vocab_indicies_01_single);
		objective_val_p2_NCE_kernel_nonshare<<<1,1,0,s_layer_info.s0>>>(nodes[index].d_p_true,d_final_NCE_OBJ,d_OBJ_val_temp,num_negative_samples,minibatch_size,d_vocab_indicies_01_single);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL NCE end forward_prop");
	}

	#ifdef REMOVE_STREAM
	devSynchAll();
	#endif

}

template<typename dType>
void NCE_layer<dType>::back_prop1(int index) {

	#ifdef REMOVE_STREAM
	devSynchAll();
	#endif

	cudaSetDevice(s_layer_info.device_number);

	if(share_samples) {
		dType alpha = 1;
		dType beta = 0;
		cublasSetStream(s_layer_info.handle,s_layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_T,
			LSTM_size, minibatch_size, num_negative_samples, &alpha, nodes[index].d_temp_embeddings, LSTM_size,
			nodes[index].d_p_true, minibatch_size, &beta, nodes[index].d_d_ERRt_ht, LSTM_size),"error h_t negative failed NCE\n");

		//positive part
		error_ht_positive_kernel<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_d_ERRt_ht,nodes[index].d_p_true,nodes[index].d_temp_embeddings + num_negative_samples*LSTM_size,num_negative_samples,LSTM_size,minibatch_size);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL error ht positive NCE\n");
	}
	else {
		//do both the positive and negative parts
		backprop_ht_SPARSE<<<256,NUM_NCE_THREADS,0,s_layer_info.s0>>>(nodes[index].d_d_ERRt_ht,d_D,nodes[index].d_p_true,d_vocab_indicies_single,LSTM_size,minibatch_size,num_negative_samples,d_reductuction_space);
	}

	//zero out d_ERR_ht
	zero_err_ht<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_d_ERRt_ht,d_vocab_indicies_01_single,LSTM_size,minibatch_size);


	//send this to the lower LSTM block
	if(dropout && !model->attent_params.attention_model) {
		dropout_kernel<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_dropout_mask,dropout_rate,nodes[index].d_d_ERRt_ht,LSTM_size*minibatch_size);
	}

	//mgpu stuff
	if(lower_layer.copy_d_Err_ht) {
		if(lower_layer.lower_input) {
			cudaMemcpyAsync(lower_layer.input_layer->nodes[index].d_d_ERRt_ht, nodes[index].d_d_ERRt_ht, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s_layer_info.s0);
		}
		else {
			cudaMemcpyAsync(lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht, nodes[index].d_d_ERRt_ht, LSTM_size*minibatch_size*sizeof(dType), cudaMemcpyDefault,s_layer_info.s0);
		}
	}
	else {
		if(lower_layer.lower_input) {
			lower_layer.input_layer->nodes[index].d_d_ERRt_ht = nodes[index].d_d_ERRt_ht;
		}
		else {
			lower_layer.hidden_layer->nodes[index].d_d_ERRt_ht = nodes[index].d_d_ERRt_ht;
		}
	}

	cudaEventRecord(s_layer_info.d_ERR_ht_done,s_layer_info.s0);

	#ifdef REMOVE_STREAM
	devSynchAll();
	#endif

	CUDA_GET_LAST_ERROR("ERROR IN KERNEL NCE end backprop 1");
}



template<typename dType>
void NCE_layer<dType>::back_prop2(int index) {

	#ifdef REMOVE_STREAM
	devSynchAll();
	#endif

	cudaSetDevice(s_layer_info.device_number);
	
	if(share_samples) {
		dType alpha = 1;
		dType beta = 0;
		//calculare the error with respect to D
		//negative part
		cublasSetStream(s_layer_info.handle,s_layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,
			LSTM_size, num_negative_samples, minibatch_size, &alpha, nodes[index].d_h_t, LSTM_size,
			nodes[index].d_p_true, minibatch_size, &beta, d_temp_D_grad, LSTM_size),"error D negative failed\n");

		//now send them to the d_Grad
		negative_embedding_NCE<<<256,256,0,s_layer_info.s0>>>(d_temp_D_grad,d_small_D_grad,d_vocab_indicies_single,num_negative_samples,LSTM_size,d_reverse_unique_indicies);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL error negative embeddings positive");

		//positive embeddings update
		positive_embedding_NCE<<<256,256,0,s_layer_info.s0>>>(nodes[index].d_h_t,d_small_D_grad,nodes[index].d_p_true+num_negative_samples*minibatch_size,d_vocab_indicies_single+num_negative_samples,LSTM_size,minibatch_size,d_vocab_indicies_01_single,
			d_reverse_unique_indicies);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL error positive embeddings positive");


		//calculate error with respect to b_d
		//negative part
		cublasSetStream(s_layer_info.handle,s_layer_info.s0);
		CUBLAS_ERROR_WRAPPER(cublas_gemv_wrapper(s_layer_info.handle,CUBLAS_OP_T,minibatch_size,num_negative_samples,&alpha,nodes[index].d_p_true,minibatch_size,
			d_ones,1,&beta,d_temp_b_d_grad,1),"cuBLAS normaliztion failed\n");

		//now send them off to b_d_grad
		negative_bias_NCE<<<256,256,0,s_layer_info.s0>>>(d_temp_b_d_grad,d_b_d_grad,d_vocab_indicies_single,num_negative_samples);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL error negative bias NCE");
		//positive part
		positive_bias_NCE<<<256,256,0,s_layer_info.s0>>>(d_b_d_grad,nodes[index].d_p_true,d_vocab_indicies_single,minibatch_size,num_negative_samples,d_vocab_indicies_01_single);
		CUDA_GET_LAST_ERROR("ERROR IN KERNEL positive bias NCE");
	}
	else {

		//calculate the embedding gradients
		embedding_gradient_sparse<<<256,256,0,s_layer_info.s0>>>(d_small_D_grad,nodes[index].d_h_t,nodes[index].d_p_true,d_vocab_indicies_single,LSTM_size,minibatch_size,num_negative_samples);

		//calculate the bias gradients
		bias_gradient_sparse<<<256,256,0,s_layer_info.s0>>>(d_b_d_grad,nodes[index].d_p_true,d_vocab_indicies_single,LSTM_size,minibatch_size,num_negative_samples);
	}

	#ifdef REMOVE_STREAM
	devSynchAll();
	#endif

	CUDA_GET_LAST_ERROR("ERROR IN KERNEL NCE end backprop 2");
}



template<typename dType>
void NCE_layer<dType>::clear_gradients() {

	cudaSetDevice(s_layer_info.device_number);

	cudaMemset(d_b_d_grad,0,output_vocab_size*sizeof(dType));

	//smarter zeroing of the gradients
	// int threads_per_block = 256;
	// int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
	// dim3 kernel(num_block,256,1);
	// zero_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D_grad,d_unique_indicies,LSTM_size,curr_num_unique);

	if(share_samples) {
		cudaMemset(d_small_D_grad,0,(curr_num_unique)*LSTM_size*sizeof(dType));
	}
	else {
		int threads_per_block = 256;
		int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		zero_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_small_D_grad,d_unique_indicies,LSTM_size,curr_num_unique);
	}

	devSynchAll();

	#ifndef NDEBUG
	//zero_check<<<256,256>>>(d_D_grad,LSTM_size*output_vocab_size);
	zero_check<<<256,256>>>(d_b_d_grad,1*output_vocab_size);
	devSynchAll();
	#endif
}



template<typename dType>
void NCE_layer<dType>::update_weights() {

	cudaSetDevice(s_layer_info.device_number);

	//scale the gradients
	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_b_d_grad,thrust_d_b_d_grad + output_vocab_size,unary_op);

	// int threads_per_block = 256;
	// int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
	// dim3 kernel(num_block,256,1);
	// scale_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D_grad,d_unique_indicies,LSTM_size,((dType)1.0)/minibatch_size,curr_num_unique);
	// CUDA_GET_LAST_ERROR();

	if(share_samples) {
		thrust::for_each(thrust_d_small_D_grad,thrust_d_small_D_grad + LSTM_size*curr_num_unique,unary_op);
	}
	else {
		int threads_per_block = 256;
		int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		scale_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_small_D_grad,d_unique_indicies,LSTM_size,((dType)1.0)/minibatch_size,curr_num_unique);
		CUDA_GET_LAST_ERROR();
	}

	/*
		Norm clipping section

		currently per matrix
	*/
	norm_clip_GPU_v2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);

	if(share_samples) {
		norm_clip_GPU_v2(thrust_d_small_D_grad,d_small_D_grad,norm_clip,LSTM_size*curr_num_unique,d_temp_result,d_result);
	}
	else {
		norm_clip_W_GPU_v2(d_temp_result,d_small_D_grad,
			d_unique_indicies,norm_clip,curr_num_unique,LSTM_size); 
	}
	// norm_clip_W_GPU_v2(d_temp_result,d_D_grad,
	// 	d_unique_indicies,norm_clip,curr_num_unique,LSTM_size); 
	// clip_mat_kernel<<<std::min(256,(LSTM_size + 256 - 1)/256),256,0,s_layer_info.s0>>>(d_b_d_grad,BZ_CUDA::ind_norm_clip_thres,output_vocab_size*1);
	// indv_clip_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D_grad,d_unique_indicies,LSTM_size, BZ_CUDA::ind_norm_clip_thres,curr_num_unique); 



	//now add the gradients
	//special D_grad
	// update_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D,d_D_grad,d_unique_indicies,learning_rate,LSTM_size,curr_num_unique);
	// CUDA_GET_LAST_ERROR();
	if(share_samples) {
		update_sparse_grad<<<256,256,0,s_layer_info.s0>>>(d_D,d_small_D_grad,d_unique_indicies,curr_num_unique,learning_rate,LSTM_size);
	}
	else {
		int threads_per_block = 256;
		int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		update_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D,d_small_D_grad,d_unique_indicies,learning_rate,LSTM_size,curr_num_unique);
	}

	gradient_update_mats<<<std::min(256,(output_vocab_size + 256 - 1)/256),256,0,s_layer_info.s0>>>(d_b_d,d_b_d_grad,learning_rate,output_vocab_size*1);

	devSynchAll();
}

template<typename dType>
softmax_layer_gpu_info NCE_layer<dType>::gpu_init(int device_number) {
	s_layer_info.init(device_number);
	return s_layer_info;
}

template<typename dType>
void NCE_layer<dType>::init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer) {
	lower_layer.init_lower_transfer_layer(lower_input,copy_d_Err_ht,input_layer,hidden_layer);
}

template<typename dType>
dType *NCE_layer<dType>::get_ht_ptr(int index) {
	return nodes[index].d_h_t;
}

template<typename dType>
void NCE_layer<dType>::set_ht_ptr(int index,dType *d_h_t) {
	nodes[index].d_h_t = d_h_t;
}

template<typename dType>
cudaEvent_t NCE_layer<dType>::get_ERR_ht_event() {
	return s_layer_info.d_ERR_ht_done;
}

template<typename dType>
dType *NCE_layer<dType>::get_dist_ptr() {
	return d_outputdist;
}

template<typename dType>
void NCE_layer<dType>::update_learning_rate(dType learning_rate) {
	this->learning_rate = learning_rate;
}

template<typename dType>
void NCE_layer<dType>::dump_weights(std::ofstream &output) {
	cudaSetDevice(s_layer_info.device_number);

	write_matrix_GPU_T(d_D,LSTM_size,output_vocab_size,output);
	write_matrix_GPU(d_b_d,output_vocab_size,1,output);
}

template<typename dType>
void NCE_layer<dType>::load_weights(std::ifstream &input) {
	cudaSetDevice(s_layer_info.device_number);

	read_matrix_GPU_T(d_D,LSTM_size,output_vocab_size,input);
	read_matrix_GPU(d_b_d,output_vocab_size,1,input);
}

template<typename dType>
void NCE_layer<dType>::calculate_global_norm() {

	cudaSetDevice(s_layer_info.device_number);

	//scale the gradients
	scale_functor unary_op(minibatch_size);
	thrust::for_each(thrust_d_b_d_grad,thrust_d_b_d_grad + output_vocab_size,unary_op);

	// int threads_per_block = 256;
	// int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
	// dim3 kernel(num_block,256,1);
	// scale_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D_grad,d_unique_indicies,LSTM_size,((dType)1.0)/minibatch_size,curr_num_unique);
	// CUDA_GET_LAST_ERROR();
	if(share_samples) {
		thrust::for_each(thrust_d_small_D_grad,thrust_d_small_D_grad + LSTM_size*curr_num_unique,unary_op);
	}
	else {
		int threads_per_block = 256;
		int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		scale_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_small_D_grad,d_unique_indicies,LSTM_size,((dType)1.0)/minibatch_size,curr_num_unique);
		CUDA_GET_LAST_ERROR();
	}

	//norm_clip_GPU_v2_p1(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);

	devSynchAll();

}

//for weight clipping
template<typename dType>
__global__
void clip_weights_kernel(dType *d_mat,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat[i] = (dType)fmaxf(-0.5f,fminf((dType)d_mat[i],0.5f));
	}
}

template<typename dType>
void NCE_layer<dType>::update_global_params() {

	cudaSetDevice(s_layer_info.device_number);

	dType alpha = learning_rate;
	dType beta = 1;

	// norm_clip_GPU_v2_p2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);

	// norm_clip_W_GPU_v2_p2(d_temp_result,d_D_grad,
	// 	d_unique_indicies,norm_clip,curr_num_unique,LSTM_size); 

	norm_clip_GPU_v2(thrust_d_b_d_grad,d_b_d_grad,norm_clip,output_vocab_size*1,d_temp_result,d_result);

	// if(BZ_CUDA::print_norms) {
	// 	HPC_output << "----------------------- PRINTING GRAD NORM FOR NCE b_d -----------------------\n";
	// 	HPC_output << BZ_CUDA::recent_sum << "\n";
	// }

	// norm_clip_W_GPU_v2(d_temp_result,d_D_grad,
	// 	d_unique_indicies,norm_clip,curr_num_unique,LSTM_size); 
	if(share_samples) {
		norm_clip_GPU_v2(thrust_d_small_D_grad,d_small_D_grad,norm_clip,LSTM_size*curr_num_unique,d_temp_result,d_result);
	}
	else {
		norm_clip_W_GPU_v2(d_temp_result,d_small_D_grad,
			d_unique_indicies,norm_clip,curr_num_unique,LSTM_size); 
	}

	// if(BZ_CUDA::print_norms) {
	// 	HPC_output << "----------------------- PRINTING GRAD NORM FOR NCE b_D -----------------------\n";
	// 	HPC_output << BZ_CUDA::recent_sum << "\n";
	// }

	devSynchAll();
	
	// int threads_per_block = 256;
	// int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
	// dim3 kernel(num_block,256,1);
	// update_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D,d_D_grad,d_unique_indicies,learning_rate,LSTM_size,curr_num_unique);
	// CUDA_GET_LAST_ERROR();

	if(share_samples) {
		update_sparse_grad<<<256,256,0,s_layer_info.s0>>>(d_D,d_small_D_grad,d_unique_indicies,curr_num_unique,learning_rate,LSTM_size);
	}
	else {
		int threads_per_block = 256;
		int num_block = (LSTM_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		update_W_gradient<<<kernel,threads_per_block,0,s_layer_info.s0>>>(d_D,d_small_D_grad,d_unique_indicies,learning_rate,LSTM_size,curr_num_unique);
		CUDA_GET_LAST_ERROR();
	}

	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_geam_wrapper(s_layer_info.handle, CUBLAS_OP_N, CUBLAS_OP_N,output_vocab_size, 1, &alpha, d_b_d_grad, output_vocab_size, &beta, 
		d_b_d, output_vocab_size, d_b_d, output_vocab_size),"CUBLAS addition update parameter failed\n");
	

	//clip_weights_kernel<<<256,256,0,s_layer_info.s0>>>(d_D,LSTM_size*output_vocab_size);
	devSynchAll();
}


template<typename dType>
void NCE_layer<dType>::check_all_gradients(dType epsilon) {

	cudaSetDevice(s_layer_info.device_number);

	std::cout << "--------------------GRADIENT CHECKING FOR NCE LAYER GPU-------------------------\n";
	std::cout << "GRADIENT CHECKING FOR D\n";
	//check_gradient_GPU(epsilon,d_D,d_D_grad,LSTM_size,output_vocab_size);
	if(share_samples) {
		check_gradient_GPU_SPARSE(epsilon,d_D,d_small_D_grad,LSTM_size,h_unique_indicies,curr_num_unique);
	}
	else {
		check_gradient_GPU(epsilon,d_D,d_small_D_grad,LSTM_size,output_vocab_size);
	}
	cudaSetDevice(s_layer_info.device_number);
		
	std::cout << "GRADIENT CHECKING FOR b_d\n";
	check_gradient_GPU(epsilon,d_b_d,d_b_d_grad,output_vocab_size,1);

	cudaSetDevice(s_layer_info.device_number);
}

template<typename dType>
double NCE_layer<dType>::compute_loss_GPU(int index) {

	cudaSetDevice(s_layer_info.device_number);

	cudaDeviceSynchronize();
	if(model->grad_check_flag) {
		forward_prop(index);
	}
	else {
		get_perplexity(nodes[index].d_h_t);
	}
	cudaSetDevice(s_layer_info.device_number);

	cudaDeviceSynchronize();
	double loss = 0;

	if(model->grad_check_flag) {
		loss = get_train_perplexity();
	}
	else {
		thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_vocab_indicies_nonneg_single);
		thrust::device_ptr<int> d_ptr_01 = thrust::device_pointer_cast(d_vocab_indicies_01_single);
		thrust::device_ptr<dType> d_ptr_sm = thrust::device_pointer_cast(d_outputdist);
		for(int i=0; i < minibatch_size; i++) {
			if(d_ptr_01[i]==1) {
				//loss+=std::log((double)d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)]);
				loss+=d_ptr_sm[IDX2C(d_ptr[i],i,output_vocab_size)];
			}
		}
	}

	return loss;
}

template<typename dType>
void NCE_layer<dType>::get_perplexity(dType *d_h_t) 
{	

	devSynchAll();
	//multiply the D matrix with the hidden state matrix
	dType alpha = 1;
	dType beta = 0;
	cublasSetStream(s_layer_info.handle,s_layer_info.s0);
	CUBLAS_ERROR_WRAPPER(cublas_gemm_wrapper(s_layer_info.handle, CUBLAS_OP_T, CUBLAS_OP_N,
	 output_vocab_size, minibatch_size, LSTM_size, &alpha, d_D, LSTM_size,
	  d_h_t, LSTM_size, &beta, d_outputdist, output_vocab_size),"get_distribution cuBLAS call failed\n");


	//add the bias vector to the matrix
	int threads_per_block = 128;
	int num_block = (output_vocab_size + threads_per_block-1)/threads_per_block;
	dim3 kernel_dim(minibatch_size,num_block,1);
	matrix_bias_kernel<<< kernel_dim,threads_per_block,0,s_layer_info.s0 >>>(output_vocab_size,d_outputdist,d_b_d,d_outputdist);
	CUDA_GET_LAST_ERROR("perplexity bias");

	//std::cout << "OVERFLOW KERNEL\n";
	outputdist_perplexity_kernel<<<minibatch_size,SOFTMAX_THREADS,0,s_layer_info.s0>>>(d_outputdist, d_outputdist, output_vocab_size,BZ_CUDA::print_partition_function,d_partition_vals);
	CUDA_GET_LAST_ERROR("Perplexity Kernel");

	if(BZ_CUDA::print_partition_function) {
		devSynchAll();
		cudaMemcpy(h_partition_vals,d_partition_vals,minibatch_size*sizeof(double),cudaMemcpyDeviceToHost);
		for(int i=0; i<minibatch_size; i++) {
			BZ_CUDA::full_partition_vals.push_back(h_partition_vals[i]);
		}
	}

	cudaDeviceSynchronize();
}

template<typename dType>
double NCE_layer<dType>::get_train_perplexity() {
	devSynchAll();
	cudaSetDevice(s_layer_info.device_number);
	double tmp_perp;
	cudaMemcpy(&tmp_perp,d_final_NCE_OBJ,1*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemset(d_final_NCE_OBJ,0,1*sizeof(double));
	return tmp_perp;
}

template<typename dType>
void NCE_layer<dType>::get_distribution_GPU_decoder_wrapper() {
	BZ_CUDA::logger << "ERROR: SHOULD NOT BE HERE IN NCE CLASS DURING DECODING\n";
	exit (EXIT_FAILURE);
}

template<typename dType>
void NCE_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {
	cudaSetDevice(s_layer_info.device_number);
	cudaDeviceSynchronize();
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
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
void NCE_layer<dType>::check_gradient_GPU_SPARSE(dType epsilon,dType *d_mat,dType *d_grad,int LSTM_size,int *h_unique_indicies,int curr_num_unique) {
	cudaSetDevice(s_layer_info.device_number);
	cudaDeviceSynchronize();
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<curr_num_unique; i++) {
		for(int j=0; j<LSTM_size; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= epsilon;
			loss = model->getError(true);
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaSetDevice(s_layer_info.device_number);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(j,h_unique_indicies[i],LSTM_size)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon)) << "     my gradient: " << d_thrust_grad[IDX2C(j,i,LSTM_size)] << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(j,i,LSTM_size)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(j,i,LSTM_size)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}


