template<typename dType>
decoder_model_wrapper<dType>::decoder_model_wrapper(int gpu_num,int beam_size,
	std::string weights_file_name,int longest_sent,bool dump_LSTM,std::string LSTM_stream_dump_name,global_params &params) {

	this->gpu_num = gpu_num;
	this->weights_file_name = weights_file_name;
	this->beam_size = beam_size;

	//now switch to the current GPU
	cudaSetDevice(gpu_num);

	//get model parameters from the model file
	extract_model_info(weights_file_name);

	//allocate d_ones
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_ones,beam_size*1*sizeof(int)),"GPU memory allocation failed\n");
	ones_mat<<<1,256>>>(d_ones,beam_size);
	cudaDeviceSynchronize();

	//allocate the output distribution on the CPU
	h_outputdist = (dType *)malloc(target_vocab_size*beam_size*sizeof(dType));
	outputdist.resize(target_vocab_size,beam_size);

	//allocate the swap values
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_swap_vals,LSTM_size*beam_size*sizeof(dType)),"GPU memory allocation failed\n");


	//allocate the input vocab indicies
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_input_vocab_indicies_source,longest_sent*sizeof(int)),"GPU memory allocation failed\n");


	//allocate the current indicies
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_current_indicies,beam_size*sizeof(int)),"GPU memory allocation failed\n");

	model = new neuralMT_model<dType>();
	//initialize the model
	model->initModel_decoding(LSTM_size,beam_size,source_vocab_size,target_vocab_size,
		num_layers,weights_file_name,gpu_num,dump_LSTM,LSTM_stream_dump_name,params);

	//initialize additional stuff for model
	model->init_prev_states(num_layers,LSTM_size,beam_size,gpu_num);

	//load in weights for the model
	model->load_weights();

	cudaSetDevice(0);
}



template<typename dType>
void decoder_model_wrapper<dType>::extract_model_info(std::string weights_file_name) {

	std::ifstream weight_stream;
	weight_stream.open(weights_file_name.c_str());

	std::vector<std::string> model_params;
	std::string temp_line; //for getline
	std::string temp_word;

	std::getline(weight_stream,temp_line);
	std::istringstream my_ss(temp_line, std::istringstream::in);
	while(my_ss >> temp_word) {
		model_params.push_back(temp_word);
	}

	if(model_params.size()!=4) {
		std::cout << "ERROR: model format is not correct for decoding with file: " << weights_file_name << "\n";
		std::cout << "You cannot do kbest with a sequence model. Only sequence to sequence models\n";
	}

	num_layers = std::stoi(model_params[0]);
	LSTM_size = std::stoi(model_params[1]);
	target_vocab_size = std::stoi(model_params[2]);
	source_vocab_size = std::stoi(model_params[3]);

	std::cout << "------------------------Model stats for filename: " << weights_file_name << "---------------------------\n";
	std::cout << "Number of LSTM layers: " << num_layers << "\n";
	std::cout << "LSTM size: " << LSTM_size << "\n";
	std::cout << "Target vocabulary size: " << target_vocab_size << "\n";
	std::cout << "Source vocabulary size: " << source_vocab_size << "\n";
	std::cout << "----------------------------------------------------------------------------------------------------------\n\n";

	weight_stream.close();
}

template<typename dType>
void decoder_model_wrapper<dType>::memcpy_vocab_indicies(int *h_input_vocab_indicies_source,int source_length) {

	this->source_length = source_length;

	cudaSetDevice(gpu_num);

	CUDA_ERROR_WRAPPER(cudaMemcpy(d_input_vocab_indicies_source,h_input_vocab_indicies_source,source_length*sizeof(int),cudaMemcpyHostToDevice),"GPU memory allocation failed\n");

	cudaSetDevice(0);
}



template<typename dType>
void decoder_model_wrapper<dType>::forward_prop_source() {
	model->forward_prop_source(d_input_vocab_indicies_source,d_ones,source_length,LSTM_size);
	model->source_length = source_length;
}


template<typename dType>
void decoder_model_wrapper<dType>::forward_prop_target(int curr_index,int *h_current_indicies) {

	//copy indicies from decoder to this model
	cudaSetDevice(gpu_num);
	CUDA_ERROR_WRAPPER(cudaMemcpy(d_current_indicies,h_current_indicies,beam_size*sizeof(int),cudaMemcpyHostToDevice),"GPU memory allocation failed\n");

	//run forward target prop on model
	model->forward_prop_target(curr_index,d_current_indicies,d_ones,LSTM_size,beam_size);
	cudaSetDevice(gpu_num);
	//copy the outputdist to CPU
	cudaDeviceSynchronize();
	CUDA_ERROR_WRAPPER(cudaMemcpy(h_outputdist,model->softmax->get_dist_ptr(),target_vocab_size*beam_size*sizeof(dType),cudaMemcpyDeviceToHost),"GPU memory allocation failed\n");

	//copy the outputdist to eigen from CPU
	copy_dist_to_eigen(h_outputdist,outputdist);
}



template<typename dType>
template<typename Derived>
void decoder_model_wrapper<dType>::copy_dist_to_eigen(dType *h_outputdist,const Eigen::MatrixBase<Derived> &outputdist_const) {
	UNCONST(Derived,outputdist_const,outputdist);
	for(int i=0; i < outputdist.rows(); i++) {
		for(int j=0; j < outputdist.cols(); j++) {
			outputdist(i,j) = h_outputdist[IDX2C(i,j,outputdist.rows())];
		}
	}
}


template<typename dType>
template<typename Derived>
void decoder_model_wrapper<dType>::swap_decoding_states(const Eigen::MatrixBase<Derived> &indicies,int index) {

	model->swap_decoding_states(indicies,index,d_temp_swap_vals);
}

template<typename dType>
void decoder_model_wrapper<dType>::target_copy_prev_states() {
	model->target_copy_prev_states(LSTM_size,beam_size);
}



