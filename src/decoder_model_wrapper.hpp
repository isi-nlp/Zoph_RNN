template<typename dType>
decoder_model_wrapper<dType>::decoder_model_wrapper(int gpu_num,int beam_size,
	std::string main_weight_file,std::string multi_src_weight_file,std::string main_integerized_file,
	std::string multi_src_integerized_file,int longest_sent,global_params &params) 
{

	this->gpu_num = gpu_num;
	this->beam_size = beam_size;
	this->longest_sent = params.longest_sent;
	BZ_CUDA::logger << "Beam size for decoding: " << beam_size << "\n";

	this->main_weight_file = main_weight_file;
	this->multi_src_weight_file = multi_src_weight_file;
	this->main_integerized_file = main_integerized_file;
	this->multi_src_integerized_file = multi_src_integerized_file;

	//now switch to the current GPU
	cudaSetDevice(gpu_num);

	//get model parameters from the model file
	BZ_CUDA::logger << "Main weight file name: " << main_weight_file << "\n";
	extract_model_info(main_weight_file);

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

	if(char_cnn) {
		extract_char_info(params.char_params.longest_word,params.char_params.num_unique_chars_source,
    	      params.char_params.num_unique_chars_target,params.source_vocab_size,params.target_vocab_size,
    	      params.char_params.char_mapping_file,params.char_params.word_mapping_file);
		this->longest_word = params.char_params.longest_word;
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_char_vocab_indicies_source,longest_sent*longest_word*sizeof(int)),"GPU memory allocation failed\n");

		h_new_char_indicies = (int *)malloc( beam_size*longest_word*sizeof(int) );
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_new_char_indicies,beam_size*longest_word*sizeof(int)),"GPU memory allocation failed\n");

		extract_charCNN(word_to_char_map,"char_word_mapping.txt.brz");

		main_integerized_file = params.char_params.word_test_file;
	}

	//now initialize the file input
	//ERROR FIX THE INITIALIZATION
	fileh = new file_helper_decoder(main_integerized_file,num_lines_in_file,params.longest_sent,params.char_params,params.char_params.char_test_file);

	//allocate other file helper if multi-source
	if(multi_src_integerized_file != "NULL") {
		//multi_source = true;
		fileh_multi_src = new file_helper_decoder(multi_src_integerized_file,num_lines_in_file,params.longest_sent,params.char_params,params.char_params.char_test_file);
	}

	if(multi_source) {
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_input_vocab_indicies_source_bi,longest_sent*sizeof(int)),"GPU memory allocation failed\n");
	}


	//allocate the current indicies
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_current_indicies,beam_size*sizeof(int)),"GPU memory allocation failed\n");

	model = new neuralMT_model<dType>();
	//initialize the model
	model->initModel_decoding(LSTM_size,beam_size,source_vocab_size,target_vocab_size,
		num_layers,main_weight_file,gpu_num,params,attention_model,
		feed_input,multi_source,combine_LSTM,char_cnn);

	//initialize additional stuff for model
	model->init_prev_states(num_layers,LSTM_size,beam_size,gpu_num,multi_source);

	//load in weights for the model
	model->load_weights();
}



template<typename dType>
void decoder_model_wrapper<dType>::extract_model_info(std::string weights_file_name) {

	BZ_CUDA::logger << "-------------------------- EXTRACT MODEL INFO ------------------------\n";
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

	if(model_params.size()!=9) {
		BZ_CUDA::logger << "ERROR: model format is not correct for decoding with file: " << weights_file_name << "\n";
	}

	num_layers = std::stoi(model_params[0]);
	LSTM_size = std::stoi(model_params[1]);
	target_vocab_size = std::stoi(model_params[2]);
	source_vocab_size = std::stoi(model_params[3]);
	attention_model = std::stoi(model_params[4]);
	feed_input = std::stoi(model_params[5]);
	multi_source = std::stoi(model_params[6]);
	combine_LSTM = std::stoi(model_params[7]);
	char_cnn = std::stoi(model_params[8]);

	BZ_CUDA::logger << "------------------------Model stats for filename: " << weights_file_name << "---------------------------\n";
	BZ_CUDA::logger << "Number of LSTM layers: " << num_layers << "\n";
	BZ_CUDA::logger << "LSTM size: " << LSTM_size << "\n";
	BZ_CUDA::logger << "Target vocabulary size: " << target_vocab_size << "\n";
	BZ_CUDA::logger << "Source vocabulary size: " << source_vocab_size << "\n";
	if(attention_model) {
		BZ_CUDA::logger << "Attention model\n";
	}
	if(feed_input) {
		BZ_CUDA::logger << "Feed Input\n";
	}

	if(multi_source) {
		BZ_CUDA::logger << "Multi Source\n";
	}

	if(combine_LSTM) {
		BZ_CUDA::logger << "Tree Combine LSTM\n";
	}
	if(char_cnn) {
		BZ_CUDA::logger << "Char CNN\n";
	}
	BZ_CUDA::logger << "----------------------------------------------------------------------------------------------------------\n\n";

	weight_stream.close();
}

template<typename dType>
void decoder_model_wrapper<dType>::memcpy_vocab_indicies() {


	fileh->read_sentence();
	this->source_length = fileh->sentence_length;
	if(multi_source) {
		fileh_multi_src->read_sentence();
		this->source_length_bi = fileh_multi_src->sentence_length;
	}
	else {
		this->source_length_bi = 0;
	}

//	BZ_CUDA::logger << "source sentence length: " << source_length << "\n";
//	BZ_CUDA::logger << "source sentence multi-source length: " << source_length_bi << "\n";

	cudaSetDevice(gpu_num);

	CUDA_ERROR_WRAPPER(cudaMemcpy(d_input_vocab_indicies_source,fileh->h_input_vocab_indicies_source,source_length*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 1\n");

	if(multi_source) {
		CUDA_ERROR_WRAPPER(cudaMemcpy(d_input_vocab_indicies_source_bi,fileh_multi_src->h_input_vocab_indicies_source,source_length_bi*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 1\n");
	}

	if(char_cnn) {
		CUDA_ERROR_WRAPPER(cudaMemcpy(d_char_vocab_indicies_source,fileh->fhc->h_char_vocab_indicies_source,source_length*longest_word*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 1\n");
	
		// for(int i=0; i<source_length; i++) {
		// 	for(int j=0; j<longest_word; j++) {
		// 		std::cout << fileh->fhc->h_char_vocab_indicies_source[j+ i*longest_word] << " ";
		// 	}
		// 	std::cout << "\n";
		// }
		// std::cout << "\n";
	}

	if(attention_model) {
		for(int i=0; i<beam_size; i++) {
			CUDA_ERROR_WRAPPER(cudaMemcpy(model->decoder_att_layer.d_batch_info+i,fileh->h_batch_info,1*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 2\n");
			CUDA_ERROR_WRAPPER(cudaMemcpy(model->decoder_att_layer.d_batch_info+beam_size+i,fileh->h_batch_info+1,1*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 2\n");

			if(multi_source) {
				CUDA_ERROR_WRAPPER(cudaMemcpy(model->decoder_att_layer.d_batch_info_v2+i,fileh_multi_src->h_batch_info,1*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 2\n");
				CUDA_ERROR_WRAPPER(cudaMemcpy(model->decoder_att_layer.d_batch_info_v2+beam_size+i,fileh_multi_src->h_batch_info+1,1*sizeof(int),cudaMemcpyHostToDevice),"decoder memcpy_vocab_indicies 2\n");
			}
		}
	}
}	

template<typename dType>
void decoder_model_wrapper<dType>::forward_prop_source() {
	model->forward_prop_source(d_input_vocab_indicies_source,d_input_vocab_indicies_source_bi,d_ones,source_length,source_length_bi,LSTM_size,
		d_char_vocab_indicies_source);
	model->source_length = source_length;
}


template<typename dType>
void decoder_model_wrapper<dType>::forward_prop_target(int curr_index,int *h_current_indicies) {

//	BZ_CUDA::logger << "Current target index: " << curr_index << "\n";
	//copy indicies from decoder to this model
	cudaSetDevice(gpu_num);
	CUDA_ERROR_WRAPPER(cudaMemcpy(d_current_indicies,h_current_indicies,beam_size*sizeof(int),cudaMemcpyHostToDevice),"forward prop target decoder 1\n");

	if(char_cnn) {
		//now create the new indicies here
		create_char_vocab(h_current_indicies,beam_size,longest_word,h_new_char_indicies,word_to_char_map);
		CUDA_ERROR_WRAPPER(cudaMemcpy(d_new_char_indicies,h_new_char_indicies,beam_size*longest_word*sizeof(int),cudaMemcpyHostToDevice),"forward prop target decoder 1\n");
		// for(int i=0; i<beam_size; i++) {
		// 	for(int j=0; j<longest_word; j++) {
		// 		std::cout << h_new_char_indicies[j + i*longest_word] << " ";
		// 	}
		// 	std::cout << "\n";
		// }
		// std::cout << "\n";
	}

	//run forward target prop on model
	model->forward_prop_target(curr_index,d_current_indicies,d_ones,LSTM_size,beam_size,d_new_char_indicies);
	cudaSetDevice(gpu_num);
	//copy the outputdist to CPU
	cudaDeviceSynchronize();
	CUDA_GET_LAST_ERROR("ERROR ABOVE!!");
	CUDA_ERROR_WRAPPER(cudaMemcpy(h_outputdist,model->softmax->get_dist_ptr(),target_vocab_size*beam_size*sizeof(dType),cudaMemcpyDeviceToHost),"forward prop target decoder 2\n");
	//copy the outputdist to eigen from CPU
	copy_dist_to_eigen(h_outputdist,outputdist);

	if (BZ_CUDA::unk_replacement) {
		viterbi_alignments_ind = BZ_CUDA::viterbi_alignments;
		viterbi_alignments_scores = BZ_CUDA::alignment_scores;
	}
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



