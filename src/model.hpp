//Model.hpp file that contains implementations for the model class
template<typename dType>
void neuralMT_model<dType>::initModel(int LSTM_size,int minibatch_size,int source_vocab_size,int target_vocab_size,
 int longest_sent,bool debug,dType learning_rate,bool clip_gradients,dType norm_clip,
 std::string input_weight_file,std::string output_weight_file,bool scaled,bool train_perplexity,
 bool truncated_softmax,int shortlist_size,int sampled_size,bool LM,int num_layers,std::vector<int> gpu_indicies,bool dropout,
 dType dropout_rate,attention_params attent_params,global_params &params) 
{

	if(gpu_indicies.size()!=0) {
		if(gpu_indicies.size()!= num_layers+1) {
			std::cout << "ERROR: multi gpu indicies you specified are invalid. There must be one index for each layer, plus one index for the softmax\n";
			exit (EXIT_FAILURE);
		}
	}

	int temp_max_gpu=0;
	for(int i=0; i<gpu_indicies.size(); i++) {
		if(gpu_indicies[i]>temp_max_gpu) {
			temp_max_gpu = gpu_indicies[i];
		}
	}


	// //for outputting alignments
	// if(attent_params.dump_alignments) {
	// 	output_alignments.open(attent_params.tmp_alignment_file.c_str());
	// }


	std::vector<int> final_gpu_indicies; // what layer is on what GPU
	if(gpu_indicies.size()!=0){
		final_gpu_indicies = gpu_indicies;
	}
	else {
		for(int i=0; i<num_layers+1; i++) {
			final_gpu_indicies.push_back(0);
		}
	}

	std::unordered_map<int,layer_gpu_info> layer_lookups; //get the layer lookups for each GPU
	for(int i=0; i<final_gpu_indicies.size()-1; i++) {
		if(layer_lookups.count(final_gpu_indicies[i])==0) {
			layer_gpu_info temp_layer_info;
			temp_layer_info.init(final_gpu_indicies[i]);
			layer_lookups[final_gpu_indicies[i]] = temp_layer_info;
		}
	}

	//before initializing the layers, get the number of layers, number of GPU's and allocate them accordingly
	//softmax = new softmax_layer<dType>();
	//softmax->s_layer_info.init(final_gpu_indicies.back());
	//s_layer_info = softmax->gpu_init(final_gpu_indicies.back());
	//s_layer_info = softmax->s_layer_info;//remove soon
	input_layer_source.ih_layer_info = layer_lookups[final_gpu_indicies[0]];
	input_layer_target.ih_layer_info = layer_lookups[final_gpu_indicies[0]];

	//Initialize the softmax layer
	//softmax = new softmax_layer<dType>();
	if(params.softmax) {
		softmax = new softmax_layer<dType>();
	}
	else if(params.NCE) {
		softmax = new NCE_layer<dType>();
	}

	s_layer_info = softmax->gpu_init(final_gpu_indicies.back());
	softmax->init_loss_layer(this,params);


	//Now print gpu info
	std::cout << "----------Memory status after softmax layer was initialized-----------\n";
	print_GPU_Info();

	if(!LM) {
		//Initialize the input layer
		input_layer_source.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout, dropout_rate);
	}

	input_layer_target.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102,dropout, dropout_rate);

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
	this->attent_params = attent_params;

	std::cout << "--------Memory status after Layer 1 was initialized--------\n";
	print_GPU_Info();

	//do this to be sure addresses stay the same
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
		}
		target_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
	}

	//now initialize hidden layers
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
			source_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,dropout, dropout_rate);
		}
		target_hidden_layers[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
		target_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,dropout, dropout_rate);
		std::cout << "--------Memory status after Layer " << i+1 << " was initialized--------\n";
		print_GPU_Info();
	}


	//initialize the attention layer on top layer, by this time all the other layers have been initialized
	if(attent_params.attention_model) {
		if(num_layers==1) {
			input_layer_target.init_attention(final_gpu_indicies[0],attent_params.D,attent_params.feed_input,this);
			for(int i=0; i<longest_sent; i++) {
				input_layer_target.attent_layer->nodes[i].d_h_t = input_layer_target.nodes[i].d_h_t;
				input_layer_target.attent_layer->nodes[i].d_d_ERRt_ht_tild = input_layer_target.nodes[i].d_d_ERRt_ht;
				input_layer_target.attent_layer->nodes[i].d_indicies_mask = &input_layer_target.nodes[i].d_input_vocab_indices_01;
			}

			if(attent_params.feed_input) {
				input_layer_target.init_feed_input(NULL);
				input_layer_target.ih_layer_info.attention_forward = input_layer_target.attent_layer->layer_info.forward_prop_done;
				input_layer_target.attent_layer->layer_info.error_htild_below= input_layer_target.ih_layer_info.error_htild_below;
			}
		}
		else {
			target_hidden_layers[num_layers-2].init_attention(final_gpu_indicies[num_layers-1],attent_params.D,attent_params.feed_input,this);
			for(int i=0; i<longest_sent; i++) {
				target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_h_t = target_hidden_layers[num_layers-2].nodes[i].d_h_t;
				target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_d_ERRt_ht_tild  = target_hidden_layers[num_layers-2].nodes[i].d_d_ERRt_ht;
				target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_indicies_mask  = &target_hidden_layers[num_layers-2].nodes[i].d_input_vocab_indices_01;
			}

			if(attent_params.feed_input) {
				input_layer_target.init_feed_input(&target_hidden_layers[num_layers-2]);
				input_layer_target.ih_layer_info.attention_forward = target_hidden_layers[num_layers-2].attent_layer->layer_info.forward_prop_done;
				target_hidden_layers[num_layers-2].attent_layer->layer_info.error_htild_below = input_layer_target.ih_layer_info.error_htild_below;
			}
		}

		std::cout << "--------Memory status after Attention Layer was initialized--------\n";
		print_GPU_Info();
	}


	if(num_layers==1) {
		if(final_gpu_indicies[0]==final_gpu_indicies[1] && !dropout && !attent_params.attention_model) {
			if(!LM) {
				input_layer_source.upper_layer.init_upper_transfer_layer(true,false,true,softmax,NULL);
			}
			input_layer_target.upper_layer.init_upper_transfer_layer(true,false,false,softmax,NULL);
			softmax->init_lower_transfer_layer(true,false,&input_layer_target,NULL);
		}
		else {
			if(!LM) {
				input_layer_source.upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
			}
			input_layer_target.upper_layer.init_upper_transfer_layer(true,true,false,softmax,NULL);
			softmax->init_lower_transfer_layer(true,true,&input_layer_target,NULL);
		}
	}
	else {
		if(final_gpu_indicies[0]==final_gpu_indicies[1] && !dropout && !attent_params.attention_model) {
			if(!LM) {
				input_layer_source.upper_layer.init_upper_transfer_layer(false,false,true,NULL,&source_hidden_layers[0]);
			}
			input_layer_target.upper_layer.init_upper_transfer_layer(false,false,false,NULL,&target_hidden_layers[0]);
		}
		else {
			if(!LM) {
				input_layer_source.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]);
			}
			input_layer_target.upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[0]);
		}

		for(int i=0; i<target_hidden_layers.size(); i++) {

			//lower transfer stuff
			if(i==0) {
				if(final_gpu_indicies[0]==final_gpu_indicies[1] && !dropout && !attent_params.attention_model) {
					if(!LM) {
						source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,false,&input_layer_source,NULL);
					}
					target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,false,&input_layer_target,NULL);
				}
				else {
					if(!LM) {
						source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source,NULL);
					}
					target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_target,NULL);
				}
			}
			else {
				if(final_gpu_indicies[i]==final_gpu_indicies[i+1] && !dropout && !attent_params.attention_model) {
					if(!LM) {
						source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,false,NULL,&source_hidden_layers[i-1]);
					}
					target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,false,NULL,&target_hidden_layers[i-1]);
				}
				else {
					if(!LM) {
						source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers[i-1]);
					}
					target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i-1]);
				}
			}

			//upper transfer stuff
			if(i==target_hidden_layers.size()-1) {
				if(final_gpu_indicies[i+1]==final_gpu_indicies[i+2] && !dropout && !attent_params.attention_model) {
					if(!LM) {
						source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,false,true,softmax,NULL);
					}
					target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,false,false,softmax,NULL);
					softmax->init_lower_transfer_layer(false,false,NULL,&target_hidden_layers[i]);
				}
				else {
					if(!LM) {
						source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
					}
					target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,false,softmax,NULL);
					softmax->init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i]);
				}
			}
			else {
				if(final_gpu_indicies[i+1]==final_gpu_indicies[i+2] && !dropout && !attent_params.attention_model) {
					if(!LM) {
						source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,false,true,NULL,&source_hidden_layers[i+1]);
					}
					target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,false,false,NULL,&target_hidden_layers[i+1]);
				}
				else {
					if(!LM) {
						source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[i+1]);
					}
					target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[i+1]);
				}
			}
		}
	}
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


//called when doing ensemble decoding
template<typename dType>
void neuralMT_model<dType>::init_prev_states(int num_layers, int LSTM_size,int minibatch_size, int device_number) {

	cudaSetDevice(device_number);
	for(int i=0; i<num_layers; i++) {
		previous_source_states.push_back( prev_source_state<dType>(LSTM_size) );
		previous_target_states.push_back( prev_target_state<dType>(LSTM_size,minibatch_size) );
	}
	cudaSetDevice(0);
}

//for this we need to initialze the source minibatch size to one
template<typename dType>
void neuralMT_model<dType>::initModel_decoding(int LSTM_size,int beam_size,int source_vocab_size,int target_vocab_size,
	int num_layers,std::string input_weight_file,int gpu_num, bool dump_LSTM,std::string LSTM_stream_dump_file,global_params &params) {

	//before initializing the layers, get the number of layers, number of GPU's and allocate them accordingly
	//softmax->s_layer_info.init(gpu_num);
	s_layer_info = softmax->gpu_init(gpu_num);
	//s_layer_info = softmax->s_layer_info;//remove soon
	input_layer_source.ih_layer_info.init(gpu_num);
	input_layer_target.ih_layer_info = input_layer_source.ih_layer_info;

	//for initializing the model for decoding
	const int longest_sent =1;
	const int minibatch_size = 1;
	const bool debug = false;
	const dType learning_rate=0;
	const bool clip_gradients = false;
	const dType norm_clip = 0;
	const bool LM = false;
	const bool truncated_softmax = false;
	const int trunc_size=0;
	const bool softmax_scaled = true;
	const bool train_perplexity = false;
	const std::string output_weight_file = "NULL";
	const bool dropout_rate = 0;

	//Initialize the softmax layer
	softmax->init_loss_layer(this,params);

	//Now print gpu info
	std::cout << "----------Memory status after softmax layer was initialized-----------\n";
	print_GPU_Info();

	if(!LM) {
		//Initialize the input layer
		input_layer_source.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,false,0);
	}

	input_layer_target.init_Input_To_Hidden_Layer(LSTM_size,beam_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102,false,0);

	//Initialize the hidden layer
	// hidden_layer.init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,input_vocab_size,output_vocab_size,
 // 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this);

	this->input_weight_file = input_weight_file;
	this->output_weight_file = output_weight_file;
	this->truncated_softmax = false;
	this->LM = false;

	//stuff for printing LSTM traces
	this->dump_LSTM = dump_LSTM;
	if(dump_LSTM) {
		LSTM_stream_dump.open(LSTM_stream_dump_file);
	}

	std::cout << "--------Memory status after Layer 1 was initialized--------\n";
	print_GPU_Info();

	//do this to be sure addresses stay the same
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
		}
		target_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
	}

	//now initialize hidden layers
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers[i-1].hh_layer_info = input_layer_target.ih_layer_info;
			source_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,false,0);
		}
		target_hidden_layers[i-1].hh_layer_info = input_layer_target.ih_layer_info;
		target_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,beam_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,false,0);
		std::cout << "--------Memory status after Layer " << i+1 << " was initialized--------\n";
		print_GPU_Info();
	}

	
	//now the layer info

	if(num_layers==1) {
		input_layer_source.upper_layer.init_upper_transfer_layer(true,false,true,softmax,NULL);
		input_layer_target.upper_layer.init_upper_transfer_layer(true,false,false,softmax,NULL);
		softmax->init_lower_transfer_layer(true,false,&input_layer_target,NULL);
	}
	else {
		input_layer_source.upper_layer.init_upper_transfer_layer(false,false,true,NULL,&source_hidden_layers[0]);
		input_layer_target.upper_layer.init_upper_transfer_layer(false,false,false,NULL,&target_hidden_layers[0]);

		for(int i=0; i<target_hidden_layers.size(); i++) {

			//lower transfer stuff
			if(i==0) {
				source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,false,&input_layer_source,NULL);
				target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,false,&input_layer_target,NULL);
			}
			else {
				source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,false,NULL,&source_hidden_layers[i-1]);
				target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,false,NULL,&target_hidden_layers[i-1]);
			}

			//upper transfer stuff
			if(i==target_hidden_layers.size()-1) {
				source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,false,true,softmax,NULL);
				target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,false,false,softmax,NULL);
				softmax->init_lower_transfer_layer(false,false,NULL,&target_hidden_layers[i]);
			}
			else {
				source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,false,true,NULL,&source_hidden_layers[i+1]);
				target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,false,false,NULL,&target_hidden_layers[i+1]);
			}
		}
	}
}

template<typename dType>
void neuralMT_model<dType>::init_GPUs() {

}


template<typename dType>
template<typename Derived>
void neuralMT_model<dType>::compute_gradients(const Eigen::MatrixBase<Derived> &source_input_minibatch_const,
	const Eigen::MatrixBase<Derived> &source_output_minibatch_const,const Eigen::MatrixBase<Derived> &target_input_minibatch_const,
	const Eigen::MatrixBase<Derived> &target_output_minibatch_const,int *h_input_vocab_indicies_source,
	int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_output_vocab_indicies_target,
	int current_source_length,int current_target_length,int *h_input_vocab_indicies_source_Wgrad,int *h_input_vocab_indicies_target_Wgrad,
	int len_source_Wgrad,int len_target_Wgrad,int *h_sampled_indices,int len_unique_words_trunc_softmax,int *h_batch_info) 
{
	//Clear the gradients before forward/backward pass
	//eventually clear gradients at the end
	//clear_gradients();

	train = true;

	source_length = current_source_length;

	//std::cout << "Starting compute gradients\n";

	//Send the CPU vocab input data to the GPU layers
	//For the input layer, 2 host vectors must be transfered since need special preprocessing for W gradient
	if(!LM){
		input_layer_source.prep_GPU_vocab_indices(h_input_vocab_indicies_source,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].prep_GPU_vocab_indices(h_input_vocab_indicies_source,current_source_length);
		}
	}
	input_layer_target.prep_GPU_vocab_indices(h_input_vocab_indicies_target,h_input_vocab_indicies_target_Wgrad,current_target_length,len_target_Wgrad);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].prep_GPU_vocab_indices(h_input_vocab_indicies_target,current_target_length);
	}

	// devSynchAll();
	// CUDA_GET_LAST_ERROR("DONT FAIL HERE 111");
	softmax->prep_GPU_vocab_indices(h_output_vocab_indicies_target,current_target_length);
	// if(truncated_softmax) {
	// 	softmax->prep_trunc(h_sampled_indices,len_unique_words_trunc_softmax);
	// }

	// devSynchAll();
	// CUDA_GET_LAST_ERROR("DONT FAIL HERE  222");

	if(attent_params.attention_model) {
		if(target_hidden_layers.size()==0) {
			input_layer_target.attent_layer->prep_minibatch_info(h_batch_info);
		}
		else {
			target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(h_batch_info);
		}
	}
	devSynchAll();


	// std::cout << "Printing out source sentence. In format of [minibatch][minibatch] ...\n";
	// for(int i=0; i<current_source_length; i++) {
	// 	for(int j=0; j<input_layer_target.minibatch_size; j++) {
	// 		std::cout << h_input_vocab_indicies_source[input_layer_target.minibatch_size*i + j] << " ";
	// 	}
	// 	std::cout << "\n";
	// }
	// std::cout << "\n";

	// std::cout << "Printing out batch info. Sentence lengths first\n";
	// for(int i=0; i< input_layer_target.minibatch_size; i++) {
	// 	std::cout << h_batch_info[i] << " ";
	// }
	// std::cout << "\n";

	// std::cout << "Printing out batch info. Now offsets\n";
	// for(int i=0; i< input_layer_target.minibatch_size; i++) {
	// 	std::cout << h_batch_info[i+input_layer_target.minibatch_size] << " ";
	// }
	// std::cout << "\n";

	//std::cout << "Starting source foward:\n";
	//std::cout << "Source Index: 0\n"; 
	if(!LM) {
		//Do the source side forward pass
		input_layer_source.nodes[0].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full,
			input_layer_source.d_input_vocab_indices_01_full,
			input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
		input_layer_source.nodes[0].forward_prop();

		//mgpu stuff
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers[i].d_input_vocab_indices_01_full,
				source_hidden_layers[i].d_init_hidden_vector,source_hidden_layers[i].d_init_cell_vector);
			source_hidden_layers[i].nodes[0].forward_prop();
		}


		//cudaDeviceSynchronize();
		//for(int i=1; i<source_input_minibatch_const.cols(); i++) {
		for(int i=1; i<current_source_length; i++) {
			//std::cout << "Source Index: " << i << "\n"; 
			int step = i*input_layer_source.minibatch_size;
			input_layer_source.nodes[i].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full+step,
				input_layer_source.d_input_vocab_indices_01_full+step,
				input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t);
			input_layer_source.nodes[i].forward_prop();
			//cudaDeviceSynchronize();


			//mgpu stuff
			for(int j=0; j<source_hidden_layers.size(); j++) {
				source_hidden_layers[j].nodes[i].update_vectors_forward_GPU(source_hidden_layers[j].d_input_vocab_indices_01_full+step,
					source_hidden_layers[j].nodes[i-1].d_h_t,source_hidden_layers[j].nodes[i-1].d_c_t);
				source_hidden_layers[j].nodes[i].forward_prop();
			}
		}
	}
	//devSynchAll();
	//Do the target side forward pass
	//int prev_source_index = source_input_minibatch_const.cols()-1;


	//print off all the forward states on source side
	// devSynchAll();
	// std::cout << "PRINTING SOURCE HIDDEN STATES\n";
	// for(int i=0; i<current_source_length; i++) {
	// 	std::cout << "Index: " << i << "\n";
	// 	if(source_hidden_layers.size()==0) {
	// 		print_GPU_Matrix(input_layer_source.nodes[i].d_h_t,input_layer_source.LSTM_size,input_layer_source.minibatch_size);
	// 	}
	// 	else {
	// 		print_GPU_Matrix(source_hidden_layers[source_hidden_layers.size()-1].nodes[i].d_h_t,input_layer_source.LSTM_size,input_layer_source.minibatch_size);
	// 	}
	// }

	// std::cout << "---------------------- STARTING TARGET FORWARD PROP ------------------------------\n";

	//std::cout << "Forward prop index: " << 0 << "\n";
	if(LM) {
		input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
			input_layer_target.d_input_vocab_indices_01_full,
			input_layer_target.d_init_hidden_vector,input_layer_target.d_init_cell_vector);

		//mgpu stuff
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
				target_hidden_layers[i].d_init_hidden_vector,target_hidden_layers[i].d_init_cell_vector);
		}
	}
	else {
		int prev_source_index = current_source_length-1;
		input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
			input_layer_target.d_input_vocab_indices_01_full,
			input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t);

		//mgpu stuff
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
				source_hidden_layers[i].nodes[prev_source_index].d_h_t,source_hidden_layers[i].nodes[prev_source_index].d_c_t);
		}
	}

	//std::cout << "Index: 0\n";
	input_layer_target.nodes[0].forward_prop();

	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();
	}


	//mgpu stuff
	softmax->backprop_prep_GPU_mgpu(0);
	softmax->forward_prop(0);

	for(int i=1; i<current_target_length; i++) {
		//std::cout << "Forward prop target index: " << i << "\n";
		int step = i*input_layer_target.minibatch_size;
		input_layer_target.nodes[i].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full+step,
			input_layer_target.d_input_vocab_indices_01_full+step,
			input_layer_target.nodes[i-1].d_h_t,input_layer_target.nodes[i-1].d_c_t);
		input_layer_target.nodes[i].forward_prop();


		//mgpu stuff
		for(int j=0; j<target_hidden_layers.size(); j++) {
			//std::cout << "Layer: " << j+1 << "\n";
			target_hidden_layers[j].nodes[i].update_vectors_forward_GPU(
				target_hidden_layers[j].d_input_vocab_indices_01_full+step,
				target_hidden_layers[j].nodes[i-1].d_h_t,target_hidden_layers[j].nodes[i-1].d_c_t);
			target_hidden_layers[j].nodes[i].forward_prop();
		}

		//mgpu stuff
		softmax->backprop_prep_GPU_mgpu(step);
		softmax->forward_prop(i);
	}

	devSynchAll();

	/////////////////////////////////////////backward pass/////////////////////////////////////////////////
	////////////////////////////Do the backward pass for the target first////////////////////////////
	int last_index = current_target_length-1;

	//std::cout << "BACKPROP INDEX: " << last_index << "\n";

	int step = (current_target_length-1)*input_layer_target.minibatch_size;
	softmax->backprop_prep_GPU(input_layer_target.nodes[last_index].d_h_t,step);

	//mgpu stuff
	softmax->backprop_prep_GPU_mgpu(step);
	softmax->back_prop1(current_target_length-1);

	//record these two events to start for the GPU

	//mgpu stuff
	for(int i=target_hidden_layers.size()-1; i>=0; i--) {
		//std::cout << "backward target index: " << i << "\n";
		target_hidden_layers[i].nodes[last_index].backprop_prep_GPU(target_hidden_layers[i].d_init_d_ERRnTOtp1_ht,target_hidden_layers[i].d_init_d_ERRnTOtp1_ct);//,
		target_hidden_layers[i].nodes[last_index].back_prop_GPU(last_index);
	}

	input_layer_target.nodes[last_index].backprop_prep_GPU(input_layer_target.d_init_d_ERRnTOtp1_ht,input_layer_target.d_init_d_ERRnTOtp1_ct);//,
	
	input_layer_target.nodes[last_index].back_prop_GPU();

	for(int i=current_target_length-2; i>=0; i--) {

		step = i*input_layer_target.minibatch_size;

		softmax->backprop_prep_GPU(input_layer_target.nodes[i].d_h_t,step);

		//mgpu stuff
		softmax->backprop_prep_GPU_mgpu(step);
		softmax->back_prop1(i);

		for(int j=target_hidden_layers.size()-1; j>=0; j--) {
			target_hidden_layers[j].nodes[i].backprop_prep_GPU(target_hidden_layers[j].d_d_ERRnTOt_htM1,target_hidden_layers[j].d_d_ERRnTOt_ctM1);//,
			target_hidden_layers[j].nodes[i].back_prop_GPU(i);
		}

		input_layer_target.nodes[i].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,input_layer_target.d_d_ERRnTOt_ctM1);

		input_layer_target.nodes[i].back_prop_GPU();
	}

	///////////////////////////Now do the backward pass for the source///////////////////////

	if(!LM) {
		int prev_source_index = current_source_length-1;

		//mgpu stuff
		int backprop2_index=0;
		softmax->backprop_prep_GPU_mgpu(0);
		softmax->back_prop2(backprop2_index);
		backprop2_index++;


		//mgpu stuff
		for(int i=source_hidden_layers.size()-1; i>=0; i--) {
			source_hidden_layers[i].nodes[prev_source_index].backprop_prep_GPU(target_hidden_layers[i].d_d_ERRnTOt_htM1,target_hidden_layers[i].d_d_ERRnTOt_ctM1);//,
			source_hidden_layers[i].nodes[prev_source_index].back_prop_GPU(prev_source_index);
		}


		input_layer_source.nodes[prev_source_index].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,
		 	input_layer_target.d_d_ERRnTOt_ctM1);//,input_layer_source.d_zeros);

		input_layer_source.nodes[prev_source_index].back_prop_GPU();
		
		for(int i=current_source_length-2; i>=0; i--) {
			//std::cout << "Backward source index " << i << "\n";

			for(int j=source_hidden_layers.size()-1; j>=0; j--) {
				source_hidden_layers[j].nodes[i].backprop_prep_GPU(source_hidden_layers[j].d_d_ERRnTOt_htM1,source_hidden_layers[j].d_d_ERRnTOt_ctM1);//,
				source_hidden_layers[j].nodes[i].back_prop_GPU(i);
			}

			input_layer_source.nodes[i].backprop_prep_GPU(input_layer_source.d_d_ERRnTOt_htM1,input_layer_source.d_d_ERRnTOt_ctM1);

			input_layer_source.nodes[i].back_prop_GPU();

			//mgpu stuff
			if(backprop2_index<current_target_length) {
				int step = backprop2_index * input_layer_target.minibatch_size;
				softmax->backprop_prep_GPU_mgpu(step);
				softmax->back_prop2(backprop2_index);
				backprop2_index++;
			}
		}
		//mgpu stuff
		for(int i=backprop2_index; i<current_target_length; i++) {
			int step = backprop2_index * input_layer_target.minibatch_size;
			softmax->backprop_prep_GPU_mgpu(step);
			softmax->back_prop2(backprop2_index);
			backprop2_index++;
		}


	}
	else {
		//mgpu stuff
		for(int i=0; i<current_target_length; i++) {
			int step = i*input_layer_target.minibatch_size;
			softmax->backprop_prep_GPU_mgpu(step);
			softmax->back_prop2(i);
		}
	}

	//std::cout << "Ending backprop\n";
	if(debug) {
		grad_check_flag = true;
		dType epsilon =(dType)1e-4;
		devSynchAll();
		check_all_gradients(epsilon);
		grad_check_flag = false;
	}

	// //Update the model parameter weights
	update_weights();

	clear_gradients();

	devSynchAll();

	if(train_perplexity_mode) {
		// cudaSetDevice(softmax->s_layer_info.device_number);
		// double tmp_perp;
		// cudaMemcpy(&tmp_perp,softmax->d_train_perplexity,1*sizeof(double),cudaMemcpyDeviceToHost);
		// train_perplexity+=tmp_perp;
		// cudaMemset(softmax->d_train_perplexity,0,1*sizeof(double));
		// cudaSetDevice(0);
		train_perplexity += softmax->get_train_perplexity();
	}

	train = false;
}


template<typename dType>
void neuralMT_model<dType>::dump_alignments(int target_length,int minibatch_size,int *h_input_vocab_indicies_source,int *h_input_vocab_indicies_target) {

	devSynchAll();

	dType *h_p_t;
	int *h_batch_info;
	h_p_t = (dType *)malloc(minibatch_size* sizeof(dType));
	h_batch_info = (int *)malloc(minibatch_size*2 * sizeof(int));

	std::vector<std::vector<int>> output_indicies;
	for(int i=0; i<minibatch_size*2; i++) {
		std::vector<int> temp;
		output_indicies.push_back( temp );
	}


	std::vector<std::string> alignment_nums; //stores in string format 1-3 2-4 4-5, etc..
	for(int i=0; i<minibatch_size; i++) {
		alignment_nums.push_back(" ");
	}
	
	if(target_hidden_layers.size()==0) {
		cudaMemcpy(h_batch_info,input_layer_target.attent_layer->d_batch_info,minibatch_size*2*sizeof(int),cudaMemcpyDeviceToHost);
	}
	else {
		cudaMemcpy(h_batch_info,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->d_batch_info,minibatch_size*2*sizeof(int),cudaMemcpyDeviceToHost);
	}

	for(int i=0; i<target_length; i++) {
		if(target_hidden_layers.size()==0) {
			cudaMemcpy(h_p_t,input_layer_target.attent_layer->nodes[i].d_p_t,minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost);
		}
		else {
			cudaMemcpy(h_p_t,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_p_t,minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost);
		}	
		//std::cout << "Target index: " << i << "  p_t: " << h_p_t[0] << "\n";
		for(int j=0; j<minibatch_size; j++) {
			if( h_input_vocab_indicies_target[ IDX2C(j,i,minibatch_size) ]!=-1) {
				output_indicies[0 + 2*j].push_back( h_input_vocab_indicies_source[ IDX2C(j,(int)h_p_t[j] + h_batch_info[j+minibatch_size],minibatch_size) ] );
				output_indicies[1 + 2*j].push_back( h_input_vocab_indicies_target[ IDX2C(j,i,minibatch_size) ] );
				alignment_nums[j]+= std::to_string((int)h_p_t[j]) + "-" + std::to_string(i) + " ";
			}
		}
	}

	// std::cout << "SOURCE LENGTH: " << file_info->current_source_length << "\n";
	// for(int i=0; i<file_info->current_source_length; i++) {
	// 	std::cout << h_input_vocab_indicies_source[i] << " ";
	// }
	// std::cout << "\n";

	std::cout << "Printing alignments\n";
	for(int i=0; i<minibatch_size;i++) {
		std::cout << alignment_nums[i] << "\n";
	}

	// for(int i=0; i<output_indicies.size(); i++) {
	// 	for(int j=0; j< output_indicies[i].size(); j++) {
	// 		output_alignments << output_indicies[i][j] << " ";
	// 	}
	// 	output_alignments << "\n";
	// }

	free(h_p_t);
	free(h_batch_info);
}



template<typename dType>
void neuralMT_model<dType>::clear_gradients() {
	devSynchAll();
	if(!LM) {
		input_layer_source.clear_gradients(false);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].clear_gradients(false);
		}
	}
	input_layer_target.clear_gradients(false);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].clear_gradients(false);
	}
	softmax->clear_gradients();
	devSynchAll();
}

template<typename dType>
double neuralMT_model<dType>::getError(bool GPU) 
{
	double loss=0;

	source_length = file_info->current_source_length;

	if(!LM) {
		input_layer_source.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_source,file_info->h_input_vocab_indicies_source_Wgrad,
			file_info->current_source_length,file_info->len_source_Wgrad);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_source,file_info->current_source_length);
		}
	}
	input_layer_target.prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_target,file_info->h_input_vocab_indicies_target_Wgrad,
		file_info->current_target_length,file_info->len_target_Wgrad);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].prep_GPU_vocab_indices(file_info->h_input_vocab_indicies_target,file_info->current_target_length);
	}
	softmax->prep_GPU_vocab_indices(file_info->h_output_vocab_indicies_target,file_info->current_target_length);

	if(attent_params.attention_model) {
		if(target_hidden_layers.size()==0) {
			input_layer_target.attent_layer->prep_minibatch_info(file_info->h_batch_info);
		}
		else {
			target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(file_info->h_batch_info);
		}
	}

	devSynchAll();

	if(!LM) {
		input_layer_source.nodes[0].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full,
			input_layer_source.d_input_vocab_indices_01_full,
			input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
		input_layer_source.nodes[0].forward_prop();

		//mgpu stuff
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers[i].d_input_vocab_indices_01_full,
				source_hidden_layers[i].d_init_hidden_vector,source_hidden_layers[i].d_init_cell_vector);
			source_hidden_layers[i].nodes[0].forward_prop();
		}

		//for(int i=1; i<file_info->minibatch_tokens_source_input.cols(); i++) {
		for(int i=1; i<file_info->current_source_length; i++) {
			int step = i*input_layer_source.minibatch_size;
			input_layer_source.nodes[i].update_vectors_forward_GPU(input_layer_source.d_input_vocab_indices_full+step,
				input_layer_source.d_input_vocab_indices_01_full+step,
				input_layer_source.nodes[i-1].d_h_t,input_layer_source.nodes[i-1].d_c_t);
			input_layer_source.nodes[i].forward_prop();
			//cudaDeviceSynchronize();

			//mgpu stuff
			for(int j=0; j<source_hidden_layers.size(); j++) {
				source_hidden_layers[j].nodes[i].update_vectors_forward_GPU(
					source_hidden_layers[j].d_input_vocab_indices_01_full+step,
					source_hidden_layers[j].nodes[i-1].d_h_t,source_hidden_layers[j].nodes[i-1].d_c_t);
				source_hidden_layers[j].nodes[i].forward_prop();
			}
		}
	}


	//std::cout << "----------------STARTING TARGET SIDE FOR GET ERROR----------------\n";
	//Do the target side forward pass
	//int prev_source_index = file_info->minibatch_tokens_source_input.cols()-1;
	if(LM) {
		input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
			input_layer_target.d_input_vocab_indices_01_full,
			input_layer_target.d_init_hidden_vector,input_layer_target.d_init_cell_vector);


		//mgpu stuff
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(
				target_hidden_layers[i].d_input_vocab_indices_01_full,
				target_hidden_layers[i].d_init_hidden_vector,target_hidden_layers[i].d_init_cell_vector);
		}
	}
	else {
		int prev_source_index = file_info->current_source_length-1;
		input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
			input_layer_target.d_input_vocab_indices_01_full,
			input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t);


		//mgpu stuff
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(
				target_hidden_layers[i].d_input_vocab_indices_01_full,
				source_hidden_layers[i].nodes[prev_source_index].d_h_t,source_hidden_layers[i].nodes[prev_source_index].d_c_t);
		}
	}	

	input_layer_target.nodes[0].forward_prop();

	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();
	}

	devSynchAll();
	//note d_h_t can be null for these as all we need is the vocab pointers correct for getting the error
	softmax->backprop_prep_GPU(input_layer_target.nodes[0].d_h_t,0);

	if(GPU) {
		loss += softmax->compute_loss_GPU(0);
	}
	else {
		std::cout << "ERROR CAN ONLY USE GPU\n";
		exit (EXIT_FAILURE);
	}
	devSynchAll();

	//for(int i=1; i<file_info->minibatch_tokens_target_input.cols(); i++) {
	for(int i=1; i<file_info->current_target_length; i++) {
		int step = i*input_layer_target.minibatch_size;

		input_layer_target.nodes[i].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full+step,
			input_layer_target.d_input_vocab_indices_01_full+step,
			input_layer_target.nodes[i-1].d_h_t,input_layer_target.nodes[i-1].d_c_t);

		input_layer_target.nodes[i].forward_prop();

		//mgpu stuff
		for(int j=0; j<target_hidden_layers.size(); j++) {
			target_hidden_layers[j].nodes[i].update_vectors_forward_GPU(
				target_hidden_layers[j].d_input_vocab_indices_01_full+step,
				target_hidden_layers[j].nodes[i-1].d_h_t,target_hidden_layers[j].nodes[i-1].d_c_t);
			target_hidden_layers[j].nodes[i].forward_prop();
		}

		devSynchAll();
		softmax->backprop_prep_GPU(input_layer_target.nodes[i].d_h_t,step);

		if(GPU) {
			loss += softmax->compute_loss_GPU(i);
		}
		else {
			std::cout << "ERROR CAN ONLY USE GPU\n";
			exit (EXIT_FAILURE);
		}
		devSynchAll();
	}

	if(attent_params.dump_alignments) {
		dump_alignments(file_info->current_target_length,input_layer_target.minibatch_size,file_info->h_input_vocab_indicies_source,file_info->h_input_vocab_indicies_target);
	}

	return loss;
}



template<typename dType>
void neuralMT_model<dType>::check_all_gradients(dType epsilon) 
{
	devSynchAll();
	if(!LM) {
		std::cout << "------------------CHECKING GRADIENTS ON SOURCE SIDE------------------------\n";
		input_layer_source.check_all_gradients(epsilon);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].check_all_gradients(epsilon);
		}
	}
	std::cout << "------------------CHECKING GRADIENTS ON TARGET SIDE------------------------\n";
	input_layer_target.check_all_gradients(epsilon);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].check_all_gradients(epsilon);
	}
	softmax->check_all_gradients(epsilon);
	//hidden_layer.check_all_gradients(epsilon,input_minibatch_const,output_minibatch_const);
}


//Update the model parameters
template<typename dType>
void neuralMT_model<dType>::update_weights() {

	devSynchAll();


	if(BZ_CUDA::global_clip_flag) {

		BZ_CUDA::global_norm = 0; //for global gradient clipping

		softmax->calculate_global_norm();
		if(!LM) {
			input_layer_source.calculate_global_norm();
			for(int i=0; i<source_hidden_layers.size(); i++) {
				source_hidden_layers[i].calculate_global_norm();
			}
		}
		input_layer_target.calculate_global_norm();
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].calculate_global_norm();
		}

		devSynchAll();

		BZ_CUDA::global_norm = std::sqrt(BZ_CUDA::global_norm);

		softmax->update_global_params();
		if(!LM) {
			input_layer_source.update_global_params();
			for(int i=0; i<source_hidden_layers.size(); i++) {
				source_hidden_layers[i].update_global_params();
			}
		}
		input_layer_target.update_global_params();
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].update_global_params();
		}

		devSynchAll();
	}
	else {

		softmax->update_weights();
		if(!LM) {
			input_layer_source.update_weights();
			for(int i=0; i<source_hidden_layers.size(); i++) {
				source_hidden_layers[i].update_weights();
			}
		}
		input_layer_target.update_weights();
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].update_weights();
		}
	}

	devSynchAll();
	if(attent_params.attention_model) {
		if(source_hidden_layers.size()==0) {
			input_layer_source.zero_attent_error();
		}
		else {
			source_hidden_layers[source_hidden_layers.size()-1].zero_attent_error();
		}
	}

	devSynchAll();
}

//Update the model parameters
template<typename dType>
void neuralMT_model<dType>::update_weights_OLD() {

	BZ_CUDA::global_norm = 0; //for global gradient clipping
	devSynchAll();
	//first calculate the global gradient sum
	softmax->calculate_global_norm();
	if(!LM) {
		input_layer_source.calculate_global_norm();
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].calculate_global_norm();
		}
	}
	input_layer_target.calculate_global_norm();
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].calculate_global_norm();
	}

	devSynchAll();

	softmax->update_global_params();
	if(!LM) {
		input_layer_source.update_global_params();
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].update_global_params();
		}
	}
	input_layer_target.update_global_params();
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].update_global_params();
	}

	devSynchAll();
	//hidden_layer.update_weights();
}


template<typename dType>
void neuralMT_model<dType>::dump_weights() {
	output.open(output_weight_file.c_str(),std::ios_base::app);

	output.precision(std::numeric_limits<dType>::digits10 + 2);
	//output.flush();
	if(!LM) {
		input_layer_source.dump_weights(output);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].dump_weights(output);
		}
	}
	//output.flush();
	input_layer_target.dump_weights(output);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].dump_weights(output);
	}
	//output.flush();
	softmax->dump_weights(output);
	//output.flush();
	output.close();
	//output.flush();
}

template<typename dType>
void neuralMT_model<dType>::dump_best_model(std::string best_model_name,std::string const_model) {

	if(boost::filesystem::exists(best_model_name)) {
		boost::filesystem::remove(best_model_name);
	}

	std::ifstream const_model_stream;
	const_model_stream.open(const_model.c_str());

	std::ofstream best_model_stream;
	best_model_stream.open(best_model_name.c_str());

	best_model_stream.precision(std::numeric_limits<dType>::digits10 + 2);


	//now create the new model file
	std::string str;
	std::string word;
	std::getline(const_model_stream, str);
	best_model_stream << str << "\n";
	std::getline(const_model_stream, str);
	best_model_stream << str << "\n";
	while(std::getline(const_model_stream, str)) {
		best_model_stream << str << "\n";
		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
		}
	}

	if(!LM) {
		while(std::getline(const_model_stream, str)) {
			best_model_stream << str << "\n";
			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
					break; //done with source mapping
			}
		}
	}

	if(!LM) {
		input_layer_source.dump_weights(best_model_stream);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].dump_weights(best_model_stream);
		}
	}
	input_layer_target.dump_weights(best_model_stream);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].dump_weights(best_model_stream);
	}
	
	softmax->dump_weights(best_model_stream);
	best_model_stream.flush();
	best_model_stream.close();
	const_model_stream.close();
	
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
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].load_weights(input);
		}
	}
	input_layer_target.load_weights(input);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].load_weights(input);
	}
	//input.sync();
	softmax->load_weights(input);

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
	for(int i=0; i<source_hidden_layers.size(); i++) {
		source_hidden_layers[i].learning_rate = new_learning_rate;
	}
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].learning_rate = new_learning_rate;
	}

	softmax->update_learning_rate(new_learning_rate);
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
		// load_shortlist_D<<<256,256>>>(softmax->d_subset_D,softmax->d_D,softmax->LSTM_size,softmax->trunc_size,softmax->output_vocab_size,softmax->shortlist_size);
		// load_shortlist_D<<<256,256>>>(softmax->d_subset_b_d,softmax->d_b_d,1,softmax->trunc_size,softmax->output_vocab_size,softmax->shortlist_size);
		// devSynchAll();
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

	if(BZ_CUDA::print_partition_function) {
		BZ_CUDA::print_partition_stats();
	}

	return perplexity_GPU;
}



template<typename dType>
void neuralMT_model<dType>::stoicastic_generation(int length,std::string output_file_name,double temperature) {

	// //load weights
	// //always load for stoic generation
	// load_weights();

	// std::cout << "\n--------------Starting stochastic generation-------------\n";

	// BZ_CUDA::gen.seed(static_cast<unsigned int>(std::time(0)));
	// //file stuff
	// std::ofstream ofs;
	// ofs.open(output_file_name.c_str());;

	// //the start index is zero, so feed it through
	// int *h_current_index = (int *)malloc(1 *sizeof(int));
	// h_current_index[0] = 0; //this is the start index, always start the generation with this
	// int *d_current_index;

	// //ofs << h_current_index[0] << " ";

	// int *h_one = (int *)malloc(1 *sizeof(int));
	// h_one[0] = 1;
	// int *d_one;

	// dType *h_outputdist = (dType *)malloc(softmax->output_vocab_size*1*sizeof(dType));
	// dType *d_outputdist;
	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist,softmax->output_vocab_size*1*sizeof(dType)),"GPU memory allocation failed\n");

	// dType *d_h_t_prev;
	// dType *d_c_t_prev;
	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_prev,softmax->LSTM_size*1*sizeof(dType)),"GPU memory allocation failed\n");
	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_c_t_prev,softmax->LSTM_size*1*sizeof(dType)),"GPU memory allocation failed\n");

	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_current_index, 1*sizeof(int)),"GPU memory allocation failed\n");
	// CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_one, 1*sizeof(int)),"GPU memory allocation failed\n");
	// cudaMemcpy(d_current_index, h_current_index, 1*sizeof(int), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_one, h_one, 1*sizeof(int), cudaMemcpyHostToDevice);

	// LSTM_IH_Node<dType> sg_node(input_layer_target.LSTM_size,1,input_layer_target.input_vocab_size,&input_layer_target,0,NULL,NULL,0);
	// //std::cout << "Current char being sent to softmax: " << h_current_index[0] << "\n";
	// //now start the generation
	// sg_node.update_vectors_forward_GPU(d_current_index,d_one,
	// 	input_layer_target.d_init_hidden_vector,input_layer_target.d_init_cell_vector);

	// sg_node.forward_prop();
	// devSynchAll();

	// softmax->backprop_prep_GPU(sg_node.d_h_t,NULL,NULL,NULL);
	// h_current_index[0] = softmax->stoic_generation(h_outputdist,d_outputdist,temperature);
	// ofs << h_current_index[0] << " ";
	// cudaMemcpy(d_current_index, h_current_index, 1*sizeof(int), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_h_t_prev,sg_node.d_h_t,softmax->LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);
	// cudaMemcpy(d_c_t_prev,sg_node.d_c_t,softmax->LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);

	// int num_sent = 0;
	// while(num_sent<length) {

	// 	//std::cout << "Current char being sent to softmax: " << h_current_index[0] << "\n";
	// 	sg_node.update_vectors_forward_GPU(d_current_index,d_one,d_h_t_prev,d_c_t_prev);
	// 	sg_node.forward_prop();
	// 	devSynchAll();

	// 	softmax->backprop_prep_GPU(sg_node.d_h_t,NULL,NULL,NULL);
	// 	h_current_index[0] = softmax->stoic_generation(h_outputdist,d_outputdist,temperature);
	// 	if(h_current_index[0]==1) {
	// 		//clear hidden state because end of file
	// 		cudaMemset(sg_node.d_h_t,0,softmax->LSTM_size*1*sizeof(dType));
	// 		cudaMemset(sg_node.d_c_t,0,softmax->LSTM_size*1*sizeof(dType));
	// 		h_current_index[0] = 0;
	// 		ofs << "\n";
	// 		num_sent++;
	// 	}
	// 	else {
	// 		ofs << h_current_index[0] << " ";
	// 	}
	// 	cudaMemcpy(d_current_index, h_current_index, 1*sizeof(int), cudaMemcpyHostToDevice);
	// 	cudaMemcpy(d_h_t_prev,sg_node.d_h_t,softmax->LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);
	// 	cudaMemcpy(d_c_t_prev,sg_node.d_c_t,softmax->LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice);
	// }

	// ofs.close();
}


//for ensembles
template<typename dType>
void neuralMT_model<dType>::forward_prop_source(int *d_input_vocab_indicies_source,int *d_ones,int source_length,int LSTM_size) 
{
	devSynchAll();
	cudaSetDevice(input_layer_target.ih_layer_info.device_number);
	input_layer_source.nodes[0].update_vectors_forward_GPU(d_input_vocab_indicies_source,d_ones,
		input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
	input_layer_source.nodes[0].forward_prop();

	if(dump_LSTM) {
		input_layer_source.nodes[0].dump_LSTM(LSTM_stream_dump,"-----------Layer 1 Source word: " + std::to_string(0) + "-----------\n");
	}

	for(int i=0; i < source_hidden_layers.size(); i++) {
		source_hidden_layers[i].nodes[0].update_vectors_forward_GPU(d_ones,
			source_hidden_layers[i].d_init_hidden_vector,source_hidden_layers[i].d_init_cell_vector);
		source_hidden_layers[i].nodes[0].forward_prop();

		if(dump_LSTM) {
			source_hidden_layers[i].nodes[0].dump_LSTM(LSTM_stream_dump,"-----------Layer" + std::to_string(i+2) + "Source word: " + std::to_string(0) + "-----------\n");
		}
	}

	devSynchAll();
	for(int i = 1; i < source_length; i++) {
		int step = i;
		//copy the h_t and c_t to the previous hidden state of node 0
		CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[0].d_h_t_prev,input_layer_source.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[0].d_c_t_prev,input_layer_source.nodes[0].d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed\n");
		for(int j = 0; j < source_hidden_layers.size(); j++) {
			CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[j+1].d_h_t_prev,source_hidden_layers[j].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed\n");
			CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[j+1].d_c_t_prev,source_hidden_layers[j].nodes[0].d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed\n");
		}

		input_layer_source.nodes[0].update_vectors_forward_GPU(d_input_vocab_indicies_source+i,d_ones,
			previous_source_states[0].d_h_t_prev,previous_source_states[0].d_c_t_prev);
		input_layer_source.nodes[0].forward_prop();

		if(dump_LSTM) {
			input_layer_source.nodes[0].dump_LSTM(LSTM_stream_dump,"-----------Layer 1 Source word: " + std::to_string(i) + "-----------\n");
		}

		for(int j=0; j < source_hidden_layers.size(); j++) {
			source_hidden_layers[j].nodes[0].update_vectors_forward_GPU(d_ones,
				previous_source_states[j+1].d_h_t_prev,previous_source_states[j+1].d_c_t_prev);
			source_hidden_layers[j].nodes[0].forward_prop();

			if(dump_LSTM) {
				source_hidden_layers[j].nodes[0].dump_LSTM(LSTM_stream_dump,"-----------Layer" + std::to_string(j+2) + "Source word: " + std::to_string(i) + "-----------\n");
			}

		}
		devSynchAll();
	}
	devSynchAll();
}


template<typename dType>
void neuralMT_model<dType>::forward_prop_target(int curr_index,int *d_current_indicies,int *d_ones,int LSTM_size, int beam_size) {

	int num_layers = 1+ target_hidden_layers.size();
	cudaSetDevice(input_layer_target.ih_layer_info.device_number);
	if(curr_index==0) {
		input_layer_target.transfer_decoding_states_GPU(input_layer_source.nodes[0].d_h_t,input_layer_source.nodes[0].d_c_t);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			target_hidden_layers[i].transfer_decoding_states_GPU(source_hidden_layers[i].nodes[0].d_h_t,source_hidden_layers[i].nodes[0].d_c_t);
		}
		input_layer_target.nodes[0].update_vectors_forward_decoder(d_current_indicies,d_ones);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			target_hidden_layers[i].nodes[0].update_vectors_forward_decoder(d_ones);
		}
	}
	else {
		input_layer_target.nodes[0].update_vectors_forward_GPU(d_current_indicies,d_ones,previous_target_states[0].d_h_t_prev,previous_target_states[0].d_c_t_prev);
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(d_ones,previous_target_states[i+1].d_h_t_prev,previous_target_states[i+1].d_c_t_prev);
		}
	}
	//now run forward prop on all the layers
	input_layer_target.nodes[0].forward_prop();

	if(dump_LSTM) {
		input_layer_target.nodes[0].dump_LSTM(LSTM_stream_dump,"-----------Layer 1 Target word: " + std::to_string(curr_index) + "-----------\n");
	}

	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();

		if(dump_LSTM) {
			target_hidden_layers[i].nodes[0].dump_LSTM(LSTM_stream_dump,"-----------Layer " + std::to_string(i+1) + " Target word: " + std::to_string(curr_index) + "-----------\n");
		}

	}
	devSynchAll();
	if(num_layers==1) {
		softmax->backprop_prep_GPU(input_layer_target.nodes[0].d_h_t,0);
	} 
	else {
		softmax->backprop_prep_GPU(target_hidden_layers[target_hidden_layers.size()-1].nodes[0].d_h_t,0);
	}
	//softmax->get_distribution_GPU(softmax->output_vocab_size,softmax->d_outputdist,softmax->d_D,softmax->d_b_d,softmax->d_h_t); //non-trunc
	softmax->get_distribution_GPU_decoder_wrapper();
	devSynchAll();
	//copy the h_t and c_t to the previous hidden state of node 0
	cudaSetDevice(input_layer_target.ih_layer_info.device_number);

	// CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[0].d_h_t_prev,input_layer_target.nodes[0].d_h_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed1\n");
	// CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[0].d_c_t_prev,input_layer_target.nodes[0].d_c_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed2\n");
	// for(int j = 0; j < target_hidden_layers.size(); j++) {
	// 	CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[j+1].d_h_t_prev,target_hidden_layers[j].nodes[0].d_h_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed3\n");
	// 	CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[j+1].d_c_t_prev,target_hidden_layers[j].nodes[0].d_c_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed4\n");
	// }




}

template<typename dType>
void neuralMT_model<dType>::target_copy_prev_states(int LSTM_size, int beam_size) {
	cudaSetDevice(input_layer_target.ih_layer_info.device_number);
	CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[0].d_h_t_prev,input_layer_target.nodes[0].d_h_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed1\n");
	CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[0].d_c_t_prev,input_layer_target.nodes[0].d_c_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed2\n");
	for(int j = 0; j < target_hidden_layers.size(); j++) {
		CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[j+1].d_h_t_prev,target_hidden_layers[j].nodes[0].d_h_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed3\n");
		CUDA_ERROR_WRAPPER(cudaMemcpy(previous_target_states[j+1].d_c_t_prev,target_hidden_layers[j].nodes[0].d_c_t,LSTM_size*beam_size*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memcpy failed4\n");
	}
}


template<typename dType>
template<typename Derived>
void neuralMT_model<dType>::swap_decoding_states(const Eigen::MatrixBase<Derived> &indicies,int index,dType *d_temp_swap_vals) {

	input_layer_target.swap_states_decoding(indicies,index,d_temp_swap_vals);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].swap_states_decoding(indicies,index,d_temp_swap_vals);
	}
}






