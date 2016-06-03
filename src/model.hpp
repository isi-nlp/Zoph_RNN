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
			BZ_CUDA::logger << "ERROR: multi gpu indicies you specified are invalid. There must be one index for each layer, plus one index for the softmax\n";
			exit (EXIT_FAILURE);
		}
	}

	this->char_cnn = params.char_params.char_cnn;
	this->char_params = params.char_params;

	int temp_max_gpu=0;
	for(int i=0; i<gpu_indicies.size(); i++) {
		if(gpu_indicies[i]>temp_max_gpu) {
			temp_max_gpu = gpu_indicies[i];
		}
	}


	//for outputting alignments
	if(attent_params.dump_alignments) {
		output_alignments.open(attent_params.alignment_file.c_str());
	}


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

	//for birdirectional part
	this->bi_dir = params.bi_dir_params.bi_dir;

	this->multi_attention = params.multi_src_params.multi_attention;

	this->multi_attention_v2 = params.multi_src_params.multi_attention_v2;

	//for multilanguage LM
	this->multi_source = params.multi_src_params.multi_source;
	this->multisource_file = params.multi_src_params.int_file_name_test;
	if(multi_source && params.train) {
		src_fh.init_file_helper_source(params.multi_src_params.int_file_name,params.minibatch_size,params.longest_sent,params.source_vocab_size);
	}

	//before initializing the layers, get the number of layers, number of GPU's and allocate them accordingly
	//softmax = new softmax_layer<dType>();
	//softmax->s_layer_info.init(final_gpu_indicies.back());
	//s_layer_info = softmax->gpu_init(final_gpu_indicies.back());
	//s_layer_info = softmax->s_layer_info;//remove soon
	input_layer_source.ih_layer_info = layer_lookups[final_gpu_indicies[0]];
	if(bi_dir || multi_source) {
		input_layer_source_bi.ih_layer_info = layer_lookups[final_gpu_indicies[0]];
	}
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
	BZ_CUDA::logger << "----------Memory status after loss (softmax/NCE) layer was initialized-----------\n";
	print_GPU_Info();


	if(!LM) {

		bool top_layer_flag = false;
		if (num_layers==1 && params.bi_dir_params.bi_dir) {
			top_layer_flag = true;
		}

		bool combine_embeddings = false;
		if(params.bi_dir_params.share_embeddings) {
			combine_embeddings = true;
		}

		//Initialize the input layer
		input_layer_source.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout, dropout_rate,top_layer_flag,false,NULL,combine_embeddings,params,true);
	}

	if(params.bi_dir_params.bi_dir) {
		bool top_layer_flag = false;
		if (num_layers==1) {
			top_layer_flag = true;
		}

		input_layer_source_bi.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout, dropout_rate,top_layer_flag,params.bi_dir_params.share_embeddings,input_layer_source.d_W,false,params,true);
	}

	if(multi_source) {
		input_layer_source_bi.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,dropout,dropout_rate,false,false,NULL,false,params,true);
	}

	input_layer_target.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102,dropout, dropout_rate,false,false,NULL,false,params,false);

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

	BZ_CUDA::logger << "--------Memory status after Layer 1 was initialized--------\n";
	print_GPU_Info();

	//do this to be sure addresses stay the same
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
		}

		if(params.bi_dir_params.bi_dir || multi_source) {
			source_hidden_layers_bi.push_back(Hidden_To_Hidden_Layer<dType>());
		}
		target_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
	}

	//now initialize hidden layers
	for(int i=1; i<num_layers; i++) {

		if(!LM) {
			bool top_layer_flag = false;
			if((i == (num_layers-1)) && params.bi_dir_params.bi_dir) {
				top_layer_flag = true;
			}
			source_hidden_layers[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
			source_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,
				norm_clip,this,103,dropout, dropout_rate,top_layer_flag,i);
		}

		if(params.bi_dir_params.bi_dir) {

			bool top_layer_flag = false;
			if((i == (num_layers-1))) {
				top_layer_flag = true;
			}

			source_hidden_layers_bi[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
			source_hidden_layers_bi[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,
				this,103,dropout, dropout_rate,top_layer_flag,i);
		}

		if(multi_source) {
			source_hidden_layers_bi[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
			source_hidden_layers_bi[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,
				this,103,dropout, dropout_rate,false,i);
		}

		target_hidden_layers[i-1].hh_layer_info = layer_lookups[final_gpu_indicies[i]];;
		target_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,
			norm_clip,this,103,dropout, dropout_rate,false,i);
		BZ_CUDA::logger << "--------Memory status after Layer " << i+1 << " was initialized--------\n";
		print_GPU_Info();
	}


	//initialize the bidirectional layer here
	if(bi_dir) {
		bi_dir_source.init_layer(params,final_gpu_indicies[num_layers-1],this,final_gpu_indicies);
	}

	if(multi_source) {
		multi_source_layer.init_layer(params,this,final_gpu_indicies);
	}

	if(this->bi_dir && params.bi_dir_params.bi_dir_comb) {
		input_layer_source_bi.nonrev_bi_dir = true;
		for(int i=1; i<num_layers;i++) {
			source_hidden_layers_bi[i-1].nonrev_bi_dir = true;
		}
	}

	//initialize the attention layer on top layer, by this time all the other layers have been initialized
	if(attent_params.attention_model) {
		if(num_layers==1) {
			input_layer_target.init_attention(final_gpu_indicies[0],attent_params.D,attent_params.feed_input,this,params);
			for(int i=0; i<longest_sent; i++) {
				if(!params.multi_src_params.multi_attention) {
					input_layer_target.attent_layer->nodes[i].d_h_t = input_layer_target.nodes[i].d_h_t;
					input_layer_target.attent_layer->nodes[i].d_d_ERRt_ht_tild = input_layer_target.nodes[i].d_d_ERRt_ht;
					input_layer_target.attent_layer->nodes[i].d_indicies_mask = &input_layer_target.nodes[i].d_input_vocab_indices_01;
				}
				else {
					input_layer_target.attent_layer->nodes[i].d_h_t = input_layer_target.nodes[i].d_h_t;
					input_layer_target.attent_layer_bi->nodes[i].d_h_t = input_layer_target.nodes[i].d_h_t;

					input_layer_target.att_comb_layer->nodes[i]->d_ht_1 = input_layer_target.attent_layer->nodes[i].d_final_temp_2;
					input_layer_target.att_comb_layer->nodes[i]->d_ht_2 = input_layer_target.attent_layer_bi->nodes[i].d_final_temp_2;

					input_layer_target.attent_layer->nodes[i].d_indicies_mask = &input_layer_target.nodes[i].d_input_vocab_indices_01;
					input_layer_target.attent_layer_bi->nodes[i].d_indicies_mask = &input_layer_target.nodes[i].d_input_vocab_indices_01;
					input_layer_target.att_comb_layer->nodes[i]->d_indicies_mask = &input_layer_target.nodes[i].d_input_vocab_indices_01;

					input_layer_target.att_comb_layer->nodes[i]->d_ERR_ht_top_loss = input_layer_target.nodes[i].d_d_ERRt_ht;

					input_layer_target.attent_layer->nodes[i].d_d_ERRt_ht_tild = input_layer_target.att_comb_layer->nodes[i]->d_ERR_ht_1;
					input_layer_target.attent_layer_bi->nodes[i].d_d_ERRt_ht_tild = input_layer_target.att_comb_layer->nodes[i]->d_ERR_ht_2;
				}
			}

			if(attent_params.feed_input) {
				input_layer_target.init_feed_input(NULL,params.multi_src_params.multi_attention);
				if(!params.multi_src_params.multi_attention) {
					input_layer_target.ih_layer_info.attention_forward = input_layer_target.attent_layer->layer_info.forward_prop_done;
					input_layer_target.attent_layer->layer_info.error_htild_below= input_layer_target.ih_layer_info.error_htild_below;
				}
				else {
					input_layer_target.ih_layer_info.attention_forward = input_layer_target.att_comb_layer->forward_prop_done;
					input_layer_target.att_comb_layer->error_htild_below= input_layer_target.ih_layer_info.error_htild_below;
				}
			}
		}
		else {
			target_hidden_layers[num_layers-2].init_attention(final_gpu_indicies[num_layers-1],attent_params.D,attent_params.feed_input,this,params);
			for(int i=0; i<longest_sent; i++) {
				if(!params.multi_src_params.multi_attention) {
					target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_h_t = target_hidden_layers[num_layers-2].nodes[i].d_h_t;
					target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_d_ERRt_ht_tild  = target_hidden_layers[num_layers-2].nodes[i].d_d_ERRt_ht;
					target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_indicies_mask  = &target_hidden_layers[num_layers-2].nodes[i].d_input_vocab_indices_01;
				}
				else {
					target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_h_t = target_hidden_layers[num_layers-2].nodes[i].d_h_t;
					target_hidden_layers[num_layers-2].attent_layer_bi->nodes[i].d_h_t = target_hidden_layers[num_layers-2].nodes[i].d_h_t;

					target_hidden_layers[num_layers-2].att_comb_layer->nodes[i]->d_ht_1 = target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_final_temp_2;
					target_hidden_layers[num_layers-2].att_comb_layer->nodes[i]->d_ht_2 = target_hidden_layers[num_layers-2].attent_layer_bi->nodes[i].d_final_temp_2;

					target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_indicies_mask  = &target_hidden_layers[num_layers-2].nodes[i].d_input_vocab_indices_01;
					target_hidden_layers[num_layers-2].attent_layer_bi->nodes[i].d_indicies_mask  = &target_hidden_layers[num_layers-2].nodes[i].d_input_vocab_indices_01;
					target_hidden_layers[num_layers-2].att_comb_layer->nodes[i]->d_indicies_mask = &target_hidden_layers[num_layers-2].nodes[i].d_input_vocab_indices_01;

					target_hidden_layers[num_layers-2].att_comb_layer->nodes[i]->d_ERR_ht_top_loss = target_hidden_layers[num_layers-2].nodes[i].d_d_ERRt_ht;

					target_hidden_layers[num_layers-2].attent_layer->nodes[i].d_d_ERRt_ht_tild  = target_hidden_layers[num_layers-2].att_comb_layer->nodes[i]->d_ERR_ht_1;
					target_hidden_layers[num_layers-2].attent_layer_bi->nodes[i].d_d_ERRt_ht_tild  = target_hidden_layers[num_layers-2].att_comb_layer->nodes[i]->d_ERR_ht_2;
				}
			}

			if(attent_params.feed_input) {
				input_layer_target.init_feed_input(&target_hidden_layers[num_layers-2],params.multi_src_params.multi_attention);
				if(!params.multi_src_params.multi_attention) {
					input_layer_target.ih_layer_info.attention_forward = target_hidden_layers[num_layers-2].attent_layer->layer_info.forward_prop_done;
					target_hidden_layers[num_layers-2].attent_layer->layer_info.error_htild_below = input_layer_target.ih_layer_info.error_htild_below;
				}
				else {
					input_layer_target.ih_layer_info.attention_forward = target_hidden_layers[num_layers-2].att_comb_layer->forward_prop_done;
					target_hidden_layers[num_layers-2].att_comb_layer->error_htild_below = input_layer_target.ih_layer_info.error_htild_below;
				}
			}
		}

		BZ_CUDA::logger << "--------Memory status after Attention Layer was initialized--------\n";
		print_GPU_Info();
	}

	if(params.bi_dir_params.bi_dir) {
		for(int i=0; i<longest_sent; i++) {
			if(num_layers==1) {
				input_layer_source.nodes[i].d_bi_dir_ht = bi_dir_source.d_ht_rev_total[i];
				input_layer_source_bi.nodes[i].d_bi_dir_ht = bi_dir_source.d_ht_nonrev_total[i];

				bi_dir_source.d_ht_rev_total_errors[i] = input_layer_source.nodes[i].d_d_ERRt_ht;
				bi_dir_source.d_ht_nonrev_total_errors[i] = input_layer_source_bi.nodes[i].d_d_ERRt_ht;
			}
			else {
				source_hidden_layers[num_layers-2].nodes[i].d_bi_dir_ht = bi_dir_source.d_ht_rev_total[i];
				source_hidden_layers_bi[num_layers-2].nodes[i].d_bi_dir_ht = bi_dir_source.d_ht_nonrev_total[i];

				bi_dir_source.d_ht_rev_total_errors[i] = source_hidden_layers[num_layers-2].nodes[i].d_d_ERRt_ht;
				bi_dir_source.d_ht_nonrev_total_errors[i] = source_hidden_layers_bi[num_layers-2].nodes[i].d_d_ERRt_ht;
			}
		}
	}

	if(num_layers==1) {
		if(final_gpu_indicies[0]==final_gpu_indicies[1] && !dropout && !attent_params.attention_model && false) {
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

			if(bi_dir || multi_source) {
				input_layer_source_bi.upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
			}
		}
	}
	else {
		if(final_gpu_indicies[0]==final_gpu_indicies[1] && !dropout && !attent_params.attention_model && false) {
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

			if(bi_dir || multi_source) {
				input_layer_source_bi.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers_bi[0]);
			}
		}

		for(int i=0; i<target_hidden_layers.size(); i++) {

			//lower transfer stuff
			if(i==0) {
				if(final_gpu_indicies[0]==final_gpu_indicies[1] && !dropout && !attent_params.attention_model && false) {
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

					if(bi_dir || multi_source) {
						source_hidden_layers_bi[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source_bi,NULL);
					}
				}
			}
			else {
				if(final_gpu_indicies[i]==final_gpu_indicies[i+1] && !dropout && !attent_params.attention_model && false) {
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

					if(bi_dir || multi_source) {
						source_hidden_layers_bi[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers_bi[i-1]);
					}
				}
			}

			//upper transfer stuff
			if(i==target_hidden_layers.size()-1) {
				if(final_gpu_indicies[i+1]==final_gpu_indicies[i+2] && !dropout && !attent_params.attention_model && false) {
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

					if(bi_dir || multi_source) {
						source_hidden_layers_bi[i].upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
					}
				}
			}
			else {
				if(final_gpu_indicies[i+1]==final_gpu_indicies[i+2] && !dropout && !attent_params.attention_model && false) {
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

					if(bi_dir || multi_source) {
						source_hidden_layers_bi[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers_bi[i+1]);
					}
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
  	//int selected = 0;
  	for (int i = 0; i < num_devices; i++) {
	    cudaDeviceProp prop;
	    cudaGetDeviceProperties(&prop, i);
	    BZ_CUDA::logger << "Device Number: " << i << "\n";
	    BZ_CUDA::logger << "Device Name: " << prop.name << "\n";
	   	cudaSetDevice(i);
	    cudaMemGetInfo( &free_bytes, &total_bytes);
	    BZ_CUDA::logger << "Total Memory (MB): " << (double)total_bytes/(1.0e6) << "\n";
	    BZ_CUDA::logger << "Memory Free (MB): " << (double)free_bytes/(1.0e6) << "\n\n";
  	}
  	cudaSetDevice(0);
}


//called when doing ensemble decoding
template<typename dType>
void neuralMT_model<dType>::init_prev_states(int num_layers, int LSTM_size,int minibatch_size, int device_number,bool multi_source) {

	cudaSetDevice(device_number);
	for(int i=0; i<num_layers; i++) {
		previous_source_states.push_back( prev_source_state<dType>(LSTM_size) );
		previous_target_states.push_back( prev_target_state<dType>(LSTM_size,minibatch_size) );
		if(multi_source) {
			previous_source_states_bi.push_back( prev_source_state<dType>(LSTM_size) );
		}
	}

}

//for this we need to initialze the source minibatch size to one
template<typename dType>
void neuralMT_model<dType>::initModel_decoding(int LSTM_size,int beam_size,int source_vocab_size,int target_vocab_size,
	int num_layers,std::string input_weight_file,int gpu_num,global_params &params,
	bool attention_model,bool feed_input,bool multi_source,bool combine_LSTM,bool char_cnn) {

	//before initializing the layers, get the number of layers, number of GPU's and allocate them accordingly
	//softmax->s_layer_info.init(gpu_num);
	softmax = new softmax_layer<dType>();
	s_layer_info = softmax->gpu_init(gpu_num);
	//s_layer_info = softmax->s_layer_info;//remove soon
	input_layer_source.ih_layer_info.init(gpu_num);
	input_layer_target.ih_layer_info = input_layer_source.ih_layer_info;

	decode = true;

	//for initializing the model for decoding
	const int longest_sent =1;
	const int minibatch_size = 1;
	const bool debug = false;
	const dType learning_rate=0;
	const bool clip_gradients = false;
	const dType norm_clip = 0;
	const bool LM = false;
	//const bool truncated_softmax = false;
	//const int trunc_size=0;
	//const bool softmax_scaled = true;
	//const bool train_perplexity = false;
	const std::string output_weight_file = "NULL";
	//const bool dropout_rate = 0;

	//change these for initialization each time
	params.minibatch_size = params.beam_size;
	params.LSTM_size = LSTM_size;

	this->multi_source = multi_source; //need this for source forward prop and for loading in weights correctly

	this->char_cnn = char_cnn;

	if(multi_source && attention_model) {
		this->multi_attention_v2 = true;
		//BZ_CUDA::logger << "Multi-source attention model is set to true\n";
	}

	//BZ_CUDA::logger << "Beam size: " << beam_size << "\n";
	//BZ_CUDA::logger << "Beam size: " << params.beam_size << "\n";
	//BZ_CUDA::logger << "LSTM size: " << LSTM_size << "\n";
	//BZ_CUDA::logger << "Multi-source: " << multi_source << "\n";
	//BZ_CUDA::logger << "Combine LSTM: " << combine_LSTM << "\n";
	//BZ_CUDA::logger << "Char cnn: " << char_cnn << "\n";
	//BZ_CUDA::logger << "Num Layers: " << num_layers << "\n";
	//BZ_CUDA::logger << "Attention model: " << attention_model << "\n";
	//BZ_CUDA::logger << "Feed Input: " << feed_input << "\n";

	if(char_cnn) {
		extract_char_info(params.char_params.longest_word,params.char_params.num_unique_chars_source,
    	      params.char_params.num_unique_chars_target,params.source_vocab_size,params.target_vocab_size,
    	      params.char_params.char_mapping_file,params.char_params.word_mapping_file);
	}

	//Initialize the softmax layer
	softmax->init_loss_layer(this,params);

	//Now print gpu info
	BZ_CUDA::logger << "----------Memory status after softmax layer was initialized-----------\n";
	print_GPU_Info();

	if(!LM) {
		//Initialize the input layer
		input_layer_source.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,false,0,false,false,NULL,false,params,true);

		if(multi_source) {
			input_layer_source_bi.ih_layer_info = input_layer_source.ih_layer_info;
			input_layer_source_bi.init_Input_To_Hidden_Layer(LSTM_size,minibatch_size,source_vocab_size,
	 			longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,101,false,0,false,false,NULL,false,params,true);
		}
	}

	input_layer_target.init_Input_To_Hidden_Layer(LSTM_size,beam_size,target_vocab_size,
 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,102,false,0,false,false,NULL,false,params,false);



	//attention model initialization
	if(attention_model) {
		//BZ_CUDA::logger << "Initializing Attention Layer\n";
		attent_params.attention_model = attention_model;
		attent_params.feed_input = feed_input;
		dType *h_temp;
		for(int i=0; i<params.longest_sent; i++) {
			top_source_states.push_back(NULL);
			top_source_states_v2.push_back(NULL);
		}
		for(int i=0; i<params.longest_sent; i++) {
			full_matrix_setup(&h_temp,&top_source_states[i],LSTM_size,beam_size);
			full_matrix_setup(&h_temp,&top_source_states_v2[i],LSTM_size,beam_size);
		}

		if(feed_input) {
		//	BZ_CUDA::logger << "Initializing feed_input\n";
			input_layer_target.decoder_init_feed_input();
			input_layer_target.nodes[0].attention_extra();
			input_layer_target.nodes[0].index = 1;
		}


		//feed input is always set as false as not transfers should be automatically sent, this is done manually in decoding
		decoder_att_layer.init_att_decoder(params.LSTM_size,params.beam_size,gpu_num,attent_params.D, params.longest_sent,input_layer_source.ih_layer_info.handle,this,
			false,top_source_states,multi_attention_v2,top_source_states_v2);

		BZ_CUDA::logger << "--------Memory status after Attention Layer was initialized--------\n";
		print_GPU_Info();
	}

	if(multi_source) {
		std::vector<int> final_gpu_indicies;
		for(int i=0; i<num_layers; i++) {
			final_gpu_indicies.push_back(gpu_num);
		}
		multi_source_layer.init_layer_decoder(this,gpu_num,combine_LSTM,LSTM_size,num_layers); //ERROR NEED TO MAKE SURE THAT PARAMS DOESN"T HAVE TO REFLECT THE CURRENT MODEL OR CHANGE IT
	}	

	//Initialize the hidden layer
	// hidden_layer.init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,input_vocab_size,output_vocab_size,
 // 		longest_sent,debug,learning_rate,clip_gradients,norm_clip,this);

	this->input_weight_file = input_weight_file;
	this->output_weight_file = output_weight_file;
	this->truncated_softmax = false;
	this->LM = false;

	BZ_CUDA::logger << "--------Memory status after Layer 1 was initialized--------\n";
	print_GPU_Info();

	//do this to be sure addresses stay the same
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
		}
		if(multi_source) {
			source_hidden_layers_bi.push_back(Hidden_To_Hidden_Layer<dType>());
		}

		target_hidden_layers.push_back(Hidden_To_Hidden_Layer<dType>());
	}

	//now initialize hidden layers
	for(int i=1; i<num_layers; i++) {
		if(!LM) {
			source_hidden_layers[i-1].hh_layer_info = input_layer_target.ih_layer_info;
			source_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,false,0,false,i);
		}

		if(multi_source) {
			source_hidden_layers_bi[i-1].hh_layer_info = input_layer_target.ih_layer_info;
			source_hidden_layers_bi[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,minibatch_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,false,0,false,i);
		}

		target_hidden_layers[i-1].hh_layer_info = input_layer_target.ih_layer_info;
		target_hidden_layers[i-1].init_Hidden_To_Hidden_Layer(LSTM_size,beam_size,longest_sent,debug,learning_rate,clip_gradients,norm_clip,this,103,false,0,false,i);
		BZ_CUDA::logger << "--------Memory status after Layer " << i+1 << " was initialized--------\n";
		print_GPU_Info();
	}

	
	//now the layer info
	if(num_layers==1) {
		input_layer_source.upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
		input_layer_target.upper_layer.init_upper_transfer_layer(true,true,false,softmax,NULL);
		softmax->init_lower_transfer_layer(true,true,&input_layer_target,NULL);

		if(multi_source) {
			input_layer_source_bi.upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
		}
	}
	else {
		input_layer_source.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[0]);
		input_layer_target.upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[0]);

		if(multi_source) {
			input_layer_source_bi.upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers_bi[0]);
		}

		for(int i=0; i<target_hidden_layers.size(); i++) {

			//lower transfer stuff
			if(i==0) {
				source_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source,NULL);
				target_hidden_layers[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_target,NULL);

				if(multi_source) {
					source_hidden_layers_bi[0].lower_layer.init_lower_transfer_layer(true,true,&input_layer_source_bi,NULL);
				}
			}
			else {
				source_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers[i-1]);
				target_hidden_layers[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i-1]);

				if(multi_source) {
					source_hidden_layers_bi[i].lower_layer.init_lower_transfer_layer(false,true,NULL,&source_hidden_layers_bi[i-1]);
				}
			}

			//upper transfer stuff
			if(i==target_hidden_layers.size()-1) {
				source_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
				target_hidden_layers[i].upper_layer.init_upper_transfer_layer(true,true,false,softmax,NULL);
				softmax->init_lower_transfer_layer(false,true,NULL,&target_hidden_layers[i]);

				if(multi_source) {
					source_hidden_layers_bi[i].upper_layer.init_upper_transfer_layer(true,true,true,softmax,NULL);
				}
			}
			else {
				source_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers[i+1]);
				target_hidden_layers[i].upper_layer.init_upper_transfer_layer(false,true,false,NULL,&target_hidden_layers[i+1]);

				if(multi_source) {
					source_hidden_layers_bi[i].upper_layer.init_upper_transfer_layer(false,true,true,NULL,&source_hidden_layers_bi[i+1]);
				}
			}
		}
	}
}

template<typename dType>
void neuralMT_model<dType>::init_GPUs() {

}


// template<typename dType>
// void print_src_hid_state(dType *d_ptr) {

// 	thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(d_ptr);
// 	std::cout << "Printing source hidden state\n";
// 	for(int i=0; i<4; i++) {
// 		std::cout << debug_ptr[i] << " ";
// 	}
// 	std::cout << "\n";
// }


template<typename dType>
template<typename Derived>
void neuralMT_model<dType>::compute_gradients(const Eigen::MatrixBase<Derived> &source_input_minibatch_const,
	const Eigen::MatrixBase<Derived> &source_output_minibatch_const,const Eigen::MatrixBase<Derived> &target_input_minibatch_const,
	const Eigen::MatrixBase<Derived> &target_output_minibatch_const,int *h_input_vocab_indicies_source,
	int *h_output_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_output_vocab_indicies_target,
	int current_source_length,int current_target_length,int *h_input_vocab_indicies_source_Wgrad,int *h_input_vocab_indicies_target_Wgrad,
	int len_source_Wgrad,int len_target_Wgrad,int *h_sampled_indices,int len_unique_words_trunc_softmax,int *h_batch_info,file_helper *temp_fh) 
{
	//Clear the gradients before forward/backward pass
	//eventually clear gradients at the end
	//clear_gradients();


	//std::cout << "----------------------STARTING COMPUTE GRADIENTS----------------------\n";
	//std::cout << "---------------------------------COMPUTE GRADIENTS STARTING---------------------------------\n";
	train = true;

	source_length = current_source_length;

	//Send the CPU vocab input data to the GPU layers
	//For the input layer, 2 host vectors must be transfered since need special preprocessing for W gradient
	if(!LM){
		input_layer_source.prep_GPU_vocab_indices(h_input_vocab_indicies_source,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].prep_GPU_vocab_indices(h_input_vocab_indicies_source,current_source_length);
		}
	}

	if(bi_dir) {
		bi_dir_source.longest_sent_minibatch = current_source_length;
		bi_dir_source.reverse_indicies(h_input_vocab_indicies_source,current_source_length);
		input_layer_source_bi.prep_GPU_vocab_indices(bi_dir_source.h_source_indicies,h_input_vocab_indicies_source_Wgrad,current_source_length,len_source_Wgrad);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].prep_GPU_vocab_indices(bi_dir_source.h_source_indicies,current_source_length);
		}
	}

	if(multi_source) {
		multi_source_layer.longest_sent_minibatch_s1 = file_info->current_source_length;
		multi_source_layer.longest_sent_minibatch_s2 = src_fh.current_source_length;
		input_layer_source_bi.prep_GPU_vocab_indices(src_fh.h_input_vocab_indicies_source,src_fh.h_input_vocab_indicies_source_Wgrad,src_fh.current_source_length,src_fh.len_source_Wgrad);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].prep_GPU_vocab_indices(src_fh.h_input_vocab_indicies_source,src_fh.current_source_length);
		}
	}

	input_layer_target.prep_GPU_vocab_indices(h_input_vocab_indicies_target,h_input_vocab_indicies_target_Wgrad,current_target_length,len_target_Wgrad);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].prep_GPU_vocab_indices(h_input_vocab_indicies_target,current_target_length);
	}

	softmax->prep_GPU_vocab_indices(h_output_vocab_indicies_target,current_target_length);

	if(char_cnn) {
		input_layer_source.prep_char_cnn(temp_fh->fhc->h_char_vocab_indicies_source,current_source_length,
			temp_fh->fhc->h_unique_chars_source,temp_fh->fhc->num_unique_chars_source);
		input_layer_target.prep_char_cnn(temp_fh->fhc->h_char_vocab_indicies_target,current_target_length,
			temp_fh->fhc->h_unique_chars_target,temp_fh->fhc->num_unique_chars_target);
	}

	if(attent_params.attention_model) {
		if(target_hidden_layers.size()==0) {
			if(!multi_attention) {
				input_layer_target.attent_layer->prep_minibatch_info(h_batch_info);

				if(multi_attention_v2) {
					input_layer_target.attent_layer->prep_minibatch_info_v2(src_fh.h_batch_info);
				}
			}
			else {
				input_layer_target.attent_layer->prep_minibatch_info(h_batch_info);
				input_layer_target.attent_layer_bi->prep_minibatch_info(src_fh.h_batch_info);
			}
		}
		else {
			if(!multi_attention) {
				target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(h_batch_info);

				if(multi_attention_v2) {
					target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info_v2(src_fh.h_batch_info);
				}
			}
			else {
				target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(h_batch_info);
				target_hidden_layers[target_hidden_layers.size()-1].attent_layer_bi->prep_minibatch_info(src_fh.h_batch_info);
			}
		}
	}
	devSynchAll();
	CUDA_GET_LAST_ERROR("POST INDICES SETUP GPU");


	//std::cout << "Starting source foward:\n";
	//std::cout << "Source Forward Index: 0\n"; 
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


		if(bi_dir) {
			input_layer_source_bi.nodes[0].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full,
				input_layer_source_bi.d_input_vocab_indices_01_full,
				input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector);
			input_layer_source_bi.nodes[0].forward_prop();

			//mgpu stuff
			for(int i=0; i<source_hidden_layers_bi.size(); i++) {
				source_hidden_layers_bi[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers_bi[i].d_input_vocab_indices_01_full,
					source_hidden_layers_bi[i].d_init_hidden_vector,source_hidden_layers_bi[i].d_init_cell_vector);
				source_hidden_layers_bi[i].nodes[0].forward_prop();
			}
		}


		//cudaDeviceSynchronize();
		//for(int i=1; i<source_input_minibatch_const.cols(); i++) {
		for(int i=1; i<current_source_length; i++) {
			//std::cout << "Source Forward Index: " << i << "\n"; 
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

			if(bi_dir) {
				input_layer_source_bi.nodes[i].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full+step,
					input_layer_source_bi.d_input_vocab_indices_01_full+step,
					input_layer_source_bi.nodes[i-1].d_h_t,input_layer_source_bi.nodes[i-1].d_c_t);
				input_layer_source_bi.nodes[i].forward_prop();
				//cudaDeviceSynchronize();

				//mgpu stuff
				for(int j=0; j<source_hidden_layers_bi.size(); j++) {
					source_hidden_layers_bi[j].nodes[i].update_vectors_forward_GPU(source_hidden_layers_bi[j].d_input_vocab_indices_01_full+step,
						source_hidden_layers_bi[j].nodes[i-1].d_h_t,source_hidden_layers_bi[j].nodes[i-1].d_c_t);
					source_hidden_layers_bi[j].nodes[i].forward_prop();
				}
			}
		}
	}

	//do all the bi-directional layers below
	if(multi_source) {
		input_layer_source_bi.nodes[0].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full,
			input_layer_source_bi.d_input_vocab_indices_01_full,
			input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector);
		input_layer_source_bi.nodes[0].forward_prop();

		//mgpu stuff
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers_bi[i].d_input_vocab_indices_01_full,
				source_hidden_layers_bi[i].d_init_hidden_vector,source_hidden_layers_bi[i].d_init_cell_vector);
			source_hidden_layers_bi[i].nodes[0].forward_prop();
		}

		for(int i=1; i<src_fh.current_source_length; i++) {
			int step = i*input_layer_source.minibatch_size;

			input_layer_source_bi.nodes[i].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full+step,
				input_layer_source_bi.d_input_vocab_indices_01_full+step,
				input_layer_source_bi.nodes[i-1].d_h_t,input_layer_source_bi.nodes[i-1].d_c_t);
			input_layer_source_bi.nodes[i].forward_prop();
			//cudaDeviceSynchronize();

			//mgpu stuff
			for(int j=0; j<source_hidden_layers_bi.size(); j++) {
				source_hidden_layers_bi[j].nodes[i].update_vectors_forward_GPU(source_hidden_layers_bi[j].d_input_vocab_indices_01_full+step,
					source_hidden_layers_bi[j].nodes[i-1].d_h_t,source_hidden_layers_bi[j].nodes[i-1].d_c_t);
				source_hidden_layers_bi[j].nodes[i].forward_prop();
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


	if(bi_dir) {
		devSynchAll();
		bi_dir_source.forward_prop();
		devSynchAll();
	}

	if(multi_source) {
		devSynchAll();
		multi_source_layer.forward_prop();
		devSynchAll();
	}

	//std::cout << "---------------------- STARTING TARGET FORWARD PROP ------------------------------\n";

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

		if(bi_dir && bi_dir_source.model_type == COMBINE) {
			input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
				input_layer_target.d_input_vocab_indices_01_full,
				bi_dir_source.d_hs_final_target[0],bi_dir_source.d_ct_final_target[0]);

			//mgpu stuff
			for(int i=0; i<target_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
					bi_dir_source.d_hs_final_target[i+1],bi_dir_source.d_ct_final_target[i+1]);
			}
		}
		else if(multi_source) {
			input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
				input_layer_target.d_input_vocab_indices_01_full,
				multi_source_layer.d_hs_final_target[0],multi_source_layer.d_ct_final_target[0]);

			//mgpu stuff
			for(int i=0; i<target_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
					multi_source_layer.d_hs_final_target[i+1],multi_source_layer.d_ct_final_target[i+1]);
			}
		}
		else {
			input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
				input_layer_target.d_input_vocab_indices_01_full,
				input_layer_source.nodes[prev_source_index].d_h_t,input_layer_source.nodes[prev_source_index].d_c_t);

			//mgpu stuff
			for(int i=0; i<target_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
					source_hidden_layers[i].nodes[prev_source_index].d_h_t,source_hidden_layers[i].nodes[prev_source_index].d_c_t);
			}
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
		//std::cout << "Forward prop target index: " << i << " out of " << current_target_length-1 <<"\n";
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
		target_hidden_layers[i].nodes[last_index].backprop_prep_GPU(target_hidden_layers[i].d_init_d_ERRnTOtp1_ht,target_hidden_layers[i].d_init_d_ERRnTOtp1_ct);//,
		target_hidden_layers[i].nodes[last_index].back_prop_GPU(last_index);
	}

	input_layer_target.nodes[last_index].backprop_prep_GPU(input_layer_target.d_init_d_ERRnTOtp1_ht,input_layer_target.d_init_d_ERRnTOtp1_ct);//,
	
	input_layer_target.nodes[last_index].back_prop_GPU(last_index);


	for(int i=current_target_length-2; i>=0; i--) {
		//std::cout << "backward target index: " << i << " out of " << 0 << "\n";
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
		input_layer_target.nodes[i].back_prop_GPU(i);
	}


	if(bi_dir) {
		devSynchAll();
		bi_dir_source.back_prop();
		devSynchAll();
	}

	if(multi_source) {
		devSynchAll();
		multi_source_layer.back_prop();
		devSynchAll();
	}


	///////////////////////////Now do the backward pass for the source///////////////////////
	//std::cout << "STARTING BACKPROP SOURCE: " << "\n";
	if(!LM) {
		int prev_source_index = current_source_length-1;

		//mgpu stuff
		int backprop2_index=0;
		softmax->backprop_prep_GPU_mgpu(0);
		softmax->back_prop2(backprop2_index);
		backprop2_index++;

		//mgpu stuff
		for(int i=source_hidden_layers.size()-1; i>=0; i--) {

			if(bi_dir && bi_dir_source.model_type == COMBINE) {
				source_hidden_layers[i].nodes[prev_source_index].backprop_prep_GPU(bi_dir_source.d_hs_rev_error_horiz[i+1],bi_dir_source.d_ct_rev_error_horiz[i+1]);
			}
			else if(multi_source) {
				source_hidden_layers[i].nodes[prev_source_index].backprop_prep_GPU(multi_source_layer.d_hs_s1_error_horiz[i+1],multi_source_layer.d_ct_s1_error_horiz[i+1]);
			}
			else {
				source_hidden_layers[i].nodes[prev_source_index].backprop_prep_GPU(target_hidden_layers[i].d_d_ERRnTOt_htM1,target_hidden_layers[i].d_d_ERRnTOt_ctM1);
			}
			source_hidden_layers[i].nodes[prev_source_index].back_prop_GPU(prev_source_index);
		}

		if(bi_dir && bi_dir_source.model_type == COMBINE) {
			input_layer_source.nodes[prev_source_index].backprop_prep_GPU(bi_dir_source.d_hs_rev_error_horiz[0],bi_dir_source.d_ct_rev_error_horiz[0]);
		}
		else if(multi_source) {
			input_layer_source.nodes[prev_source_index].backprop_prep_GPU(multi_source_layer.d_hs_s1_error_horiz[0],multi_source_layer.d_ct_s1_error_horiz[0]);
		}
		else {
			input_layer_source.nodes[prev_source_index].backprop_prep_GPU(input_layer_target.d_d_ERRnTOt_htM1,
			 	input_layer_target.d_d_ERRnTOt_ctM1);//,input_layer_source.d_zeros);
		}
		input_layer_source.nodes[prev_source_index].back_prop_GPU(prev_source_index);


		if(bi_dir) {
			//pass in zero errors
			for(int i=source_hidden_layers_bi.size()-1; i>=0; i--) {
				source_hidden_layers_bi[i].nodes[prev_source_index].backprop_prep_GPU(source_hidden_layers_bi[i].d_init_d_ERRnTOtp1_ht,source_hidden_layers_bi[i].d_init_d_ERRnTOtp1_ct);
				source_hidden_layers_bi[i].nodes[prev_source_index].back_prop_GPU(prev_source_index);
			}
			input_layer_source_bi.nodes[prev_source_index].backprop_prep_GPU(input_layer_source_bi.d_init_d_ERRnTOtp1_ht,
			 	input_layer_source_bi.d_init_d_ERRnTOtp1_ct);//,input_layer_source.d_zeros);
			input_layer_source_bi.nodes[prev_source_index].back_prop_GPU(prev_source_index);
		}

			
		for(int i=current_source_length-2; i>=0; i--) {
			//std::cout << "------------------------Backward source index------------------------ " << i << "\n";

			//std::cout << "INDICIES CHECK GLOBAL 10\n";
			for(int j=source_hidden_layers.size()-1; j>=0; j--) {
				source_hidden_layers[j].nodes[i].backprop_prep_GPU(source_hidden_layers[j].d_d_ERRnTOt_htM1,source_hidden_layers[j].d_d_ERRnTOt_ctM1);//,
				source_hidden_layers[j].nodes[i].back_prop_GPU(i);
			}
			input_layer_source.nodes[i].backprop_prep_GPU(input_layer_source.d_d_ERRnTOt_htM1,input_layer_source.d_d_ERRnTOt_ctM1);
			input_layer_source.nodes[i].back_prop_GPU(i);

			if(bi_dir) {
				for(int j=source_hidden_layers_bi.size()-1; j>=0; j--) {
					source_hidden_layers_bi[j].nodes[i].backprop_prep_GPU(source_hidden_layers_bi[j].d_d_ERRnTOt_htM1,source_hidden_layers_bi[j].d_d_ERRnTOt_ctM1);//,
					source_hidden_layers_bi[j].nodes[i].back_prop_GPU(i);
				}
				input_layer_source_bi.nodes[i].backprop_prep_GPU(input_layer_source_bi.d_d_ERRnTOt_htM1,input_layer_source_bi.d_d_ERRnTOt_ctM1);
				input_layer_source_bi.nodes[i].back_prop_GPU(i);
			}

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


	//if multi source
	if(multi_source) {
		int prev_source_index = src_fh.current_source_length-1;

		//pass in zero errors
		for(int i=source_hidden_layers_bi.size()-1; i>=0; i--) {
			source_hidden_layers_bi[i].nodes[prev_source_index].backprop_prep_GPU(multi_source_layer.d_hs_s2_error_horiz[i+1],multi_source_layer.d_ct_s2_error_horiz[i+1]);
			source_hidden_layers_bi[i].nodes[prev_source_index].back_prop_GPU(prev_source_index);
		}
		input_layer_source_bi.nodes[prev_source_index].backprop_prep_GPU(multi_source_layer.d_hs_s2_error_horiz[0],multi_source_layer.d_ct_s2_error_horiz[0]);//,input_layer_source.d_zeros);
		input_layer_source_bi.nodes[prev_source_index].back_prop_GPU(prev_source_index);

		for(int i = src_fh.current_source_length-2; i>=0; i--) {
			//std::cout << "------------------------Backward source index------------------------ " << i << "\n";
			for(int j=source_hidden_layers_bi.size()-1; j>=0; j--) {
				source_hidden_layers_bi[j].nodes[i].backprop_prep_GPU(source_hidden_layers_bi[j].d_d_ERRnTOt_htM1,source_hidden_layers_bi[j].d_d_ERRnTOt_ctM1);//,
				source_hidden_layers_bi[j].nodes[i].back_prop_GPU(i);
			}
			input_layer_source_bi.nodes[i].backprop_prep_GPU(input_layer_source_bi.d_d_ERRnTOt_htM1,input_layer_source_bi.d_d_ERRnTOt_ctM1);
			input_layer_source_bi.nodes[i].back_prop_GPU(i);
		}
	}


	//std::cout << "Ending backprop\n";
	if(debug) {
		grad_check_flag = true;
		dType epsilon =(dType)1e-4;
		devSynchAll();
		src_fh_test = &src_fh;
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


//this function will only be entered in force-decode
template<typename dType>
void neuralMT_model<dType>::dump_alignments(int target_length,int minibatch_size,int *h_input_vocab_indicies_source,int *h_input_vocab_indicies_target,int *h_input_vocab_indicies_source_2) {

	BZ_CUDA::logger << "------------------Starting dump alignments-------------------\n";
	devSynchAll();

	if(minibatch_size!=1) {
		BZ_CUDA::logger << "ERROR: for printing alignments you must set the minibatch size to one\n";
		exit (EXIT_FAILURE);
	}

	dType *h_p_t;
	int *h_batch_info;
	h_p_t = (dType *)malloc(minibatch_size* sizeof(dType));
	h_batch_info = (int *)malloc(minibatch_size*2 * sizeof(int));
	int *h_indicies = (int *)malloc(minibatch_size * (attent_params.D*2+1) * sizeof(int));
	dType *h_alignments = (dType *)malloc(minibatch_size * (attent_params.D*2+1) * sizeof(dType));
	dType *h_cached_exp = (dType *)malloc(minibatch_size * (attent_params.D*2+1) * sizeof(dType));


	//for the multi-source attention model if it is being used
	dType *h_p_t_ms;
	int *h_batch_info_ms;
	int *h_indicies_ms;
	dType *h_alignments_ms;
	dType *h_cached_exp_ms;
	//print out the double attention alignments
	if(multi_attention_v2) {
		h_p_t_ms = (dType *)malloc(minibatch_size* sizeof(dType));
		h_batch_info_ms = (int *)malloc(minibatch_size*2 * sizeof(int));
		h_indicies_ms = (int *)malloc(minibatch_size * (attent_params.D*2+1) * sizeof(int));
		h_alignments_ms = (dType *)malloc(minibatch_size * (attent_params.D*2+1) * sizeof(dType));
		h_cached_exp_ms = (dType *)malloc(minibatch_size * (attent_params.D*2+1) * sizeof(dType));
	}

	std::vector<std::vector<int>> output_indicies;
	for(int i=0; i<minibatch_size*2; i++) {
		std::vector<int> temp;
		output_indicies.push_back( temp );
	}

	//push back one more
	if(multi_attention_v2) {
		std::vector<int> temp;
		output_indicies.push_back( temp );
	}


	std::vector<std::string> alignment_nums; //stores in string format 1-3 2-4 4-5, etc..
	for(int i=0; i<minibatch_size; i++) {
		alignment_nums.push_back(" ");
	}
	
	if(target_hidden_layers.size()==0) {
		cudaMemcpy(h_batch_info,input_layer_target.attent_layer->d_batch_info,minibatch_size*2*sizeof(int),cudaMemcpyDeviceToHost);

		if(multi_attention_v2) {
			cudaMemcpy(h_batch_info_ms,input_layer_target.attent_layer->d_batch_info_v2,minibatch_size*2*sizeof(int),cudaMemcpyDeviceToHost);
		}
	}
	else {
		cudaMemcpy(h_batch_info,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->d_batch_info,minibatch_size*2*sizeof(int),cudaMemcpyDeviceToHost);

		if(multi_attention_v2) {
			cudaMemcpy(h_batch_info_ms,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->d_batch_info_v2,minibatch_size*2*sizeof(int),cudaMemcpyDeviceToHost);
		}
	}


	// for(int i=h_batch_info[0]-1; i>=0; i--) {
	// 	std::cout << h_input_vocab_indicies_source[i] << " ";
	// }
	// std::cout << "\n";

	// for(int i=0; i<target_length; i++) {
	// 	std::cout << h_input_vocab_indicies_target[i] << " ";
	// }
	// std::cout << "\n";

	std::string output_indicies_string = ""; //for tgt_indx-src1_index-src2_indx

	for(int i=1; i<target_length; i++) {
		if(target_hidden_layers.size()==0) {
			cudaMemcpy(h_p_t,input_layer_target.attent_layer->nodes[i].d_p_t,minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_indicies,input_layer_target.attent_layer->nodes[i].d_indicies,minibatch_size * (attent_params.D*2+1)*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_alignments,input_layer_target.attent_layer->nodes[i].d_alignments,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_cached_exp,input_layer_target.attent_layer->nodes[i].d_cached_exp,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);

			if(multi_attention_v2) {
				cudaMemcpy(h_p_t_ms,input_layer_target.attent_layer->nodes[i].d_p_t_v2,minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost);
				cudaMemcpy(h_indicies_ms,input_layer_target.attent_layer->nodes[i].d_indicies_v2,minibatch_size * (attent_params.D*2+1)*sizeof(int),cudaMemcpyDeviceToHost);
				cudaMemcpy(h_alignments_ms,input_layer_target.attent_layer->nodes[i].d_alignments_v2,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
				cudaMemcpy(h_cached_exp_ms,input_layer_target.attent_layer->nodes[i].d_cached_exp_v2,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
			}
		}
		else {
			cudaMemcpy(h_p_t,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_p_t,minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_indicies,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_indicies,minibatch_size * (attent_params.D*2+1)*sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_alignments,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_alignments,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_cached_exp,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_cached_exp,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
			
			if(multi_attention_v2) {
				cudaMemcpy(h_p_t_ms,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_p_t_v2,minibatch_size*sizeof(dType),cudaMemcpyDeviceToHost);
				cudaMemcpy(h_indicies_ms,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_indicies_v2,minibatch_size * (attent_params.D*2+1)*sizeof(int),cudaMemcpyDeviceToHost);
				cudaMemcpy(h_alignments_ms,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_alignments_v2,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
				cudaMemcpy(h_cached_exp_ms,target_hidden_layers[target_hidden_layers.size()-1].attent_layer->nodes[i].d_cached_exp_v2,minibatch_size * (attent_params.D*2+1)*sizeof(dType),cudaMemcpyDeviceToHost);
			}
		}	
		BZ_CUDA::logger << "Target index: " << i << "  p_t: " << h_p_t[0] << "  p_t_v2: " << h_p_t_ms[0] << "\n";
		for(int j=0; j<minibatch_size; j++) {
			if( h_input_vocab_indicies_target[ IDX2C(j,i,minibatch_size) ]!=-1) {

				//find the position with the highest alignment
				double max_val = 0;
				int max_index = -1;
				double max_val_ms = 0;
				int max_index_ms = -1;
				BZ_CUDA::logger << "Printing alignment weights and indexes for first encoder\n";
				for(int k=0; k < 2*attent_params.D+1; k++) {
					BZ_CUDA::logger << "alignment index: " << h_indicies[k] << "   alignment value: " << h_alignments[k] << "\n";
					if(h_alignments[k] > max_val) {
						max_val = h_alignments[k];
						max_index = h_indicies[k];
					} 
				}

				BZ_CUDA::logger << "Printing alignment weights and indexes for second encoder\n";
				if(multi_attention_v2) {
					for(int k=0; k < 2*attent_params.D+1; k++) {
						std::cout << "alignment index: " << h_indicies_ms[k] << "   alignment value: " << h_alignments_ms[k] << "\n";
						if(h_alignments_ms[k] > max_val_ms) {
							max_val_ms = h_alignments_ms[k];
							max_index_ms = h_indicies_ms[k];
						} 
					}
				}

				output_indicies[0 + 2*j].push_back( h_input_vocab_indicies_source[ IDX2C(j,h_batch_info[j] - 1 - (int)max_index + h_batch_info[j+minibatch_size],minibatch_size) ] );
				output_indicies[1 + 2*j].push_back( h_input_vocab_indicies_target[ IDX2C(j,i,minibatch_size) ] );
				output_indicies[2 + 2*j].push_back( h_input_vocab_indicies_source_2[ IDX2C(j,h_batch_info_ms[j] - 1 - (int)max_index_ms + h_batch_info_ms[j+minibatch_size],minibatch_size) ] );
				//std::cout << "PREV Source: " << h_input_vocab_indicies_source[ IDX2C(j,h_batch_info[j] - 1 - (int)max_index + h_batch_info[j+minibatch_size],minibatch_size) ] << "\n";
				if(multi_attention_v2) {
					output_indicies_string += std::to_string(output_indicies[1].back()) + "-" + std::to_string(output_indicies[0].back()) + "-" + std::to_string(output_indicies[2].back()) + " ";
				}
				alignment_nums[j] += std::to_string(i) + "-" + std::to_string((int)max_index+1) + "-" + std::to_string((int)max_index_ms+1) + " ";
			}
		}
	}

	// std::cout << "SOURCE LENGTH: " << file_info->current_source_length << "\n";
	// for(int i=0; i<file_info->current_source_length; i++) {
	// 	std::cout << h_input_vocab_indicies_source[i] << " ";
	// }
	// std::cout << "\n";

	BZ_CUDA::logger << "----------------------Printing alignments (source-target)----------------------\n";
	for(int i=0; i<minibatch_size;i++) {
		BZ_CUDA::logger << alignment_nums[i] << "\n";
	}

	output_alignments << alignment_nums[0] << "\n";
	output_alignments << output_indicies_string << "\n";
	output_alignments.flush();
	// for(int i=0; i<output_indicies.size(); i++) {
	// 	for(int j=0; j< output_indicies[i].size(); j++) {
	// 		output_alignments << output_indicies[i][j] << " ";
	// 	}
	// 	output_alignments << "\n";
	// }

	free(h_p_t);
	free(h_batch_info);
	free(h_indicies);
	free(h_alignments);

	if(multi_attention_v2) {
		free(h_p_t_ms);
		free(h_batch_info_ms);
		free(h_indicies_ms);
		free(h_alignments_ms);
	}
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

	if(bi_dir || multi_source) {
		input_layer_source_bi.clear_gradients(false);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].clear_gradients(false);
		}

		if(bi_dir) {
			bi_dir_source.clear_gradients();
		}

		if(multi_source) {
			multi_source_layer.clear_gradients();
		}
	}
	devSynchAll();
}

template<typename dType>
double neuralMT_model<dType>::getError(bool GPU) 
{

	//std::cout << "---------------------------------GET ERROR STARTING---------------------------------\n";
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


	if(char_cnn) {
		input_layer_source.prep_char_cnn(file_info->fhc->h_char_vocab_indicies_source,file_info->current_source_length,
			file_info->fhc->h_unique_chars_source,file_info->fhc->num_unique_chars_source);
		input_layer_target.prep_char_cnn(file_info->fhc->h_char_vocab_indicies_target,file_info->current_target_length,
			file_info->fhc->h_unique_chars_target,file_info->fhc->num_unique_chars_target);
	}

	if(attent_params.attention_model) {
		if(target_hidden_layers.size()==0) {
			if(!multi_attention) {
				input_layer_target.attent_layer->prep_minibatch_info(file_info->h_batch_info);

				if(multi_attention_v2) {
					input_layer_target.attent_layer->prep_minibatch_info_v2(src_fh_test->h_batch_info);
				}
			}
			else {
				input_layer_target.attent_layer->prep_minibatch_info(file_info->h_batch_info);
				input_layer_target.attent_layer_bi->prep_minibatch_info(src_fh_test->h_batch_info);
			}
		}
		else {
			if(!multi_attention) {
				target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(file_info->h_batch_info);

				if(multi_attention_v2) {
					target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info_v2(src_fh_test->h_batch_info);
				}
			}
			else {
				target_hidden_layers[target_hidden_layers.size()-1].attent_layer->prep_minibatch_info(file_info->h_batch_info);
				target_hidden_layers[target_hidden_layers.size()-1].attent_layer_bi->prep_minibatch_info(src_fh_test->h_batch_info);
			}
		}
	}

	if(bi_dir) {
		bi_dir_source.longest_sent_minibatch = file_info->current_source_length;
		bi_dir_source.reverse_indicies(file_info->h_input_vocab_indicies_source,file_info->current_source_length);
		input_layer_source_bi.prep_GPU_vocab_indices(bi_dir_source.h_source_indicies,file_info->h_input_vocab_indicies_source_Wgrad,file_info->current_source_length,file_info->len_source_Wgrad);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].prep_GPU_vocab_indices(bi_dir_source.h_source_indicies,file_info->current_source_length);
		}
	}



	if(multi_source) {
		multi_source_layer.longest_sent_minibatch_s1 = file_info->current_source_length;
		multi_source_layer.longest_sent_minibatch_s2 = src_fh_test->current_source_length;
		//std::cout << "Current source length from s2: " << src_fh_test->current_source_length << "\n";
		input_layer_source_bi.prep_GPU_vocab_indices(src_fh_test->h_input_vocab_indicies_source,src_fh_test->h_input_vocab_indicies_source_Wgrad,src_fh_test->current_source_length,src_fh_test->len_source_Wgrad);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].prep_GPU_vocab_indices(src_fh_test->h_input_vocab_indicies_source,multi_source_layer.longest_sent_minibatch_s2);
		}
	}
	devSynchAll();
	CUDA_GET_LAST_ERROR("POST INDICES SETUP GETERROR");


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

		// std::cout << "Source index: " << 0 << "\n";
		// devSynchAll();
		// thrust::device_ptr<dType> thrust_d_h_t = thrust::device_pointer_cast(input_layer_source.nodes[0].d_h_t);
		// thrust::device_ptr<dType> thrust_d_h_t_2 = thrust::device_pointer_cast(source_hidden_layers[0].nodes[0].d_h_t);
		// std::cout << "top hidden state: " << thrust_d_h_t[0] << " , " << thrust_d_h_t[input_layer_source.nodes[0].LSTM_size-1] \
		// 	<< " " << thrust_d_h_t_2[0] << "\n\n";

		if(bi_dir) {
			input_layer_source_bi.nodes[0].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full,
				input_layer_source_bi.d_input_vocab_indices_01_full,
				input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector);
			input_layer_source_bi.nodes[0].forward_prop();

			//mgpu stuff
			for(int i=0; i<source_hidden_layers_bi.size(); i++) {
				source_hidden_layers_bi[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers_bi[i].d_input_vocab_indices_01_full,
					source_hidden_layers_bi[i].d_init_hidden_vector,source_hidden_layers_bi[i].d_init_cell_vector);
				source_hidden_layers_bi[i].nodes[0].forward_prop();
			}
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


			// std::cout << "Source index: " << i << "\n";
			// devSynchAll();
			// thrust::device_ptr<dType> thrust_d_h_t = thrust::device_pointer_cast(input_layer_source.nodes[i].d_h_t);
			// thrust::device_ptr<dType> thrust_d_h_t_2 = thrust::device_pointer_cast(source_hidden_layers[0].nodes[i].d_h_t);
			// std::cout << "top hidden state: " << thrust_d_h_t[0] << " , " << thrust_d_h_t[input_layer_source.nodes[i].LSTM_size-1] << \
			// 	" " << thrust_d_h_t_2[0] << "\n\n";

			if(bi_dir) {
				input_layer_source_bi.nodes[i].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full+step,
					input_layer_source_bi.d_input_vocab_indices_01_full+step,
					input_layer_source_bi.nodes[i-1].d_h_t,input_layer_source_bi.nodes[i-1].d_c_t);
				input_layer_source_bi.nodes[i].forward_prop();
				//cudaDeviceSynchronize();

				//mgpu stuff
				for(int j=0; j<source_hidden_layers_bi.size(); j++) {
					source_hidden_layers_bi[j].nodes[i].update_vectors_forward_GPU(
						source_hidden_layers_bi[j].d_input_vocab_indices_01_full+step,
						source_hidden_layers_bi[j].nodes[i-1].d_h_t,source_hidden_layers_bi[j].nodes[i-1].d_c_t);
					source_hidden_layers_bi[j].nodes[i].forward_prop();
				}
			}
		}
	}

	//do all the bi-directional layers below
	if(multi_source) {
		input_layer_source_bi.nodes[0].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full,
			input_layer_source_bi.d_input_vocab_indices_01_full,
			input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector);
		input_layer_source_bi.nodes[0].forward_prop();

		//mgpu stuff
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].nodes[0].update_vectors_forward_GPU(source_hidden_layers_bi[i].d_input_vocab_indices_01_full,
				source_hidden_layers_bi[i].d_init_hidden_vector,source_hidden_layers_bi[i].d_init_cell_vector);
			source_hidden_layers_bi[i].nodes[0].forward_prop();
		}

		for(int i=1; i<src_fh_test->current_source_length; i++) {
			int step = i*input_layer_source.minibatch_size;

			input_layer_source_bi.nodes[i].update_vectors_forward_GPU(input_layer_source_bi.d_input_vocab_indices_full+step,
				input_layer_source_bi.d_input_vocab_indices_01_full+step,
				input_layer_source_bi.nodes[i-1].d_h_t,input_layer_source_bi.nodes[i-1].d_c_t);
			input_layer_source_bi.nodes[i].forward_prop();
			//cudaDeviceSynchronize();

			//mgpu stuff
			for(int j=0; j<source_hidden_layers_bi.size(); j++) {
				source_hidden_layers_bi[j].nodes[i].update_vectors_forward_GPU(source_hidden_layers_bi[j].d_input_vocab_indices_01_full+step,
					source_hidden_layers_bi[j].nodes[i-1].d_h_t,source_hidden_layers_bi[j].nodes[i-1].d_c_t);
				source_hidden_layers_bi[j].nodes[i].forward_prop();
			}
		}
	}


	if(bi_dir) {
		devSynchAll();
		bi_dir_source.forward_prop();
		devSynchAll();
	}

	if(multi_source) {
		devSynchAll();
		multi_source_layer.forward_prop();
		devSynchAll();
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

		if(bi_dir && bi_dir_source.model_type == COMBINE) {

			//int prev_source_index = file_info->current_source_length-1;
			input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
				input_layer_target.d_input_vocab_indices_01_full,
				bi_dir_source.d_hs_final_target[0],bi_dir_source.d_ct_final_target[0]);

			//mgpu stuff
			for(int i=0; i<target_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(
					target_hidden_layers[i].d_input_vocab_indices_01_full,
					bi_dir_source.d_hs_final_target[i+1],bi_dir_source.d_ct_final_target[i+1]);
			}
		}
		else if(multi_source) {
			input_layer_target.nodes[0].update_vectors_forward_GPU(input_layer_target.d_input_vocab_indices_full,
				input_layer_target.d_input_vocab_indices_01_full,
				multi_source_layer.d_hs_final_target[0],multi_source_layer.d_ct_final_target[0]);

			//mgpu stuff
			for(int i=0; i<target_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_GPU(target_hidden_layers[i].d_input_vocab_indices_01_full,
					multi_source_layer.d_hs_final_target[i+1],multi_source_layer.d_ct_final_target[i+1]);
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
	}	

	//std::cout << "Target layer get error index: " << 0 << "\n";
	input_layer_target.nodes[0].forward_prop();

	//mgpu stuff
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();
	}


	devSynchAll();

	// std::cout << "Target index: " << 0 << "\n";
	// devSynchAll();
	// thrust::device_ptr<dType> thrust_d_h_t = thrust::device_pointer_cast(input_layer_target.nodes[0].d_h_t);
	// std::cout << "top hidden state: " << thrust_d_h_t[0] << " , " << thrust_d_h_t[input_layer_target.nodes[0].LSTM_size-1] << "\n\n";


	// std::cout << "Printing source hidden states\n";
	// for(int i=0; i<file_info->current_source_length; i++) {
	// 	std::cout << "Index: " << i << "   Vocab index: " << file_info->h_input_vocab_indicies_source[i] <<"\n";
	// 	print_src_hid_state(input_layer_source.nodes[i].d_h_t);
	// 	std::cout << "\n";
	// }
	//note d_h_t can be null for these as all we need is the vocab pointers correct for getting the error
	softmax->backprop_prep_GPU(input_layer_target.nodes[0].d_h_t,0);

	if(GPU) {
		loss += softmax->compute_loss_GPU(0);
	}
	else {
		BZ_CUDA::logger << "ERROR CAN ONLY USE GPU\n";
		exit (EXIT_FAILURE);
	}
	devSynchAll();


	//for(int i=1; i<file_info->minibatch_tokens_target_input.cols(); i++) {
	for(int i=1; i<file_info->current_target_length; i++) {
		int step = i*input_layer_target.minibatch_size;

		//std::cout << "Target layer get error index: " << i << "\n";

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
			BZ_CUDA::logger << "ERROR CAN ONLY USE GPU\n";
			exit (EXIT_FAILURE);
		}
		devSynchAll();
	}

	if(attent_params.dump_alignments) {
		dump_alignments(file_info->current_target_length,input_layer_target.minibatch_size,file_info->h_input_vocab_indicies_source,file_info->h_input_vocab_indicies_target,src_fh_test->h_input_vocab_indicies_source);
	}

	return loss;
}



template<typename dType>
void neuralMT_model<dType>::check_all_gradients(dType epsilon) 
{
	devSynchAll();
	if(!LM) {
		BZ_CUDA::logger << "------------------CHECKING GRADIENTS ON SOURCE SIDE------------------------\n";
		input_layer_source.check_all_gradients(epsilon);
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].check_all_gradients(epsilon);
		}
	}
	BZ_CUDA::logger << "------------------CHECKING GRADIENTS ON TARGET SIDE------------------------\n";
	input_layer_target.check_all_gradients(epsilon);
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].check_all_gradients(epsilon);
	}
	softmax->check_all_gradients(epsilon);


	if(bi_dir || multi_source) {
		BZ_CUDA::logger << "------------------CHECKING GRADIENTS ON SOURCE SIDE BI-DIR/MULTI SOURCE------------------------\n";
		input_layer_source_bi.check_all_gradients(epsilon);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].check_all_gradients(epsilon);
		}
		if(bi_dir) {
			bi_dir_source.check_all_gradients(epsilon);
		}
		if(multi_source) {
			multi_source_layer.check_all_gradients(epsilon);
		}
	}

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
			if(BZ_CUDA::print_norms) {
				BZ_CUDA::logger << "**************************SOURCE SIDE GRADIENTS**************************\n";
			}
			input_layer_source.calculate_global_norm();
			for(int i=0; i<source_hidden_layers.size(); i++) {
				source_hidden_layers[i].calculate_global_norm();
			}
		}
		if(BZ_CUDA::print_norms) {
			BZ_CUDA::logger << "**************************TARGET SIDE GRADIENTS**************************\n";
		}
		input_layer_target.calculate_global_norm();
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].calculate_global_norm();
		}

		if(bi_dir || multi_source) {
			if(BZ_CUDA::print_norms) {
				BZ_CUDA::logger << "**************************BI-DIR SOURCE SIDE GRADIENTS**************************\n";
			}
			input_layer_source_bi.calculate_global_norm();
			for(int i=0; i<source_hidden_layers_bi.size(); i++) {
				source_hidden_layers_bi[i].calculate_global_norm();
			}

			if(bi_dir) {
				bi_dir_source.calculate_global_norm();
			}

			if(multi_source) {
				multi_source_layer.calculate_global_norm();
			}
		}

		devSynchAll();

		BZ_CUDA::global_norm = std::sqrt(BZ_CUDA::global_norm);

		softmax->update_global_params();
		deniz::source_side = true;
		if(!LM) {
			input_layer_source.update_global_params();
			for(int i=0; i<source_hidden_layers.size(); i++) {
				source_hidden_layers[i].update_global_params();
			}
		}
		deniz::source_side = false;
		input_layer_target.update_global_params();
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].update_global_params();
		}

		if(bi_dir || multi_source) {
			input_layer_source_bi.update_global_params();
			for(int i=0; i<source_hidden_layers_bi.size(); i++) {
				source_hidden_layers_bi[i].update_global_params();
			}

			if(bi_dir) {
				bi_dir_source.update_global_params();
			}

			if(multi_source) {
				multi_source_layer.update_global_params();
			}
		}

		devSynchAll();
	}
	else {

		softmax->update_weights();
		deniz::source_side = true;
		if(!LM) {
			input_layer_source.update_weights();
			for(int i=0; i<source_hidden_layers.size(); i++) {
				source_hidden_layers[i].update_weights();
			}
		}
		deniz::source_side = false;
		input_layer_target.update_weights();
		for(int i=0; i<target_hidden_layers.size(); i++) {
			target_hidden_layers[i].update_weights();
		}

		if(bi_dir || multi_source) {
			input_layer_source_bi.update_weights();
			for(int i=0; i<source_hidden_layers_bi.size(); i++) {
				source_hidden_layers_bi[i].update_weights();
			}

			if(bi_dir) {
				bi_dir_source.update_weights();
			}

			if(multi_source) {
				multi_source_layer.update_weights();
			}
		}
	}

	devSynchAll();
	if(attent_params.attention_model) {
		if(source_hidden_layers.size()==0) {
			input_layer_source.zero_attent_error();
			if(bi_dir || multi_attention || multi_attention_v2) {
				input_layer_source_bi.zero_attent_error();
			}
		}
		else {
			source_hidden_layers[source_hidden_layers.size()-1].zero_attent_error();
			if(bi_dir || multi_attention || multi_attention_v2) {
				source_hidden_layers_bi[source_hidden_layers.size()-1].zero_attent_error();
			}
		}
	}

	devSynchAll();
}

//Update the model parameters
template<typename dType>
void neuralMT_model<dType>::update_weights_OLD() {

	// BZ_CUDA::global_norm = 0; //for global gradient clipping
	// devSynchAll();
	// //first calculate the global gradient sum
	// softmax->calculate_global_norm();
	// if(!LM) {
	// 	input_layer_source.calculate_global_norm();
	// 	for(int i=0; i<source_hidden_layers.size(); i++) {
	// 		source_hidden_layers[i].calculate_global_norm();
	// 	}
	// }
	// input_layer_target.calculate_global_norm();
	// for(int i=0; i<target_hidden_layers.size(); i++) {
	// 	target_hidden_layers[i].calculate_global_norm();
	// }

	// devSynchAll();

	// softmax->update_global_params();
	// if(!LM) {
	// 	input_layer_source.update_global_params();
	// 	for(int i=0; i<source_hidden_layers.size(); i++) {
	// 		source_hidden_layers[i].update_global_params();
	// 	}
	// }
	// input_layer_target.update_global_params();
	// for(int i=0; i<target_hidden_layers.size(); i++) {
	// 	target_hidden_layers[i].update_global_params();
	// }

	// devSynchAll();
	// //hidden_layer.update_weights();
}


template<typename dType>
void neuralMT_model<dType>::dump_weights() {
	
    if(BZ_CUDA::cont_train) {
        std::ifstream tmp_input(output_weight_file.c_str());
        tmp_input.clear();
        tmp_input.seekg(0, std::ios::beg);
    	std::string str;
        std::vector<std::string> beg_lines; 
    	std::getline(tmp_input, str);
        beg_lines.push_back(str);
    	std::getline(tmp_input, str);
        beg_lines.push_back(str);
    	while(std::getline(tmp_input, str)) {
            beg_lines.push_back(str);
    		if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
    				break; //done with source mapping
    		}
    	}
    	if(!LM) {
    		while(std::getline(tmp_input, str)) {
                beg_lines.push_back(str); 
    			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
    					break; //done with source mapping
    			}
    		}
    	}
        
        long pos = tmp_input.tellg();
       
        tmp_input.close();
        output.open(output_weight_file.c_str());
	    output.precision(std::numeric_limits<dType>::digits10 + 2);
        output.clear();
        output.seekp(0,std::ios_base::beg); //output.seekp(pos,std::ios_base::beg);
        for(int i=0; i<beg_lines.size(); i++) {
            output << beg_lines[i] << "\n";    
        }
    }
    else {
        output.open(output_weight_file.c_str(),std::ios_base::app);
	    output.precision(std::numeric_limits<dType>::digits10 + 2);
    }	

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

	if(bi_dir || multi_source) {
		input_layer_source_bi.dump_weights(output);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].dump_weights(output);
		}
		if(bi_dir) {
			bi_dir_source.dump_weights(output);
		}
		if(multi_source) {
			multi_source_layer.dump_weights(output);
		}
	}

	//output.flush();
	output.close();
	//output.flush();
}

template<typename dType>
void neuralMT_model<dType>::dump_best_model(std::string best_model_name,std::string const_model) {

	if(BZ_CUDA::dump_every_best) {
		best_model_name += "_save_all_models_"+std::to_string(BZ_CUDA::curr_dump_num)+".nn";
		BZ_CUDA::curr_dump_num += 1;
	}

    BZ_CUDA::logger << "Writing model file " << best_model_name  << "\n";

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

	if(bi_dir || multi_source) {
		input_layer_source_bi.dump_weights(best_model_stream);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].dump_weights(best_model_stream);
		}

		if(bi_dir) {
			bi_dir_source.dump_weights(best_model_stream);
		}

		if(multi_source) {
			multi_source_layer.dump_weights(best_model_stream);
		}
	}

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
		if(char_cnn && decode) {
			input_layer_source.load_weights_charCNN(input);
		}
		for(int i=0; i<source_hidden_layers.size(); i++) {
			source_hidden_layers[i].load_weights(input);
		}
	}
	//BZ_CUDA::logger << "--------------------------- Loading in Target Weights -----------------------------\n";
	input_layer_target.load_weights(input);
	if(attent_params.feed_input && decode) {
		input_layer_target.load_weights_decoder_feed_input(input);
	}
	if(char_cnn && decode) {
		input_layer_target.load_weights_charCNN(input);
	}
	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].load_weights(input);
	}

	if(decode) {
		if(attent_params.attention_model) {
			decoder_att_layer.load_weights(input);
		}
	}

	//input.sync();
	softmax->load_weights(input);

	if(bi_dir || multi_source) {
		input_layer_source_bi.load_weights(input);
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].load_weights(input);
		}

		if(bi_dir) {
			bi_dir_source.load_weights(input);
		}

		if(multi_source) {
			multi_source_layer.load_weights(input);
		}
	}

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

	if(bi_dir || multi_source) {
		input_layer_source_bi.learning_rate = new_learning_rate;
		for(int i=0; i<source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].learning_rate = new_learning_rate;
		}

		if(bi_dir) {
			bi_dir_source.learning_rate = new_learning_rate;
		}

		if(multi_source) {
			multi_source_layer.learning_rate = new_learning_rate;
		}
	}

	softmax->update_learning_rate(new_learning_rate);
}


template<typename dType>
double neuralMT_model<dType>::get_perplexity(std::string test_file_name,int minibatch_size,int &test_num_lines_in_file, int longest_sent,
	int source_vocab_size,int target_vocab_size,bool load_weights_val,int &test_total_words,bool HPC_output_flag,
	bool force_decode,std::string fd_filename) 
{

	if(load_weights_val) {
		load_weights();
	}
	//set trunc softmax to zero always for perplexity!
	file_helper file_info(test_file_name,minibatch_size,test_num_lines_in_file,longest_sent,
		source_vocab_size,target_vocab_size,test_total_words,false,0,0,char_params,char_params.char_dev_file); //Initialize the file information
	initFileInfo(&file_info);

	file_helper_source perp_fhs;
	if(multi_source) {
		perp_fhs.init_file_helper_source(multisource_file,minibatch_size,longest_sent,source_vocab_size);
		this->src_fh_test = &perp_fhs;
	}


	std::ofstream fd_stream;
	if(force_decode) {
		fd_stream.open(fd_filename);
	}


	int current_epoch = 1;
	BZ_CUDA::logger << "Getting perplexity of dev set\n";
	// if(HPC_output_flag) {
	// 	HPC_output << "Getting perplexity of dev set" << std::endl;
	// }
	//int total_words = 0; //For perplexity
	//double P_data = 0;
	double P_data_GPU = 0;
	int num_sents = 0; //for force decoding
	while(current_epoch <= 1) {
		bool success = file_info.read_minibatch();
		if(multi_source) {
			src_fh_test->read_minibatch();
		}
		num_sents+=file_info.minibatch_size;
		//P_data += getError(false);
		double temp = getError(true);
		fd_stream << temp << "\n";
		P_data_GPU += temp;
		//total_words += file_info.words_in_minibatch;
		if(!success) {
			current_epoch+=1;
		}

		if(BZ_CUDA::force_decode) {
			BZ_CUDA::logger << "Current sent: " << num_sents << "\n";
		}
	}

	//P_data = P_data/std::log(2.0); //Change to base 2 log
	P_data_GPU = P_data_GPU/std::log(2.0); 
	//double perplexity = std::pow(2,-1*P_data/file_info.num_target_words);
	double perplexity_GPU = std::pow(2,-1*P_data_GPU/file_info.total_target_words);
	BZ_CUDA::logger << "Total target words: " << file_info.total_target_words << "\n";
	//std::cout << "Perplexity CPU : " << perplexity << std::endl;
	BZ_CUDA::logger <<  std::setprecision(15) << "Perplexity dev set: " << perplexity_GPU << "\n";
	BZ_CUDA::logger <<  std::setprecision(15) << "P_data dev set: " << P_data_GPU << "\n";
	//fd_stream << perplexity_GPU << "\n";
	// if(HPC_output_flag) {
	// 	HPC_output <<  std::setprecision(15) << "P_data: " << P_data_GPU << std::endl;
	// 	HPC_output <<  std::setprecision(15) << "Perplexity dev set: " << perplexity_GPU << std::endl;
	// }

	if(BZ_CUDA::print_partition_function) {
		BZ_CUDA::print_partition_stats();
	}

	return perplexity_GPU;
}



template<typename dType>
void neuralMT_model<dType>::stoicastic_generation(int length,std::string output_file_name,double temperature) {

}


//for ensembles
template<typename dType>
void neuralMT_model<dType>::forward_prop_source(int *d_input_vocab_indicies_source,int *d_input_vocab_indicies_source_bi,int *d_ones,int source_length,int source_length_bi,int LSTM_size,
	int *d_char_cnn_indicies) 
{
	//std::cout << "##########################################Source index: " << 0 << "\n";

	//charcnn prep
	if(char_cnn) {
		input_layer_source.prep_char_cnn(d_char_cnn_indicies,source_length,
			NULL,0);
	}

	devSynchAll();
	cudaSetDevice(input_layer_target.ih_layer_info.device_number);
	input_layer_source.nodes[0].update_vectors_forward_GPU(d_input_vocab_indicies_source,d_ones,
		input_layer_source.d_init_hidden_vector,input_layer_source.d_init_cell_vector);
	input_layer_source.nodes[0].forward_prop();

	
	devSynchAll();

	for(int i=0; i < source_hidden_layers.size(); i++) {
		source_hidden_layers[i].nodes[0].update_vectors_forward_GPU(d_ones,
			source_hidden_layers[i].d_init_hidden_vector,source_hidden_layers[i].d_init_cell_vector);
		source_hidden_layers[i].nodes[0].forward_prop();
	}

	devSynchAll();
	if(attent_params.attention_model) {
		if(source_hidden_layers.size() == 0) {
			for(int i=0; i<input_layer_target.minibatch_size; i++) {
				CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states[0]+LSTM_size*i,input_layer_source.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
			}
		}
		else {
			for(int i=0; i<input_layer_target.minibatch_size; i++) {
				CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states[0]+LSTM_size*i,source_hidden_layers[source_hidden_layers.size()-1].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
			}
		}
	}

	devSynchAll();
	//thrust::device_ptr<dType> thrust_d_h_t = thrust::device_pointer_cast(input_layer_source.nodes[0].d_h_t);
	// thrust::device_ptr<dType> thrust_d_h_t_2 = thrust::device_pointer_cast(source_hidden_layers[0].nodes[0].d_h_t);
	// std::cout << "top hidden state: " << thrust_d_h_t[0] << " , " << thrust_d_h_t[input_layer_source.nodes[0].LSTM_size-1] \
	// 	<< " " << thrust_d_h_t_2[0] << "\n\n";
	for(int i = 1; i < source_length; i++) {
		//int step = i;
		//std::cout << "Source index: " << i << "\n";
		//copy the h_t and c_t to the previous hidden state of node 0
		CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[0].d_h_t_prev,input_layer_source.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s1\n");
		CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[0].d_c_t_prev,input_layer_source.nodes[0].d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s2\n");
		for(int j = 0; j < source_hidden_layers.size(); j++) {
			CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[j+1].d_h_t_prev,source_hidden_layers[j].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s3\n");
			CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states[j+1].d_c_t_prev,source_hidden_layers[j].nodes[0].d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s3\n");
		}

		input_layer_source.nodes[0].update_vectors_forward_GPU(d_input_vocab_indicies_source+i,d_ones,
			previous_source_states[0].d_h_t_prev,previous_source_states[0].d_c_t_prev);
		input_layer_source.nodes[0].forward_prop();

		for(int j=0; j < source_hidden_layers.size(); j++) {
			source_hidden_layers[j].nodes[0].update_vectors_forward_GPU(d_ones,
				previous_source_states[j+1].d_h_t_prev,previous_source_states[j+1].d_c_t_prev);
			source_hidden_layers[j].nodes[0].forward_prop();
		}

		devSynchAll();
		if(attent_params.attention_model) {
			if(source_hidden_layers.size() == 0) {
				for(int j=0; j<input_layer_target.minibatch_size; j++) {
					CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states[i]+j*LSTM_size,input_layer_source.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
				}
			}
			else {
				for(int j=0; j<input_layer_target.minibatch_size; j++) {
					CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states[i]+j*LSTM_size,source_hidden_layers[source_hidden_layers.size()-1].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
				}
			}
		}
		devSynchAll();
		// thrust::device_ptr<dType> thrust_d_h_t = thrust::device_pointer_cast(input_layer_source.nodes[0].d_h_t);
		// thrust::device_ptr<dType> thrust_d_h_t_2 = thrust::device_pointer_cast(source_hidden_layers[0].nodes[0].d_h_t);
		// std::cout << "top hidden state: " << thrust_d_h_t[0] << " , " << thrust_d_h_t[input_layer_source.nodes[0].LSTM_size-1] \
		// 	<< " " <<  thrust_d_h_t_2[0] << "\n\n";
	}
	devSynchAll();

	if(multi_source) {
		input_layer_source_bi.nodes[0].update_vectors_forward_GPU(d_input_vocab_indicies_source_bi,d_ones,
		input_layer_source_bi.d_init_hidden_vector,input_layer_source_bi.d_init_cell_vector);
		input_layer_source_bi.nodes[0].forward_prop();

		
		devSynchAll();

		for(int i=0; i < source_hidden_layers_bi.size(); i++) {
			source_hidden_layers_bi[i].nodes[0].update_vectors_forward_GPU(d_ones,
				source_hidden_layers_bi[i].d_init_hidden_vector,source_hidden_layers_bi[i].d_init_cell_vector);
			source_hidden_layers_bi[i].nodes[0].forward_prop();
		}

		devSynchAll();
		if(attent_params.attention_model) {
			if(source_hidden_layers_bi.size() == 0) {
				for(int i=0; i<input_layer_target.minibatch_size; i++) {
					CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states_v2[0]+LSTM_size*i,input_layer_source_bi.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
				}
			}
			else {
				for(int i=0; i<input_layer_target.minibatch_size; i++) {
					CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states_v2[0]+LSTM_size*i,source_hidden_layers_bi[source_hidden_layers_bi.size()-1].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
				}
			}
		}

		devSynchAll();
		for(int i = 1; i < source_length_bi; i++) {
			//int step = i;
			//copy the h_t and c_t to the previous hidden state of node 0
			CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states_bi[0].d_h_t_prev,input_layer_source_bi.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s1\n");
			CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states_bi[0].d_c_t_prev,input_layer_source_bi.nodes[0].d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s2\n");
			for(int j = 0; j < source_hidden_layers_bi.size(); j++) {
				CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states_bi[j+1].d_h_t_prev,source_hidden_layers_bi[j].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s3\n");
				CUDA_ERROR_WRAPPER(cudaMemcpy(previous_source_states_bi[j+1].d_c_t_prev,source_hidden_layers_bi[j].nodes[0].d_c_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU memory allocation failed s3\n");
			}

			input_layer_source_bi.nodes[0].update_vectors_forward_GPU(d_input_vocab_indicies_source_bi+i,d_ones,
				previous_source_states_bi[0].d_h_t_prev,previous_source_states_bi[0].d_c_t_prev);
			input_layer_source_bi.nodes[0].forward_prop();

			for(int j=0; j < source_hidden_layers_bi.size(); j++) {
				source_hidden_layers_bi[j].nodes[0].update_vectors_forward_GPU(d_ones,
					previous_source_states_bi[j+1].d_h_t_prev,previous_source_states_bi[j+1].d_c_t_prev);
				source_hidden_layers_bi[j].nodes[0].forward_prop();
			}

			devSynchAll();
			if(attent_params.attention_model) {
				if(source_hidden_layers_bi.size() == 0) {
					for(int j=0; j<input_layer_target.minibatch_size; j++) {
						CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states_v2[i]+j*LSTM_size,input_layer_source_bi.nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
					}
				}
				else {
					for(int j=0; j<input_layer_target.minibatch_size; j++) {
						CUDA_ERROR_WRAPPER(cudaMemcpy(top_source_states_v2[i]+j*LSTM_size,source_hidden_layers_bi[source_hidden_layers_bi.size()-1].nodes[0].d_h_t,LSTM_size*1*sizeof(dType),cudaMemcpyDeviceToDevice),"GPU fprop attention copy decoder source\n");
					}
				}
			}
			devSynchAll();

		}
		devSynchAll();

		//now combine h_t and c_t accordingly
		multi_source_layer.forward_prop();

		devSynchAll();
	}

	//now we can dump the hidden states for tsne
	if(BZ_STATS::tsne_dump) {

		if(BZ_STATS::h_dump_ht==NULL) {
			BZ_STATS::h_dump_ht = (dType *)malloc(LSTM_size*sizeof(dType));
		}

		for(int i=0; i<source_hidden_layers.size()+1; i++) {
			if(i==0) {
				cudaMemcpy(BZ_STATS::h_dump_ht,input_layer_source.nodes[0].d_h_t,LSTM_size*sizeof(dType),cudaMemcpyDeviceToHost);
			}
			else {
				cudaMemcpy(BZ_STATS::h_dump_ht,source_hidden_layers[i-1].nodes[0].d_h_t,LSTM_size*sizeof(dType),cudaMemcpyDeviceToHost);
			}
			for(int j=0; j<LSTM_size; j++) {
				BZ_STATS::tsne_dump_stream << BZ_STATS::h_dump_ht[j] << ",";
			}
		}
		BZ_STATS::tsne_dump_stream << "\n";
		for(int i=0; i<source_hidden_layers.size()+1; i++) {
			if(i==0) {
				cudaMemcpy(BZ_STATS::h_dump_ht,input_layer_source_bi.nodes[0].d_h_t,LSTM_size*sizeof(dType),cudaMemcpyDeviceToHost);
			}
			else {
				cudaMemcpy(BZ_STATS::h_dump_ht,source_hidden_layers_bi[i-1].nodes[0].d_h_t,LSTM_size*sizeof(dType),cudaMemcpyDeviceToHost);
			}
			for(int j=0; j<LSTM_size; j++) {
				BZ_STATS::tsne_dump_stream << BZ_STATS::h_dump_ht[j] << ",";
			}
		}
		BZ_STATS::tsne_dump_stream << "\n";

		//now dump the hidden vector from the combination step
		for(int i=0; i<source_hidden_layers.size()+1; i++) {
			cudaMemcpy(BZ_STATS::h_dump_ht,multi_source_layer.d_hs_final_target[i],LSTM_size*sizeof(dType),cudaMemcpyDeviceToHost);
			for(int j=0; j<LSTM_size; j++) {
				BZ_STATS::tsne_dump_stream << BZ_STATS::h_dump_ht[j] << ",";
			}
		}

		BZ_STATS::tsne_dump_stream << "\n";
	}
}


template<typename dType>
void neuralMT_model<dType>::forward_prop_target(int curr_index,int *d_current_indicies,int *d_ones,int LSTM_size, int beam_size,
	int *d_char_cnn_indicies) {

	//charcnn prep
	if(char_cnn) {
		input_layer_target.prep_char_cnn(d_char_cnn_indicies,1,
			NULL,0);
	}

	input_layer_target.nodes[0].index = curr_index;

	//source_hidden_layers[1].nodes[0].debug_operation();
	//std::cout << "Current index target: " << curr_index << "\n";

	int num_layers = 1+ target_hidden_layers.size();
	cudaSetDevice(input_layer_target.ih_layer_info.device_number);
	if(curr_index==0) {

		// if(attent_params.feed_input) {
		// 	cudaMemset(input_layer_target.nodes[0].d_h_tild,0,input_layer_target.LSTM_size*input_layer_target.minibatch_size*sizeof(dType));
		// }

		if(multi_source) {
			input_layer_target.transfer_decoding_states_GPU(multi_source_layer.d_hs_final_target[0],multi_source_layer.d_ct_final_target[0]);
			for(int i=0; i<source_hidden_layers.size(); i++) {
				target_hidden_layers[i].transfer_decoding_states_GPU(multi_source_layer.d_hs_final_target[i+1],multi_source_layer.d_ct_final_target[i+1]);
			}
			input_layer_target.nodes[0].update_vectors_forward_decoder(d_current_indicies,d_ones);
			for(int i=0; i<source_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_decoder(d_ones);
			}
		}
		else {
			input_layer_target.transfer_decoding_states_GPU(input_layer_source.nodes[0].d_h_t,input_layer_source.nodes[0].d_c_t);
			for(int i=0; i<source_hidden_layers.size(); i++) {
				target_hidden_layers[i].transfer_decoding_states_GPU(source_hidden_layers[i].nodes[0].d_h_t,source_hidden_layers[i].nodes[0].d_c_t);
			}
			input_layer_target.nodes[0].update_vectors_forward_decoder(d_current_indicies,d_ones);
			for(int i=0; i<source_hidden_layers.size(); i++) {
				target_hidden_layers[i].nodes[0].update_vectors_forward_decoder(d_ones);
			}
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
	devSynchAll();

	// if(curr_index==0) {
	// 	thrust::device_ptr<dType> thrust_d_h_t = thrust::device_pointer_cast(input_layer_target.nodes[0].d_h_t);
	// 	std::cout << "Target index: 0 \n";
	// 	std::cout << "top hidden state: " << thrust_d_h_t[0] << " , " << thrust_d_h_t[input_layer_target.nodes[0].LSTM_size-1] << "\n\n";
	// }

	for(int i=0; i<target_hidden_layers.size(); i++) {
		target_hidden_layers[i].nodes[0].forward_prop();
	}
	devSynchAll();

	//now attention stuff
	if(attent_params.attention_model) {
		if(num_layers==1) {
			decoder_att_layer.nodes[0].d_h_t = input_layer_target.nodes[0].d_h_t;
		}
		else {
			decoder_att_layer.nodes[0].d_h_t = target_hidden_layers[target_hidden_layers.size()-1].nodes[0].d_h_t;
		}
		decoder_att_layer.nodes[0].forward_prop();
		devSynchAll();
	}

	if(attent_params.attention_model) {
		softmax->backprop_prep_GPU(decoder_att_layer.nodes[0].d_final_temp_2,0);
	}
	else if(num_layers==1) {
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

	// devSynchAll();
	// std::cout << " ENTERING swap decoding states\n";
	// CUDA_GET_LAST_ERROR("ENTERING SWAP DECODING STATES");

	if(attent_params.feed_input) {
		for(int i=0; i<input_layer_target.minibatch_size; i++) {
			cudaMemcpy(input_layer_target.nodes[0].d_h_tild + i*input_layer_target.LSTM_size,decoder_att_layer.nodes[0].d_final_temp_2 + indicies(i)*input_layer_target.LSTM_size,input_layer_target.LSTM_size*sizeof(dType),cudaMemcpyDeviceToDevice);
		}
	}

	devSynchAll();

	// devSynchAll();
	// std::cout << " Finished swap decoding states\n";
	// CUDA_GET_LAST_ERROR("SWAP DECODING STATES");
}






