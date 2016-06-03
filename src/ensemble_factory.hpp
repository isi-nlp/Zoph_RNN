
template<typename dType>
ensemble_factory<dType>::ensemble_factory(std::vector<std::string> weight_file_names,int num_hypotheses,int beam_size, dType min_decoding_ratio,\
	dType penalty, int longest_sent,bool print_score,std::string decoder_output_file,
	std::vector<int> gpu_nums,dType max_decoding_ratio, int target_vocab_size,global_params &params)
{

	//get the target vocab from the first file
	this->target_vocab_size = target_vocab_size;
	this->max_decoding_ratio = max_decoding_ratio;
	this->longest_sent = longest_sent;


	//std::ofstream &LSTM_stream_dump

	//to make sure beam search does halt
	if(beam_size > (int)std::sqrt(target_vocab_size) ) {
		beam_size = (int)std::sqrt(target_vocab_size);
	}

	//fileh = new file_helper_decoder(input_file_name,num_lines_in_file,longest_sent);
	std::ifstream temp_input;
	temp_input.open(params.decode_temp_files[0]);
	get_file_stats_source(num_lines_in_file,temp_input);
	temp_input.close();

	model_decoder = new decoder<dType>(beam_size,target_vocab_size,start_symbol,end_symbol,longest_sent,min_decoding_ratio,
		penalty,decoder_output_file,num_hypotheses,print_score);
	
	//initialize all of the models
	for(int i=0; i < weight_file_names.size(); i++) {
		models.push_back( decoder_model_wrapper<dType>(gpu_nums[i],beam_size,params.model_names[i],params.model_names_multi_src[i],
			params.decode_temp_files[i],params.decode_temp_files_additional[i],longest_sent,params));
	}

	//check to be sure all models have the same target vocab size and vocab indicies and get the target vocab size
	this->target_vocab_size = models[0].target_vocab_size;
	for(int i=0; i< models.size(); i++) {
		if(models[0].target_vocab_size != target_vocab_size) {
			BZ_CUDA::logger << "ERROR: The target vocabulary sizes are not all the same for the models you wanted in your ensemble\n";
			exit (EXIT_FAILURE);
		}
	}

	//resise the outputdist that gets sent to the decoder
	outputdist.resize(target_vocab_size,beam_size);
	normalization.resize(1,beam_size);
}



template<typename dType>
void ensemble_factory<dType>::decode_file() {
	
	for(int i = 0; i < num_lines_in_file; i++) {
		BZ_CUDA::logger << "Decoding sentence: " << i << " out of " << num_lines_in_file << "\n";
		//fileh->read_sentence(); //read sentence from file

		//copy the indicies all the models on the gpu
		//in memcpy_vocab_indicies
		for(int j=0; j < models.size(); j++) {
			models[j].memcpy_vocab_indicies();
		}

		devSynchAll();

		//init decoder
		model_decoder->init_decoder();
		//run forward prop on the source
		for(int j=0; j < models.size(); j++) {
			models[j].forward_prop_source();
		}
		int last_index = 0;

		//for dumping hidden states we can just return
		if(BZ_STATS::tsne_dump) {
			continue;
		}

		//run the forward prop of target
		//BZ_CUDA::logger << "Source length bi: " << models[0].source_length_bi << "\n";
		int source_length = std::max(models[0].source_length,models[0].source_length_bi);
		for(int curr_index=0; curr_index < std::min( (int)(max_decoding_ratio*source_length) , longest_sent-2 ); curr_index++) {
			
			for(int j=0; j < models.size(); j++) {
				models[j].forward_prop_target(curr_index,model_decoder->h_current_indices);
				//now take the viterbi alignments
			}

			//now ensemble the models together
			//this also does voting for unk-replacement
		//	BZ_CUDA::logger << "Source length: " << source_length << "\n";
			ensembles_models();

			//run decoder for this iteration
			model_decoder->expand_hypothesis(outputdist,curr_index,BZ_CUDA::viterbi_alignments);

			//swap the decoding states
			for(int j=0; j<models.size(); j++) {
				models[j].swap_decoding_states(model_decoder->new_indicies_changes,curr_index);
				models[j].target_copy_prev_states();
			}

			//for the scores of the last hypothesis
			last_index = curr_index;
		}

		//now run one last iteration
		for(int j=0; j < models.size(); j++) {
			models[j].forward_prop_target(last_index+1,model_decoder->h_current_indices);
		}
		//output the final results of the decoder
		ensembles_models();
		model_decoder->finish_current_hypotheses(outputdist,BZ_CUDA::viterbi_alignments);
		model_decoder->output_k_best_hypotheses(source_length);
		//model_decoder->print_current_hypotheses();
	}

}

template<typename dType>
void ensemble_factory<dType>::ensembles_models() {

	int num_models = models.size();
	for(int i=0; i<outputdist.rows(); i++) {
		for(int j=0; j< outputdist.cols(); j++) {
			double temp_sum = 0;
			for(int k=0; k<models.size(); k++) {
				temp_sum+=models[k].outputdist(i,j);
			}
			outputdist(i,j) = temp_sum/num_models;
		}
	}

	//normalize now
	// for(int j=0; j< outputdist.cols(); j++) {
	// 	double temp_sum = 0;
	// 	for(int i=0; i<outputdist.rows(); i++) {
	// 		temp_sum+=outputdist(i,j);
	// 	}
	// 	for(int i=0; i<outputdist.rows(); i++) {
	// 		outputdist(i,j) = outputdist(i,j)/temp_sum;
	// 	}
	// }
	normalization.setZero();

	for(int i=0; i<outputdist.rows(); i++) {
		normalization+=outputdist.row(i);
	}
	for(int i=0; i<outputdist.rows(); i++) {
		outputdist.row(i) = (outputdist.row(i).array()/normalization.array()).matrix();
	}

	//now averaging alignment scores for unk replacement
	if(BZ_CUDA::unk_replacement) {
		//average the scores
		for(int i=0; i<models[0].longest_sent;i++) {
			for(int j=0; j<models[0].beam_size; j++) {
				dType temp_sum = 0;
				for(int k=0; k<models.size(); k++) {
					temp_sum+=models[k].viterbi_alignments_scores[IDX2C(i,j,models[0].longest_sent)];
				}
				BZ_CUDA::alignment_scores[IDX2C(i,j,models[0].longest_sent)] = temp_sum;
			}
		}

		// std::cout << "-------------------------------------------\n";
		// for(int i=0; i<models[0].longest_sent;i++) {
		// 	for(int j=0; j<models[0].beam_size; j++) {
		// 		std::cout << BZ_CUDA::alignment_scores[IDX2C(i,j,models[0].longest_sent)] << " ";
		// 	}
		// 	std::cout << "\n";
		// }
		// std::cout << "\n";
		// std::cout << "-------------------------------------------\n\n";
		//choose the max and fill in BZ_CUDA::viterbi_alignments
		for(int i=0; i<models[0].beam_size; i++) {
			dType max_val = 0;
			int max_index = -1;
			for(int j=0; j<models[0].longest_sent; j++) {
				dType temp_val = BZ_CUDA::alignment_scores[IDX2C(j,i,models[0].longest_sent)];
				if(temp_val > max_val) {
					max_val = temp_val;
					max_index = j;
				}
			}
			// if(max_index==-1) {
			// 	std::cout << "ERROR: max_index is still -1, so all values are zero\n";
			// }
			BZ_CUDA::viterbi_alignments[i] = max_index;
		}
	}
}

