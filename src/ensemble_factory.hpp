
template<typename dType>
ensemble_factory<dType>::ensemble_factory(std::vector<std::string> weight_file_names,int num_hypotheses,int beam_size, dType min_decoding_ratio,\
	dType penalty, int longest_sent,int &num_lines_in_file,bool print_score,std::string decoder_output_file,std::string input_file_name,
	std::vector<int> gpu_nums,dType max_decoding_ratio, int target_vocab_size,bool dump_LSTM,std::string LSTM_stream_dump_name,global_params &params)
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

	fileh = new file_helper_decoder(input_file_name,num_lines_in_file,longest_sent);
	this->num_lines_in_file = num_lines_in_file;
	model_decoder = new decoder<dType>(beam_size,target_vocab_size,start_symbol,end_symbol,longest_sent,min_decoding_ratio,
		penalty,decoder_output_file,num_hypotheses,print_score);
	
	//initialize all of the models
	for(int i=0; i < weight_file_names.size(); i++) {
		models.push_back( decoder_model_wrapper<dType>(gpu_nums[i],beam_size,weight_file_names[i],longest_sent,dump_LSTM,LSTM_stream_dump_name,params) );
	}

	//check to be sure all models have the same target vocab size and vocab indicies and get the target vocab size
	this->target_vocab_size = models[0].target_vocab_size;
	for(int i=0; i< models.size(); i++) {
		if(models[0].target_vocab_size != this->target_vocab_size) {
			std::cout << "ERROR: The target vocabulary sizes are not all the same for the models you wanted in your ensemble\n";
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
		std::cout << "Decoding sentence: " << i << " out of " << num_lines_in_file << "\n";
		fileh->read_sentence(); //read sentence from file

		//copy the indicies all the models on the gpu
		for(int j=0; j < models.size(); j++) {
			models[j].memcpy_vocab_indicies(fileh->h_input_vocab_indicies_source,fileh->sentence_length);
		}

		//init decoder
		model_decoder->init_decoder();
		//run forward prop on the source
		for(int j=0; j < models.size(); j++) {
			models[j].forward_prop_source();
		}
		int last_index = 0;

		//run the forward prop of target
		for(int curr_index=0; curr_index < std::min( (int)(max_decoding_ratio*fileh->sentence_length) , longest_sent-2 ); curr_index++) {
			
			for(int j=0; j < models.size(); j++) {
				models[j].forward_prop_target(curr_index,model_decoder->h_current_indices);
			}

			//now ensemble the models together
			ensembles_models();

			//run decoder for this iteration
			model_decoder->expand_hypothesis(outputdist,curr_index);

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
		model_decoder->finish_current_hypotheses(outputdist);
		model_decoder->output_k_best_hypotheses(fileh->sentence_length);
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
}

