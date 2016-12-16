//additional ugly file stuff

bool input_file_prep::prep_files_train_nonLM_multi_source_ensemble(int minibatch_size,int max_sent_cutoff,
		std::string source_file_name,std::string target_file_name,
		std::string output_file_name,int &source_vocab_size,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,
		int num_layers,std::string source_file_name_2,std::string output_file_name_2,
		std::string model_output_file_name_2,std::string ensemble_model_name_big,std::string ensemble_model_name_small) 
{


		target_input.open(target_file_name.c_str());
		final_output.open(output_file_name.c_str());
		final_output_2.open(output_file_name_2.c_str());
		source_input.open(source_file_name.c_str());
		source_input_2.open(source_file_name_2.c_str());

		std::vector<comb_sent_info_ms> data; //can be used to sort by mult of minibatch

		//first stage is load all data into RAM
		std::string src_str;
		std::string src_str_2;
		std::string tgt_str; 
		std::string word;

		int source_len = 0;
		int source_len_2 = 0;
		int target_len = 0;

		source_input.clear();
		source_input_2.clear();
		target_input.clear();

		source_input.seekg(0, std::ios::beg);
		while(std::getline(source_input, src_str)) {
			source_len++;
		}

		source_input_2.seekg(0, std::ios::beg);
		while(std::getline(source_input_2, src_str_2)) {
			source_len_2++;
		}

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}

		//do check to be sure the two files are the same length
		if(source_len!=target_len || source_len_2!=source_len) {
			BZ_CUDA::logger << "ERROR: Input files are not the same length\n";
			return false;
			exit (EXIT_FAILURE);
		}

		if(minibatch_size>source_len) {
			BZ_CUDA::logger << "ERROR: minibatch size cannot be greater than the file size\n";
			return false;
			exit (EXIT_FAILURE);
		}


		//filter any long sentences and get ready to shuffle
		source_input.clear();
		source_input_2.clear();
		target_input.clear();
		source_input.seekg(0, std::ios::beg);
		source_input_2.seekg(0, std::ios::beg);
		target_input.seekg(0, std::ios::beg);
		for(int i=0; i<source_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> src_sentence_2;
			std::vector<std::string> tgt_sentence;
			std::getline(source_input, src_str);
			std::getline(source_input_2, src_str_2);
			std::getline(target_input, tgt_str);

			std::istringstream iss_src(src_str, std::istringstream::in);
			std::istringstream iss_src_2(src_str_2, std::istringstream::in);
			std::istringstream iss_tgt(tgt_str, std::istringstream::in);
			while(iss_src >> word) {
				src_sentence.push_back(word);
			}

			while(iss_src_2 >> word) {
				src_sentence_2.push_back(word);
			}

			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2 || src_sentence_2.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info_ms(src_sentence,src_sentence_2,tgt_sentence));
			}
		}
		
		//shuffle the entire data
		if(shuffle) {
			std::random_shuffle(data.begin(),data.end());
		}

		//remove last sentences that do not fit in the minibatch
		if(data.size()%minibatch_size!=0) {
			int num_to_remove = data.size()%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}

		if(data.size()==0) {
			BZ_CUDA::logger << "ERROR: file size is zero, could be wrong input file or all lines are above max sent length\n";
			return false;
			exit (EXIT_FAILURE);
		}

		//sort the data based on minibatch
		compare_nonLM_multisrc comp;
		int curr_index = 0;
		while(curr_index<data.size()) {
			if(curr_index+minibatch_size*minibatch_mult <= data.size()) {
				std::sort(data.begin()+curr_index,data.begin()+curr_index+minibatch_size*minibatch_mult,comp);
				curr_index+=minibatch_size*minibatch_mult;
			}
			else {
				std::sort(data.begin()+curr_index,data.end(),comp);
				break;
			}
		}


		std::ifstream ensemble_file_big;
		ensemble_file_big.open(ensemble_model_name_big.c_str());

		std::ifstream ensemble_file_small;
		ensemble_file_small.open(ensemble_model_name_small.c_str());

		std::vector<std::string> file_input_vec;
		std::string str;

		std::getline(ensemble_file_big, str);
		std::istringstream iss(str, std::istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

		// if(file_input_vec.size()!=4) {
		// 	BZ_CUDA::logger << "ERROR: Neural network file format has been corrupted\n";
		// 	//exit (EXIT_FAILURE);
		// }

		target_vocab_size = std::stoi(file_input_vec[2]);
		source_vocab_size = std::stoi(file_input_vec[3]);
		

		//output the model info to first line of output weights file
		std::ofstream output_model;
		std::ofstream output_model_2;
		output_model.open(model_output_file_name.c_str());
		output_model_2.open(model_output_file_name_2.c_str());
		output_model << num_layers << " " << hiddenstate_size << " " << target_vocab_size << " " << source_vocab_size << "\n";
		output_model << "==========================================================\n";
		
		//now get the mappings
		std::getline(ensemble_file_big, str); //get this line, since all equals
		while(std::getline(ensemble_file_big, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			src_mapping[word] = tmp_index;
			output_model << tmp_index << " " << word << "\n";
		}

		output_model << "==========================================================\n";
		while(std::getline(ensemble_file_big, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			tgt_mapping[word] = tmp_index;
			output_model << tmp_index << " " << word << "\n";
		}

		output_model << "==========================================================\n";
		ensemble_file_big.close();
		

		output_model_2 << "==========================================================\n";
		std::getline(ensemble_file_small, str); //get this line, since all equals
		while(std::getline(ensemble_file_small, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			src_mapping_2[word] = tmp_index;
			output_model_2 << tmp_index << " " << word << "\n";
		}
		output_model_2 << "==========================================================\n";
		ensemble_file_small.close();

		output_model.flush();
		output_model_2.flush();

		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;
			std::vector<int> src_int_2;
			std::vector<int> tgt_int;
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_mapping.count(data[i].src_sent[j])==0) {
					src_int.push_back(src_mapping["<UNK>"]);
				}
				else {
					src_int.push_back(src_mapping[data[i].src_sent[j]]);
				}	
			}
			std::reverse(src_int.begin(), src_int.end());
			data[i].src_sent.clear();
			data[i].src_sent_int = src_int;

			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
			}


			//for second source
			for(int j=0; j<data[i].src_sent_2.size(); j++) {
				if(src_mapping_2.count(data[i].src_sent_2[j])==0) {
					src_int_2.push_back(src_mapping_2["<UNK>"]);
				}
				else {
					src_int_2.push_back(src_mapping_2[data[i].src_sent_2[j]]);
				}	
			}
			std::reverse(src_int_2.begin(), src_int_2.end());
			data[i].src_sent_2.clear();
			data[i].src_sent_int_2 = src_int_2;

			while(data[i].minus_two_source_2.size()!=data[i].src_sent_int_2.size()) {
				data[i].minus_two_source_2.push_back(-2);
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
					
					if(tgt_mapping.count("<UNK>")==0) {
						tgt_int.push_back(tgt_mapping["<UNK>NULL"]);
					}
					else {
						tgt_int.push_back(tgt_mapping["<UNK>"]);
					}
				}
				else {
					tgt_int.push_back(tgt_mapping[data[i].tgt_sent[j]]);
				}	
			}
			data[i].tgt_sent.clear();
			data[i].tgt_sent_int_i = tgt_int;
			data[i].tgt_sent_int_o = tgt_int;
			data[i].tgt_sent_int_i.insert(data[i].tgt_sent_int_i.begin(),0);
			data[i].tgt_sent_int_o.push_back(1);

		}

		//now pad based on minibatch
		curr_index = 0;
		while(curr_index < data.size()) {
			int max_source_minibatch=0;
			int max_source_minibatch_2=0;
			int max_target_minibatch=0;

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				if(data[i].src_sent_int.size()>max_source_minibatch) {
					max_source_minibatch = data[i].src_sent_int.size();
				}

				if(data[i].src_sent_int_2.size()>max_source_minibatch_2) {
					max_source_minibatch_2 = data[i].src_sent_int_2.size();
				}

				if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
					max_target_minibatch = data[i].tgt_sent_int_i.size();
				}
			}


			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {

				while(data[i].src_sent_int.size()<max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
				}

				while(data[i].src_sent_int_2.size()<max_source_minibatch_2) {
					data[i].src_sent_int_2.insert(data[i].src_sent_int_2.begin(),-1);
					data[i].minus_two_source_2.insert(data[i].minus_two_source_2.begin(),-1);
				}

				while(data[i].tgt_sent_int_i.size()<max_target_minibatch) {
					data[i].tgt_sent_int_i.push_back(-1);
					data[i].tgt_sent_int_o.push_back(-1);
				}
			}
			curr_index+=minibatch_size;
		}

		//now output to the file
		for(int i=0; i<data.size(); i++) {

			for(int j=0; j<data[i].src_sent_int.size(); j++) {
				final_output << data[i].src_sent_int[j];
				if(j!=data[i].src_sent_int.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int j=0; j<data[i].minus_two_source.size(); j++) {
				final_output << data[i].minus_two_source[j];
				if(j!=data[i].minus_two_source.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int j=0; j<data[i].src_sent_int_2.size(); j++) {
				final_output_2 << data[i].src_sent_int_2[j];
				if(j!=data[i].src_sent_int_2.size()) {
					final_output_2 << " ";
				}
			}
			final_output_2 << "\n";

			for(int j=0; j<data[i].minus_two_source_2.size(); j++) {
				final_output_2 << data[i].minus_two_source_2[j];
				if(j!=data[i].minus_two_source_2.size()) {
					final_output_2 << " ";
				}
			}
			final_output_2 << "\n";

			for(int j=0; j<data[i].tgt_sent_int_i.size(); j++) {
				final_output << data[i].tgt_sent_int_i[j];
				if(j!=data[i].tgt_sent_int_i.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";


			for(int j=0; j<data[i].tgt_sent_int_o.size(); j++) {
				final_output << data[i].tgt_sent_int_o[j];
				if(j!=data[i].tgt_sent_int_o.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";
		}

		final_output.close();
		source_input.close();
		target_input.close();	

		return true;	
}



