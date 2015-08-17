#ifndef INPUT_FILE_PREP_H
#define INPUT_FILE_PREP_H

#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <algorithm>
#include <queue>


struct comb_sent_info {

	std::vector<std::string> src_sent;
	std::vector<std::string> tgt_sent;

	std::vector<int> src_sent_int;
	std::vector<int> minus_two_source;
	std::vector<int> tgt_sent_int_i;
	std::vector<int> tgt_sent_int_o;
	int total_len;

	comb_sent_info(std::vector<std::string> &src_sent,std::vector<std::string> &tgt_sent) {
		this->src_sent = src_sent;
		this->tgt_sent = tgt_sent;
		total_len = tgt_sent.size() + src_sent.size();
	}
};

struct compare_nonLM {
    bool operator()(const struct comb_sent_info& first, const struct comb_sent_info& second) {
        return first.total_len < second.total_len;
    }
};

struct mapping_pair {
	std::string word;
	int count;
	mapping_pair(std::string word,int count) {
		this->word = word;
		this->count = count;
	}
};

struct mapping_pair_compare_functor {
  	bool operator() (mapping_pair &a,mapping_pair &b) const { return (a.count < b.count); }
};


//this will unk based on the source and target vocabulary
struct input_file_prep {

	std::ifstream source_input;
	std::ifstream target_input;
	std::ofstream final_output;

	std::unordered_map<std::string,int> src_mapping;
	std::unordered_map<std::string,int> tgt_mapping;

	std::unordered_map<int,std::string> tgt_reverse_mapping;

	std::unordered_map<std::string,int> src_counts;
	std::unordered_map<std::string,int> tgt_counts;

	int minibatch_mult = 10; //montreal uses 20
	std::vector<comb_sent_info> data; //can be used to sort by mult of minibatch


	void prep_files_train_nonLM(int minibatch_size,int max_sent_cutoff,
		std::string source_file_name,std::string target_file_name,
		std::string output_file_name,int &source_vocab_size,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size) 
	{

		target_input.open(target_file_name.c_str());
		final_output.open(output_file_name.c_str());
		source_input.open(source_file_name.c_str());

		//first stage is load all data into RAM
		std::string src_str;
		std::string tgt_str; 
		std::string word;

		int source_len = 0;
		int target_len = 0;

		source_input.clear();
		target_input.clear();

		source_input.seekg(0, std::ios::beg);
		while(std::getline(source_input, src_str)) {
			source_len++;
		}

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}


		//do check to be sure the two files are the same length
		if(source_len!=target_len) {
			std::cerr << "ERROR: Input files are not the same length\n";
			exit (EXIT_FAILURE);
		}

		if(minibatch_size>source_len) {
			std::cerr << "ERROR: minibatch size cannot be greater than the file size\n";
			exit (EXIT_FAILURE);
		}


		//filter any long sentences and get ready to shuffle
		source_input.clear();
		target_input.clear();
		source_input.seekg(0, std::ios::beg);
		target_input.seekg(0, std::ios::beg);
		for(int i=0; i<source_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			std::getline(source_input, src_str);
			std::getline(target_input, tgt_str);

			std::istringstream iss_src(src_str, std::istringstream::in);
			std::istringstream iss_tgt(tgt_str, std::istringstream::in);
			while(iss_src >> word) {
				src_sentence.push_back(word);
			}
			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if( !(src_sentence.size()>=max_sent_cutoff || tgt_sentence.size()>=max_sent_cutoff) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
			}
		}


		//shuffle the entire data
		if(shuffle) {
			std::random_shuffle(data.begin(),data.end());
		}


		//remove last sentences that do not fit in the minibatch
		if(source_len%minibatch_size!=0) {
			int num_to_remove = source_len%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}

		//sort the data based on minibatch
		compare_nonLM comp;
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


		//now get counts for mappings
		for(int i=0; i<data.size(); i++) {
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_counts.count(data[i].src_sent[j])==0) {
					src_counts[data[i].src_sent[j]] = 1;
				}
				else {
					src_counts[data[i].src_sent[j]]+=1;
				}
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_counts.count(data[i].tgt_sent[j])==0) {
					tgt_counts[data[i].tgt_sent[j]] = 1;
				}
				else {
					tgt_counts[data[i].tgt_sent[j]]+=1;
				}
			}
		}


		//now use heap to get the highest source and target mappings
		if(source_vocab_size==-1) {
			source_vocab_size = src_counts.size();
		}
		if(target_vocab_size==-1) {
			target_vocab_size = tgt_counts.size();
		}


		source_vocab_size = std::min(source_vocab_size,(int)src_counts.size());
		target_vocab_size = std::min(target_vocab_size,(int)tgt_counts.size());

		//output the model info to first line of output weights file
		std::ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << 0 << " " << hiddenstate_size << " " << target_vocab_size << " " << source_vocab_size << "\n";


		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> src_map_heap;
		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> tgt_map_heap;

		for ( auto it = src_counts.begin(); it != src_counts.end(); ++it ) {
			src_map_heap.push( mapping_pair(it->first,it->second) );
		}

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			tgt_map_heap.push( mapping_pair(it->first,it->second) );
		}

		output_model << "==========================================================\n";
		src_mapping["<START>"] = 0;
		output_model << 0 << " " << "<START>" << "\n";
		for(int i=1; i<source_vocab_size-1; i++) {
			src_mapping[src_map_heap.top().word] = i;
			output_model << i << " " << src_map_heap.top().word << "\n";
			src_map_heap.pop();
		}
		src_mapping["<UNK>"] = source_vocab_size-1;
		output_model << source_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model << "==========================================================\n";

		tgt_mapping["<START>"] = 0;
		tgt_mapping["<EOF>"] = 1;
		output_model << 0 << " " << "<START>" << "\n";
		output_model << 1 << " " << "<EOF>" << "\n";
		for(int i=2; i<target_vocab_size-1; i++) {
			tgt_mapping[tgt_map_heap.top().word] = i;
			output_model << i << " " << tgt_map_heap.top().word << "\n";
			tgt_map_heap.pop();
		}
		tgt_mapping["<UNK>"] = target_vocab_size-1;
		output_model << target_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model << "==========================================================\n";

		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;
			std::vector<int> tgt_int;
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_mapping.count(data[i].src_sent[j])==0) {
					src_int.push_back(source_vocab_size-1);
				}
				else {
					src_int.push_back(src_mapping[data[i].src_sent[j]]);
				}	
			}
			std::reverse(src_int.begin(), src_int.end());
			data[i].src_sent.clear();
			data[i].src_sent_int = src_int;
			data[i].src_sent_int.insert(data[i].src_sent_int.begin(),0);
			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
					tgt_int.push_back(target_vocab_size-1);
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
			int max_target_minibatch=0;

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				if(data[i].src_sent_int.size()>max_source_minibatch) {
					max_source_minibatch = data[i].src_sent_int.size();
				}
				if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
					max_target_minibatch = data[i].tgt_sent_int_i.size();
				}
			}

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {

				while(data[i].src_sent_int.size()<=max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
				}

				while(data[i].tgt_sent_int_i.size()<=max_target_minibatch) {
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
	}



	void prep_files_train_LM(int minibatch_size,int max_sent_cutoff,
		std::string target_file_name,
		std::string output_file_name,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size) 
	{


		target_input.open(target_file_name.c_str());
		final_output.open(output_file_name.c_str());
		//first stage is load all data into RAM
		std::string tgt_str; 
		std::string word;

		int target_len = 0;

		target_input.clear();

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}

		if(minibatch_size>target_len) {
			std::cerr << "ERROR: minibatch size cannot be greater than the file size\n";
			exit (EXIT_FAILURE);
		}

		//filter any long sentences and get ready to shuffle
		target_input.clear();
		target_input.seekg(0, std::ios::beg);
		for(int i=0; i<target_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			std::getline(target_input, tgt_str);

			std::istringstream iss_tgt(tgt_str, std::istringstream::in);
			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}
			if( !(src_sentence.size()>=max_sent_cutoff || tgt_sentence.size()>=max_sent_cutoff) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
			}
		}


		//shuffle the entire data
		if(shuffle) {
			std::random_shuffle(data.begin(),data.end());
		}


		//remove last sentences that do not fit in the minibatch
		if(target_len%minibatch_size!=0) {
			int num_to_remove = target_len%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}

		//sort the data based on minibatch
		compare_nonLM comp;
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


		//now get counts for mappings
		for(int i=0; i<data.size(); i++) {
			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_counts.count(data[i].tgt_sent[j])==0) {
					tgt_counts[data[i].tgt_sent[j]] = 1;
				}
				else {
					tgt_counts[data[i].tgt_sent[j]]+=1;
				}
			}
		}


		//now use heap to get the highest source and target mappings
		if(target_vocab_size==-1) {
			target_vocab_size = tgt_counts.size();
		}


		target_vocab_size = std::min(target_vocab_size,(int)tgt_counts.size());

		//output the model info to first line of output weights file
		std::ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << 1 << " " << hiddenstate_size << " " << target_vocab_size << "\n";

		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> tgt_map_heap;

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			tgt_map_heap.push( mapping_pair(it->first,it->second) );
		}

		output_model << "==========================================================\n";
		tgt_mapping["<START>"] = 0;
		tgt_mapping["<EOF>"] = 1;
		output_model << 0 << " " << "<START>" << "\n";
		output_model << 1 << " " << "<EOF>" << "\n";
		for(int i=2; i<target_vocab_size-1; i++) {
			tgt_mapping[tgt_map_heap.top().word] = i;
			output_model << i << " " << tgt_map_heap.top().word << "\n";
			tgt_map_heap.pop();
		}
		tgt_mapping["<UNK>"] = target_vocab_size-1;
		output_model << target_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model << "==========================================================\n";

		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> tgt_int;

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
					tgt_int.push_back(target_vocab_size-1);
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
			int max_target_minibatch=0;

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
					max_target_minibatch = data[i].tgt_sent_int_i.size();
				}
			}

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				while(data[i].tgt_sent_int_i.size()<=max_target_minibatch) {
					data[i].tgt_sent_int_i.push_back(-1);
					data[i].tgt_sent_int_o.push_back(-1);
				}
			}
			curr_index+=minibatch_size;
		}

		//now output to the file
		for(int i=0; i<data.size(); i++) {
			final_output << "\n";
			final_output << "\n";

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
		target_input.close();
	}


	//for reading file from user input, then mapping to tmp/, such as dev sets, decoding input,stoic input, etc..
	void integerize_file_nonLM(std::string output_weights_name,std::string source_file_name,std::string target_file_name,std::string tmp_output_name,
		int max_sent_cutoff,int minibatch_size,int &hiddenstate_size,int &source_vocab_size,int &target_vocab_size) 
	{

		std::ifstream weights_file;
		weights_file.open(output_weights_name.c_str());

		std::vector<std::string> file_input_vec;
		std::string str;
		std::string word;

		std::getline(weights_file, str);
		std::istringstream iss(str, std::istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

		if(file_input_vec.size()!=4) {
			std::cout << "ERROR: Neural network file format has been corrupted\n";
			exit (EXIT_FAILURE);
		}

		hiddenstate_size = std::stoi(file_input_vec[1]);
		target_vocab_size = std::stoi(file_input_vec[2]);
		source_vocab_size = std::stoi(file_input_vec[3]);

		//now get the mappings
		std::getline(weights_file, str); //get this line, since all equals
		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with source mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			src_mapping[word] = tmp_index;
		}

		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			tgt_mapping[word] = tmp_index;
		}

		//now that we have the mappings, integerize the file
		std::ofstream final_output;
		final_output.open(tmp_output_name.c_str());
		std::ifstream source_input;
		source_input.open(source_file_name.c_str());
		std::ifstream target_input;
		target_input.open(target_file_name.c_str());


		//first get the number of lines the the files and check to be sure they are the same
		int source_len = 0;
		int target_len = 0;
		std::string src_str;
		std::string tgt_str;

		source_input.clear();
		target_input.clear();

		source_input.seekg(0, std::ios::beg);
		while(std::getline(source_input, src_str)) {
			source_len++;
		}

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}


		//do check to be sure the two files are the same length
		if(source_len!=target_len) {
			std::cerr << "ERROR: Input files are not the same length\n";
			exit (EXIT_FAILURE);
		}


		source_input.clear();
		target_input.clear();
		source_input.seekg(0, std::ios::beg);
		target_input.seekg(0, std::ios::beg);
		for(int i=0; i<source_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			std::getline(source_input, src_str);
			std::getline(target_input, tgt_str);

			std::istringstream iss_src(src_str, std::istringstream::in);
			std::istringstream iss_tgt(tgt_str, std::istringstream::in);
			while(iss_src >> word) {
				src_sentence.push_back(word);
			}
			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if( !(src_sentence.size()>=max_sent_cutoff || tgt_sentence.size()>=max_sent_cutoff) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
			}
		}


		if(target_len%minibatch_size!=0) {
			std::random_shuffle(data.begin(),data.end());
			int num_to_remove = target_len%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}


		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;
			std::vector<int> tgt_int;
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_mapping.count(data[i].src_sent[j])==0) {
					//src_int.push_back(source_vocab_size-1);
					src_int.push_back(src_mapping["<UNK>"]);
				}
				else {
					src_int.push_back(src_mapping[data[i].src_sent[j]]);
				}	
			}
			std::reverse(src_int.begin(), src_int.end());
			data[i].src_sent.clear();
			data[i].src_sent_int = src_int;
			data[i].src_sent_int.insert(data[i].src_sent_int.begin(),0);
			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
					//tgt_int.push_back(target_vocab_size-1);
					tgt_int.push_back(tgt_mapping["<UNK>"]);
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

		//now pad
		int curr_index = 0;
		while(curr_index < data.size()) {
			int max_source_minibatch=0;
			int max_target_minibatch=0;

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
				if(data[i].src_sent_int.size()>max_source_minibatch) {
					max_source_minibatch = data[i].src_sent_int.size();
				}
				if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
					max_target_minibatch = data[i].tgt_sent_int_i.size();
				}
			}

			for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {

				while(data[i].src_sent_int.size()<=max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
				}

				while(data[i].tgt_sent_int_i.size()<=max_target_minibatch) {
					data[i].tgt_sent_int_i.push_back(-1);
					data[i].tgt_sent_int_o.push_back(-1);
				}
			}
			curr_index+=minibatch_size;
		}


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


		weights_file.close();
		final_output.close();
		source_input.close();
		target_input.close();
	}




	void integerize_file_LM(std::string output_weights_name,std::string target_file_name,std::string tmp_output_name,
		int max_sent_cutoff,int minibatch_size,bool dev,int &hiddenstate_size,int &target_vocab_size,bool kbest,int &source_vocab_size) 
	{

		std::ifstream weights_file;
		weights_file.open(output_weights_name.c_str());

		std::vector<std::string> file_input_vec;
		std::string str;
		std::string word;

		std::getline(weights_file, str);
		std::istringstream iss(str, std::istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

		if(file_input_vec.size()!=3) {
			//std::cout << "ERROR: Neural network file format has been corrupted\n";
			//exit (EXIT_FAILURE);
		}

		hiddenstate_size = std::stoi(file_input_vec[1]);
		target_vocab_size = std::stoi(file_input_vec[2]);
		if(kbest) {
			source_vocab_size = std::stoi(file_input_vec[3]);
		}

		//now get the target mappings
		std::getline(weights_file, str); //get this line, since all equals
		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
			}

			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			tgt_mapping[word] = tmp_index;
		}

		//now that we have the mappings, integerize the file
		std::ofstream final_output;
		final_output.open(tmp_output_name.c_str());
		std::ifstream target_input;
		target_input.open(target_file_name.c_str());


		//first get the number of lines the the files and check to be sure they are the same
		int target_len = 0;
		std::string tgt_str;

		target_input.clear();

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}


		target_input.clear();
		target_input.seekg(0, std::ios::beg);
		for(int i=0; i<target_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			std::getline(target_input, tgt_str);

			std::istringstream iss_tgt(tgt_str, std::istringstream::in);
			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if( !(tgt_sentence.size()>=max_sent_cutoff) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
			}
		}


		if(target_len%minibatch_size!=0) {
			std::random_shuffle(data.begin(),data.end());
			int num_to_remove = target_len%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}


		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> tgt_int;

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
					//tgt_int.push_back(target_vocab_size-1);
					tgt_int.push_back(tgt_mapping["<UNK>"]);
				}
				else {
					tgt_int.push_back(tgt_mapping[data[i].tgt_sent[j]]);
				}	
			}

			//reverse if kbest
			if(kbest) {
				std::reverse(tgt_int.begin(),tgt_int.end());
			}

			data[i].tgt_sent.clear();
			data[i].tgt_sent_int_i = tgt_int;
			data[i].tgt_sent_int_o = tgt_int;
			data[i].tgt_sent_int_i.insert(data[i].tgt_sent_int_i.begin(),0);
			data[i].tgt_sent_int_o.push_back(1);

		}

		//now pad based on minibatch
		if(dev) {
			int curr_index = 0;
			while(curr_index < data.size()) {
				int max_target_minibatch=0;

				for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
					if(data[i].tgt_sent_int_i.size()>max_target_minibatch) {
						max_target_minibatch = data[i].tgt_sent_int_i.size();
					}
				}

				for(int i=curr_index; i<std::min((int)data.size(),curr_index+minibatch_size); i++) {
					while(data[i].tgt_sent_int_i.size()<=max_target_minibatch) {
						data[i].tgt_sent_int_i.push_back(-1);
						data[i].tgt_sent_int_o.push_back(-1);
					}
				}
				curr_index+=minibatch_size;
			}
		}

		for(int i=0; i<data.size(); i++) {

			if(!kbest) {
				final_output << "\n";
				final_output << "\n";
			}

			for(int j=0; j<data[i].tgt_sent_int_i.size(); j++) {
				final_output << data[i].tgt_sent_int_i[j];
				if(j!=data[i].tgt_sent_int_i.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";

			if(!kbest) {
				for(int j=0; j<data[i].tgt_sent_int_o.size(); j++) {
					final_output << data[i].tgt_sent_int_o[j];
					if(j!=data[i].tgt_sent_int_o.size()) {
						final_output << " ";
					}
				}
				final_output << "\n";
			}
		}


		weights_file.close();
		final_output.close();
		target_input.close();
	}

	//need outputweights to get the int mapping
	void unint_file(std::string output_weights_name,std::string unint_file,std::string output_final_name,bool LM,bool decoder) {

		std::ifstream weights_file;
		weights_file.open(output_weights_name.c_str());
		weights_file.clear();
		weights_file.seekg(0, std::ios::beg);

		std::string str;
		std::string word;

		std::getline(weights_file, str); //info from first sentence
		std::getline(weights_file, str); //======== stuff
		if(!LM) {
			while(std::getline(weights_file, str)) {
				if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
					break; //done with target mapping
				}
			}
		}

		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
			}
			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			tgt_reverse_mapping[tmp_index] = word;
		}


		weights_file.close();

		std::ifstream unint;
		unint.open(unint_file.c_str());

		std::ofstream final_output;
		final_output.open(output_final_name.c_str());

		while(std::getline(unint, str)) {
			std::istringstream iss(str, std::istringstream::in);
			std::vector<int> sent_int;

			if(decoder) {
				if(str[0]=='-'|| str[0] == ' ' || str.size()==0) {
					final_output << str << "\n";
					continue;
				}
			}

			while(iss >> word) {
				sent_int.push_back(std::stoi(word));
			}

			for(int i=0; i<sent_int.size(); i++) {
				final_output << tgt_reverse_mapping[sent_int[i]];
				if(i!=sent_int.size()-1) {
					final_output << " ";
				}
			}
			final_output << "\n";
		}

		final_output.close();
		unint.close();
	}






};



#endif