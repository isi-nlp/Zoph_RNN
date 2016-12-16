#ifndef INPUT_FILE_PREP_H
#define INPUT_FILE_PREP_H

#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include "BZ_CUDA_UTIL.h"


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


struct comb_sent_info_ms {

	std::vector<std::string> src_sent;
	std::vector<std::string> src_sent_2;
	std::vector<std::string> tgt_sent;

	std::vector<int> src_sent_int;
	std::vector<int> minus_two_source;
	std::vector<int> src_sent_int_2;
	std::vector<int> minus_two_source_2;
	std::vector<int> tgt_sent_int_i;
	std::vector<int> tgt_sent_int_o;
	int total_len;

	comb_sent_info_ms(std::vector<std::string> &src_sent,std::vector<std::string> &src_sent_2,std::vector<std::string> &tgt_sent) {
		this->src_sent = src_sent;
		this->src_sent_2 = src_sent_2;
		this->tgt_sent = tgt_sent;
		total_len = tgt_sent.size() + src_sent.size() + src_sent_2.size();
	}
};

struct compare_nonLM_multisrc {
    bool operator()(const struct comb_sent_info_ms& first, const struct comb_sent_info_ms& second) {
        return first.total_len < second.total_len;
    }
};


//this will unk based on the source and target vocabulary
struct input_file_prep {

	std::ifstream source_input;
	std::ifstream source_input_2;
	std::ifstream target_input;
	std::ofstream final_output;
	std::ofstream final_output_2;

	std::unordered_map<std::string,int> src_mapping;
	std::unordered_map<std::string,int> src_mapping_2;
	std::unordered_map<std::string,int> tgt_mapping;

	std::unordered_map<int,std::string> tgt_reverse_mapping;
	std::unordered_map<int,std::string> src_reverse_mapping;

	std::unordered_map<std::string,int> src_counts;
	std::unordered_map<std::string,int> src_counts_2;
	std::unordered_map<std::string,int> tgt_counts;

	const int minibatch_mult = 10; //montreal uses 20
	std::vector<comb_sent_info> data; //can be used to sort by mult of minibatch


	bool prep_files_train_nonLM_multi_source_ensemble(int minibatch_size,int max_sent_cutoff,
		std::string source_file_name,std::string target_file_name,
		std::string output_file_name,int &source_vocab_size,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,
		int num_layers,std::string source_file_name_2,std::string output_file_name_2,
		std::string model_output_file_name_2,std::string ensemble_model_name_big,std::string ensemble_model_name_small);


	bool prep_files_train_nonLM(int minibatch_size,int max_sent_cutoff,
		std::string source_file_name,std::string target_file_name,
		std::string output_file_name,int &source_vocab_size,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,int num_layers,bool unk_replace,int unk_align_range,bool attention_model) 
	{
		int VISUAL_num_source_word_tokens =0;
		int VISUAL_total_source_vocab_size=0;
		int VISUAL_num_single_source_words=0;
		int VISUAL_num_segment_pairs=0;
		double VISUAL_avg_source_seg_len=0;
		int VISUAL_source_longest_sent=0;

		int VISUAL_num_target_word_tokens =0;
		int VISUAL_total_target_vocab_size=0;
		int VISUAL_num_single_target_words=0;
		VISUAL_num_segment_pairs=0;
		double VISUAL_avg_target_seg_len=0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;

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

		VISUAL_num_segment_pairs = target_len;

		//do check to be sure the two files are the same length
		if(source_len!=target_len) {
			BZ_CUDA::logger << "ERROR: Input files are not the same length\n";
			return false;
			//exit (EXIT_FAILURE);
		}

		if(minibatch_size>source_len) {
			BZ_CUDA::logger << "ERROR: minibatch size cannot be greater than the file size\n";
			return false;
			//exit (EXIT_FAILURE);
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

			if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
				VISUAL_avg_source_seg_len+=src_sentence.size();
				VISUAL_avg_target_seg_len+=tgt_sentence.size();
				VISUAL_num_source_word_tokens+=src_sentence.size();
				VISUAL_num_target_word_tokens+=tgt_sentence.size();

				if(VISUAL_source_longest_sent < src_sentence.size()) {
					VISUAL_source_longest_sent = src_sentence.size();
				}
				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}
		VISUAL_avg_source_seg_len = VISUAL_avg_source_seg_len/( (double)VISUAL_num_segment_pairs);
		VISUAL_avg_target_seg_len = VISUAL_avg_target_seg_len/( (double)VISUAL_num_segment_pairs);

		//shuffle the entire data
		if(BZ_CUDA::shuffle_data) {
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
			//exit (EXIT_FAILURE);
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
				if(data[i].src_sent[j]!= "<UNK>") {
					if(src_counts.count(data[i].src_sent[j])==0) {
						src_counts[data[i].src_sent[j]] = 1;
					}
					else {
						src_counts[data[i].src_sent[j]]+=1;
					}
				}
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(data[i].tgt_sent[j]!= "<UNK>") {
					if(tgt_counts.count(data[i].tgt_sent[j])==0) {
						tgt_counts[data[i].tgt_sent[j]] = 1;
					}
					else {
						tgt_counts[data[i].tgt_sent[j]]+=1;
					}
				}
			}
		}

		//now use heap to get the highest source and target mappings
		if(source_vocab_size==-1) {
			source_vocab_size = src_counts.size()+1;
		}
		if(target_vocab_size==-1) {
			if(!unk_replace) {
				target_vocab_size = tgt_counts.size()+3;
			}
			else {
				target_vocab_size = tgt_counts.size() + 3 + 1 + unk_align_range*2;
			}
		}

		VISUAL_total_source_vocab_size = src_counts.size();
		VISUAL_total_target_vocab_size = tgt_counts.size();

		if(!unk_replace) {
			source_vocab_size = std::min(source_vocab_size,(int)src_counts.size()+1);
			target_vocab_size = std::min(target_vocab_size,(int)tgt_counts.size()+3);
		}

		//std::cout << "source vocab size: " << source_vocab_size << "\n";
		//std::cout << "target vocab size: " << target_vocab_size << "\n";

		//output the model info to first line of output weights file
		std::ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << num_layers << " " << hiddenstate_size << " " << target_vocab_size << " " << source_vocab_size << "\n";

		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> src_map_heap;
		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> tgt_map_heap;

		for ( auto it = src_counts.begin(); it != src_counts.end(); ++it ) {
			src_map_heap.push( mapping_pair(it->first,it->second) );
			if(it->second==1) {
				VISUAL_num_single_source_words++;
			}
		}

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			tgt_map_heap.push( mapping_pair(it->first,it->second) );
			if(it->second==1) {
				VISUAL_num_single_target_words++;
			}
		}

		if(!unk_replace) {
			//std::cout << "DEBUG: source vocab size: " << source_vocab_size << "\n";
			output_model << "==========================================================\n";
			//src_mapping["<START>"] = 0;
			src_mapping["<UNK>"] = 0;
			output_model << 0 << " " << "<UNK>" << "\n";

			for(int i=1; i<source_vocab_size; i++) {
				//std::cout << "Debug: i= " << i << "\n";
				src_mapping[src_map_heap.top().word] = i;
				output_model << i << " " << src_map_heap.top().word << "\n";
				src_map_heap.pop();
			}
			// src_mapping["<UNK>"] = source_vocab_size-1;
			// output_model << source_vocab_size-1 << " " << "<UNK>" << "\n";
			output_model << "==========================================================\n";

			tgt_mapping["<START>"] = 0;
			tgt_mapping["<EOF>"] = 1;
			tgt_mapping["<UNK>"] = 2;
			output_model << 0 << " " << "<START>" << "\n";
			output_model << 1 << " " << "<EOF>" << "\n";
			output_model << 2 << " " << "<UNK>" << "\n";

			for(int i=3; i<target_vocab_size; i++) {
				tgt_mapping[tgt_map_heap.top().word] = i;
				output_model << i << " " << tgt_map_heap.top().word << "\n";
				tgt_map_heap.pop();
			}
			// tgt_mapping["<UNK>"] = target_vocab_size-1;
			// output_model << target_vocab_size-1 << " " << "<UNK>" << "\n";
			output_model << "==========================================================\n";
		}
		else {
			output_model << "==========================================================\n";
			src_mapping["<UNK>"] = 0;
			output_model << 0 << " " << "<UNK>" << "\n";

			//std::cout << "source vocab in unk: " << source_vocab_size << "\n";
			//std::cout << src_map_heap.size() << "\n";
			for(int i=1; i<source_vocab_size; i++) {
				src_mapping[src_map_heap.top().word] = i;
				output_model << i << " " << src_map_heap.top().word << "\n";
				src_map_heap.pop();
			}
			// src_mapping["<UNK>"] = source_vocab_size-1;
			// output_model << source_vocab_size-1 << " " << "<UNK>" << "\n";
			output_model << "==========================================================\n";

			tgt_mapping["<START>"] = 0;
			tgt_mapping["<EOF>"] = 1;
			tgt_mapping["<UNK>NULL"] = 2;
			output_model << 0 << " " << "<START>" << "\n";
			output_model << 1 << " " << "<EOF>" << "\n";
			output_model << 2 << " " << "<UNK>NULL" << "\n";

			int curr_index = 3;
			for(int i= -unk_align_range; i < unk_align_range + 1; i++) {
				tgt_mapping["<UNK>"+std::to_string(i)] = curr_index;
				output_model << curr_index << " " << "<UNK>"+std::to_string(i) << "\n";
				curr_index++;
			}
			//std::cout << "curr index " << curr_index << "\n";
			//std::cout << "target vocab size " << target_vocab_size << "\n";
			for(int i=curr_index; i < target_vocab_size; i++) {
				if(tgt_mapping.count(tgt_map_heap.top().word)==0) {
					tgt_mapping[tgt_map_heap.top().word] = i;
					output_model << i << " " << tgt_map_heap.top().word << "\n";
				}
				tgt_map_heap.pop();
			}
			// tgt_mapping["<UNK>"] = target_vocab_size-1;
			// output_model << target_vocab_size-1 << " " << "<UNK>" << "\n";
			output_model << "==========================================================\n";
		}


		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;
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
			// if(!attention_model) {
			// 	data[i].src_sent_int.insert(data[i].src_sent_int.begin(),0);
			// }

			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
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

				while(data[i].src_sent_int.size()<max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
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

		//print file stats:
		BZ_CUDA::logger << "----------------------------source train file info-----------------------------\n";
		BZ_CUDA::logger << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		BZ_CUDA::logger << "Source vocabulary size (before <unk>ing): " << VISUAL_total_source_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of source singleton word types: " << VISUAL_num_single_source_words<<"\n";
		BZ_CUDA::logger << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average source segment length: " << VISUAL_avg_source_seg_len<< "\n";
		BZ_CUDA::logger << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		//print file stats:
		BZ_CUDA::logger << "----------------------------target train file info-----------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Target vocabulary size (before <unk>ing): " << VISUAL_total_target_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of target singleton word types: " << VISUAL_num_single_target_words<<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average target segment length: " << VISUAL_avg_target_seg_len<< "\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff (source + target): " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
	
		return true;
	}

	bool prep_files_train_nonLM_multi_source(int minibatch_size,int max_sent_cutoff,
		std::string source_file_name,std::string target_file_name,
		std::string output_file_name,int &source_vocab_size,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,
		int num_layers,std::string source_file_name_2,std::string output_file_name_2,
		std::string model_output_file_name_2) 
	{
		int VISUAL_num_source_word_tokens =0;
		int VISUAL_total_source_vocab_size=0;
		int VISUAL_num_single_source_words=0;
		int VISUAL_num_segment_pairs=0;
		double VISUAL_avg_source_seg_len=0;
		int VISUAL_source_longest_sent=0;

		int VISUAL_num_source_2_word_tokens =0;
		int VISUAL_total_source_2_vocab_size=0;
		int VISUAL_num_single_source_2_words=0;
		double VISUAL_avg_source_2_seg_len=0;
		int VISUAL_source_2_longest_sent=0;

		int VISUAL_num_target_word_tokens =0;
		int VISUAL_total_target_vocab_size=0;
		int VISUAL_num_single_target_words=0;
		VISUAL_num_segment_pairs=0;
		double VISUAL_avg_target_seg_len=0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;

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

		VISUAL_num_segment_pairs = target_len;

		//do check to be sure the two files are the same length
		if(source_len!=target_len || source_len_2!=source_len) {
			BZ_CUDA::logger << "ERROR: Input files are not the same length\n";
			return false;
			//exit (EXIT_FAILURE);
		}

		if(minibatch_size>source_len) {
			BZ_CUDA::logger << "ERROR: minibatch size cannot be greater than the file size\n";
			return false;
			//exit (EXIT_FAILURE);
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
				VISUAL_avg_source_seg_len+=src_sentence.size();
				VISUAL_avg_source_2_seg_len+=src_sentence_2.size();
				VISUAL_avg_target_seg_len+=tgt_sentence.size();
				VISUAL_num_source_word_tokens+=src_sentence.size();
				VISUAL_num_source_2_word_tokens+=src_sentence_2.size();
				VISUAL_num_target_word_tokens+=tgt_sentence.size();

				if(VISUAL_source_longest_sent < src_sentence.size()) {
					VISUAL_source_longest_sent = src_sentence.size();
				}
				if(VISUAL_source_2_longest_sent < src_sentence_2.size()) {
					VISUAL_source_2_longest_sent = src_sentence_2.size();
				}
				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + src_sentence_2.size() + tgt_sentence.size();
			}
		}
		VISUAL_avg_source_seg_len = VISUAL_avg_source_seg_len/( (double)VISUAL_num_segment_pairs);
		VISUAL_avg_source_2_seg_len = VISUAL_avg_source_2_seg_len/( (double)VISUAL_num_segment_pairs);
		VISUAL_avg_target_seg_len = VISUAL_avg_target_seg_len/( (double)VISUAL_num_segment_pairs);

		//shuffle the entire data
		if(BZ_CUDA::shuffle_data) {
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
			BZ_CUDA::logger << "ERROR: file size is zero, could be wrong input file, all lines are above max sent length, or the minibatch size is too big\n";
			return false;
			//exit (EXIT_FAILURE);
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


		//now get counts for mappings
		for(int i=0; i<data.size(); i++) {
			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(data[i].src_sent[j]!= "<UNK>") {
					if(src_counts.count(data[i].src_sent[j])==0) {
						src_counts[data[i].src_sent[j]] = 1;
					}
					else {
						src_counts[data[i].src_sent[j]]+=1;
					}
				}
			}

			for(int j=0; j<data[i].src_sent_2.size(); j++) {
				if(data[i].src_sent_2[j]!= "<UNK>") {
					if(src_counts_2.count(data[i].src_sent_2[j])==0) {
						src_counts_2[data[i].src_sent_2[j]] = 1;
					}
					else {
						src_counts_2[data[i].src_sent_2[j]]+=1;
					}
				}
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(data[i].tgt_sent[j]!= "<UNK>") {
					if(tgt_counts.count(data[i].tgt_sent[j])==0) {
						tgt_counts[data[i].tgt_sent[j]] = 1;
					}
					else {
						tgt_counts[data[i].tgt_sent[j]]+=1;
					}
				}
			}
		}

		//now use heap to get the highest source and target mappings
		if(source_vocab_size==-1) {
			source_vocab_size = std::min(src_counts.size()+1,src_counts_2.size()+1);
		}
		if(target_vocab_size==-1) {
			target_vocab_size = tgt_counts.size()+3;
		}

		VISUAL_total_source_vocab_size = src_counts.size();
		VISUAL_total_source_2_vocab_size = src_counts_2.size();
		VISUAL_total_target_vocab_size = tgt_counts.size();

		
		source_vocab_size = std::min(source_vocab_size,(int)src_counts.size()+1);
		source_vocab_size = std::min(source_vocab_size,(int)src_counts_2.size()+1);
		target_vocab_size = std::min(target_vocab_size,(int)tgt_counts.size()+3);
		

		//output the model info to first line of output weights file
		std::ofstream output_model;
		std::ofstream output_model_2;
		output_model.open(model_output_file_name.c_str());
		output_model_2.open(model_output_file_name_2.c_str());
		output_model << num_layers << " " << hiddenstate_size << " " << target_vocab_size << " " << source_vocab_size << "\n";

		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> src_map_heap;
		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> src_map_heap_2;
		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> tgt_map_heap;

		for ( auto it = src_counts.begin(); it != src_counts.end(); ++it ) {
			src_map_heap.push( mapping_pair(it->first,it->second) );
			if(it->second==1) {
				VISUAL_num_single_source_words++;
			}
		}

		for ( auto it = src_counts_2.begin(); it != src_counts_2.end(); ++it ) {
			src_map_heap_2.push( mapping_pair(it->first,it->second) );
			if(it->second==1) {
				VISUAL_num_single_source_2_words++;
			}
		}

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			tgt_map_heap.push( mapping_pair(it->first,it->second) );
			if(it->second==1) {
				VISUAL_num_single_target_words++;
			}
		}

		
		output_model << "==========================================================\n";
		src_mapping["<UNK>"] = 0;
		output_model << 0 << " " << "<UNK>" << "\n";

		for(int i=1; i<source_vocab_size; i++) {
			src_mapping[src_map_heap.top().word] = i;
			output_model << i << " " << src_map_heap.top().word << "\n";
			src_map_heap.pop();
		}
		// src_mapping["<UNK>"] = source_vocab_size-1;
		// output_model << source_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model << "==========================================================\n";

		tgt_mapping["<START>"] = 0;
		tgt_mapping["<EOF>"] = 1;
		tgt_mapping["<UNK>"] = 2;
		output_model << 0 << " " << "<START>" << "\n";
		output_model << 1 << " " << "<EOF>" << "\n";
		output_model << 2 << " " << "<UNK>" << "\n";

		for(int i=3; i<target_vocab_size; i++) {
			tgt_mapping[tgt_map_heap.top().word] = i;
			output_model << i << " " << tgt_map_heap.top().word << "\n";
			tgt_map_heap.pop();
		}
		// tgt_mapping["<UNK>"] = target_vocab_size-1;
		// output_model << target_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model << "==========================================================\n";
	

		output_model_2 << "==========================================================\n";
		src_mapping_2["<UNK>"] = 0;
		output_model_2 << 0 << " " << "<UNK>" << "\n";

		for(int i=1; i<source_vocab_size; i++) {
			src_mapping_2[src_map_heap_2.top().word] = i;
			output_model_2 << i << " " << src_map_heap_2.top().word << "\n";
			src_map_heap_2.pop();
		}
		// src_mapping["<UNK>"] = source_vocab_size-1;
		// output_model << source_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model_2 << "==========================================================\n";

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

		//print file stats:
		BZ_CUDA::logger << "----------------------------source train file info-----------------------------\n";
		BZ_CUDA::logger << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		BZ_CUDA::logger << "Source vocabulary size (before <unk>ing): " << VISUAL_total_source_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of source singleton word types: " << VISUAL_num_single_source_words<<"\n";
		BZ_CUDA::logger << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average source segment length: " << VISUAL_avg_source_seg_len<< "\n";
		BZ_CUDA::logger << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		//second source file
		BZ_CUDA::logger << "----------------------------source train 2 file info-----------------------------\n";
		BZ_CUDA::logger << "Number of source word tokens: " << VISUAL_num_source_2_word_tokens <<"\n";
		BZ_CUDA::logger << "Source vocabulary size (before <unk>ing): " << VISUAL_total_source_2_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of source singleton word types: " << VISUAL_num_single_source_2_words<<"\n";
		BZ_CUDA::logger << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average source segment length: " << VISUAL_avg_source_2_seg_len<< "\n";
		BZ_CUDA::logger << "Longest source segment (after removing long sentences for training): " << VISUAL_source_2_longest_sent << "\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		//print file stats:
		BZ_CUDA::logger << "----------------------------target train file info-----------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Target vocabulary size (before <unk>ing): " << VISUAL_total_target_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of target singleton word types: " << VISUAL_num_single_target_words<<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average target segment length: " << VISUAL_avg_target_seg_len<< "\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff (source + target): " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		

		return true;
	}



	bool prep_files_train_nonLM_ensemble(int minibatch_size,int max_sent_cutoff,
		std::string source_file_name,std::string target_file_name,
		std::string output_file_name,int &source_vocab_size,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,int num_layers,
		std::string ensemble_model_name,bool attention_model)
	{

		int VISUAL_num_source_word_tokens =0;
		int VISUAL_total_source_vocab_size=0;
		int VISUAL_num_single_source_words=0;
		int VISUAL_num_segment_pairs=0;
		double VISUAL_avg_source_seg_len=0;
		int VISUAL_source_longest_sent=0;

		int VISUAL_num_target_word_tokens =0;
		int VISUAL_total_target_vocab_size=0;
		int VISUAL_num_single_target_words=0;
		VISUAL_num_segment_pairs=0;
		double VISUAL_avg_target_seg_len=0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;


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

		VISUAL_num_segment_pairs = target_len;

		//do check to be sure the two files are the same length
		if(source_len!=target_len) {
			BZ_CUDA::logger << "ERROR: Input files are not the same length\n";
			return false;
			//exit (EXIT_FAILURE);
		}

		if(minibatch_size>source_len) {
			BZ_CUDA::logger << "ERROR: minibatch size cannot be greater than the file size\n";
			return false;
			//exit (EXIT_FAILURE);
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

			if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
				VISUAL_avg_source_seg_len+=src_sentence.size();
				VISUAL_avg_target_seg_len+=tgt_sentence.size();
				VISUAL_num_source_word_tokens+=src_sentence.size();
				VISUAL_num_target_word_tokens+=tgt_sentence.size();

				if(VISUAL_source_longest_sent < src_sentence.size()) {
					VISUAL_source_longest_sent = src_sentence.size();
				}
				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}
		VISUAL_avg_source_seg_len = VISUAL_avg_source_seg_len/( (double)VISUAL_num_segment_pairs);
		VISUAL_avg_target_seg_len = VISUAL_avg_target_seg_len/( (double)VISUAL_num_segment_pairs);

		//shuffle the entire data
		if(BZ_CUDA::shuffle_data) {
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
			//exit (EXIT_FAILURE);
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
				if(data[i].src_sent[j]!= "<UNK>") {
					if(src_counts.count(data[i].src_sent[j])==0) {
						src_counts[data[i].src_sent[j]] = 1;
					}
					else {
						src_counts[data[i].src_sent[j]]+=1;
					}
				}
			}

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(data[i].tgt_sent[j] != "<UNK>") {
					if(tgt_counts.count(data[i].tgt_sent[j])==0) {
						tgt_counts[data[i].tgt_sent[j]] = 1;
					}
					else {
						tgt_counts[data[i].tgt_sent[j]]+=1;
					}
				}
			}
		}

		VISUAL_total_source_vocab_size = src_counts.size();
		VISUAL_total_target_vocab_size = tgt_counts.size();



		for ( auto it = src_counts.begin(); it != src_counts.end(); ++it ) {
			if(it->second==1) {
				VISUAL_num_single_source_words++;
			}
		}

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			if(it->second==1) {
				VISUAL_num_single_target_words++;
			}
		}


		//now load in the integer mappings from the other file for ensemble training
		std::ifstream ensemble_file;
		ensemble_file.open(ensemble_model_name.c_str());

		std::vector<std::string> file_input_vec;
		std::string str;

		std::getline(ensemble_file, str);
		std::istringstream iss(str, std::istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

//		if(file_input_vec.size()!=4) {
//		 	BZ_CUDA::logger << "ERROR: Neural network file format is not correct\n";
//		 	exit (EXIT_FAILURE);
//		}

		target_vocab_size = std::stoi(file_input_vec[2]);
		source_vocab_size = std::stoi(file_input_vec[3]);


		std::ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << num_layers << " " << hiddenstate_size << " " << target_vocab_size << " " << source_vocab_size << "\n";

		output_model << "==========================================================\n";

		//now get the mappings
		std::getline(ensemble_file, str); //get this line, since all equals
		while(std::getline(ensemble_file, str)) {
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
		while(std::getline(ensemble_file, str)) {
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
		ensemble_file.close();


		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;
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
			// if(!attention_model) {
			// 	data[i].src_sent_int.insert(data[i].src_sent_int.begin(),0);
			// }
			while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
				data[i].minus_two_source.push_back(-2);
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

				while(data[i].src_sent_int.size()<max_source_minibatch) {
					data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
					data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
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

		//print file stats:
		BZ_CUDA::logger << "----------------------------source train file info-----------------------------\n";
		BZ_CUDA::logger << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		BZ_CUDA::logger << "Source vocabulary size (before <unk>ing): " << VISUAL_total_source_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of source singleton word types: " << VISUAL_num_single_source_words<<"\n";
		BZ_CUDA::logger << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average source segment length: " << VISUAL_avg_source_seg_len<< "\n";
		BZ_CUDA::logger << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		//print file stats:
		BZ_CUDA::logger << "----------------------------target train file info-----------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Target vocabulary size (before <unk>ing): " << VISUAL_total_target_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of target singleton word types: " << VISUAL_num_single_target_words<<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average target segment length: " << VISUAL_avg_target_seg_len<< "\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff (source + target): " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
	
		return true;
	}



	bool prep_files_train_LM(int minibatch_size,int max_sent_cutoff,
		std::string target_file_name,
		std::string output_file_name,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,int num_layers) 
	{

		int VISUAL_num_target_word_tokens =0;
		int VISUAL_total_target_vocab_size=0;
		int VISUAL_num_single_target_words=0;
		int VISUAL_num_segment_pairs=0;
		double VISUAL_avg_target_seg_len=0;
		int VISUAL_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;

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

		VISUAL_num_segment_pairs = target_len;

		if(minibatch_size>target_len) {
			std::cerr << "ERROR: minibatch size cannot be greater than the file size\n";
			return false;
			//exit (EXIT_FAILURE);
		}


		double VISUAL_tmp_running_seg_len=0;

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
			if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size() + 1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
				VISUAL_tmp_running_seg_len+=tgt_sentence.size();
				VISUAL_num_target_word_tokens+=tgt_sentence.size();
				if(tgt_sentence.size() > VISUAL_longest_sent) {
					VISUAL_longest_sent = tgt_sentence.size() ;
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}

		VISUAL_avg_target_seg_len = VISUAL_tmp_running_seg_len/VISUAL_num_segment_pairs;


		//shuffle the entire data
		if(BZ_CUDA::shuffle_data) {
			std::random_shuffle(data.begin(),data.end());
		}

		// //remove last sentences that do not fit in the minibatch
		// if(data.size()%minibatch_size!=0) {
		// 	int num_to_remove = data.size()%minibatch_size;
		// 	for(int i=0; i<num_to_remove; i++) {
		// 		data.pop_back();
		// 	}
		// }

		if(data.size()==0) {
			BZ_CUDA::logger << "ERROR: your dataset if of length zero\n";
			return false;
			//exit (EXIT_FAILURE);
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
				if(data[i].tgt_sent[j] != "<UNK>") {
					if(tgt_counts.count(data[i].tgt_sent[j])==0) {

						tgt_counts[data[i].tgt_sent[j]] = 1;
					}
					else {
						tgt_counts[data[i].tgt_sent[j]]+=1;
					}
				}
			}
		}


		//now use heap to get the highest source and target mappings
		if(target_vocab_size==-1) {
			target_vocab_size = tgt_counts.size()+3;
		}

		VISUAL_total_target_vocab_size = tgt_counts.size();

		target_vocab_size = std::min(target_vocab_size,(int)tgt_counts.size()+3);

		//output the model info to first line of output weights file
		std::ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << num_layers << " " << hiddenstate_size << " " << target_vocab_size << "\n";

		std::priority_queue<mapping_pair,std::vector<mapping_pair>, mapping_pair_compare_functor> tgt_map_heap;

		for ( auto it = tgt_counts.begin(); it != tgt_counts.end(); ++it ) {
			tgt_map_heap.push( mapping_pair(it->first,it->second) );
			if(it->second==1) {
				VISUAL_num_single_target_words++;
			}
		}

		output_model << "==========================================================\n";
		tgt_mapping["<START>"] = 0;
		tgt_mapping["<EOF>"] = 1;
		tgt_mapping["<UNK>"] = 2;
		output_model << 0 << " " << "<START>" << "\n";
		output_model << 1 << " " << "<EOF>" << "\n";
		output_model << 2 << " " << "<UNK>" << "\n";

		for(int i=3; i<target_vocab_size; i++) {
			tgt_mapping[tgt_map_heap.top().word] = i;
			output_model << i << " " << tgt_map_heap.top().word << "\n";
	
			tgt_map_heap.pop();
		}
		// tgt_mapping["<UNK>"] = target_vocab_size-1;
		// output_model << target_vocab_size-1 << " " << "<UNK>" << "\n";
		output_model << "==========================================================\n";


		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> tgt_int;

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
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



		//now add in all -1's to make the last minibatch complete
		int num_extra_to_add = minibatch_size - data.size()%minibatch_size;
		if(num_extra_to_add==minibatch_size) {
			num_extra_to_add = 0;
		}
		int target_sent_len = data.back().tgt_sent_int_i.size();
		for(int i=0; i<num_extra_to_add; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			data.push_back(comb_sent_info(src_sentence,tgt_sentence));

			std::vector<int> tgt_int_m1;
			for(int j=0; j<target_sent_len; j++) {
				tgt_int_m1.push_back(-1);
			}
			data.back().tgt_sent_int_i = tgt_int_m1;
			data.back().tgt_sent_int_o = tgt_int_m1;
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

		//print file stats:
		BZ_CUDA::logger << "----------------------------target train file info-------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Target vocabulary size (before <unk>ing): " << VISUAL_total_target_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of target singleton word types: " << VISUAL_num_single_target_words<<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average target segment length: " << VISUAL_avg_target_seg_len<< "\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff: " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		
		return true;
	}



	bool prep_files_train_LM_ensemble(int minibatch_size,int max_sent_cutoff,
		std::string target_file_name,
		std::string output_file_name,int &target_vocab_size,
		bool shuffle,std::string model_output_file_name,int hiddenstate_size,int num_layers,std::string ensemble_model_name) 
	{

		int VISUAL_num_target_word_tokens =0;
		int VISUAL_total_target_vocab_size=0;
		int VISUAL_num_single_target_words=0;
		int VISUAL_num_segment_pairs=0;
		double VISUAL_avg_target_seg_len=0;
		int VISUAL_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;

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

		VISUAL_num_segment_pairs = target_len;

		if(minibatch_size>target_len) {
			std::cerr << "ERROR: minibatch size cannot be greater than the file size\n";
			return false;
			//exit (EXIT_FAILURE);
		}


		double VISUAL_tmp_running_seg_len=0;

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
			if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size() + 1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
				VISUAL_tmp_running_seg_len+=tgt_sentence.size();
				VISUAL_num_target_word_tokens+=tgt_sentence.size();
				if(tgt_sentence.size() > VISUAL_longest_sent) {
					VISUAL_longest_sent = tgt_sentence.size() ;
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}

		VISUAL_avg_target_seg_len = VISUAL_tmp_running_seg_len/VISUAL_num_segment_pairs;


		//shuffle the entire data
		if(BZ_CUDA::shuffle_data) {
			std::random_shuffle(data.begin(),data.end());
		}

		//remove last sentences that do not fit in the minibatch
		// if(data.size()%minibatch_size!=0) {
		// 	int num_to_remove = data.size()%minibatch_size;
		// 	for(int i=0; i<num_to_remove; i++) {
		// 		data.pop_back();
		// 	}
		// }

		if(data.size()==0) {
			BZ_CUDA::logger << "ERROR: your dataset is of length zero\n";
			return false;
			//exit (EXIT_FAILURE);
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
				if(data[i].tgt_sent[j] != "<UNK>") {
					if(tgt_counts.count(data[i].tgt_sent[j])==0) {
						tgt_counts[data[i].tgt_sent[j]] = 1;
					}
					else {
						tgt_counts[data[i].tgt_sent[j]]+=1;
					}
				}
			}
		}


		VISUAL_total_target_vocab_size = tgt_counts.size();

		//now load in the integer mappings from the other file for ensemble training
		std::ifstream ensemble_file;
		ensemble_file.open(ensemble_model_name.c_str());

		std::vector<std::string> file_input_vec;
		std::string str;

		std::getline(ensemble_file, str);
		std::istringstream iss(str, std::istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

		// if(file_input_vec.size()!=3) {
		// 	BZ_CUDA::logger << "ERROR: Neural network file format has been corrupted\n";
		// 	//exit (EXIT_FAILURE);
		// }

		target_vocab_size = std::stoi(file_input_vec[2]);

		std::ofstream output_model;
		output_model.open(model_output_file_name.c_str());
		output_model << num_layers << " " << hiddenstate_size << " " << target_vocab_size << "\n";

		output_model << "==========================================================\n";
		std::getline(ensemble_file, str);
		while(std::getline(ensemble_file, str)) {
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
		ensemble_file.close();

		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> tgt_int;

			for(int j=0; j<data[i].tgt_sent.size(); j++) {
				if(tgt_mapping.count(data[i].tgt_sent[j])==0) {
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


		//now add in all -1's to make the last minibatch complete
		int num_extra_to_add = minibatch_size - data.size()%minibatch_size;
		int target_sent_len = data.back().tgt_sent_int_i.size();
		for(int i=0; i<num_extra_to_add; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			data.push_back(comb_sent_info(src_sentence,tgt_sentence));

			std::vector<int> tgt_int_m1;
			for(int j=0; j<target_sent_len; j++) {
				tgt_int_m1.push_back(-1);
			}
			data.back().tgt_sent_int_i = tgt_int_m1;
			data.back().tgt_sent_int_o = tgt_int_m1;

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

		//print file stats:
		BZ_CUDA::logger << "----------------------------target train file info-------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Target vocabulary size (before <unk>ing): " << VISUAL_total_target_vocab_size<<"\n";
		BZ_CUDA::logger << "Number of target singleton word types: " << VISUAL_num_single_target_words<<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Average target segment length: " << VISUAL_avg_target_seg_len<< "\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff: " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		
		return true;
	}

	//for reading file from user input, then mapping to tmp/, such as dev sets, decoding input,stoic input, etc..
	void integerize_file_nonLM(std::string output_weights_name,std::string source_file_name,std::string target_file_name,std::string tmp_output_name,
		int max_sent_cutoff,int minibatch_size,int &hiddenstate_size,int &source_vocab_size,int &target_vocab_size,int &num_layers,bool attention_model,
		bool multi_source,std::string mult_source_file,std::string tmp_output_name_ms,std::string ms_mapping_file) 
	{

		int VISUAL_num_source_word_tokens =0;
		int VISUAL_num_segment_pairs=0;
		int VISUAL_source_longest_sent=0;

		int VISUAL_num_target_word_tokens =0;
		VISUAL_num_segment_pairs=0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;


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

		// if(file_input_vec.size()!=4) {
		// 	BZ_CUDA::logger << "ERROR: Neural network file format has been corrupted\n";
		// 	//exit (EXIT_FAILURE);
		// }

		num_layers = std::stoi(file_input_vec[0]);
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


		std::vector<comb_sent_info_ms> data_ms; //can be used to sort by mult of minibatch
		std::ifstream mapping_ms;
		if(multi_source) {
			mapping_ms.open(ms_mapping_file.c_str());
		}

		if(multi_source) {
			std::getline(mapping_ms, str);
			while(std::getline(mapping_ms, str)) {
				int tmp_index;

				if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
					break; //done with target mapping
				}

				std::istringstream iss(str, std::istringstream::in);
				iss >> word;
				tmp_index = std::stoi(word);
				iss >> word;
				src_mapping_2[word] = tmp_index;
			}
		}

		//now that we have the mappings, integerize the file
		std::ofstream final_output;
		final_output.open(tmp_output_name.c_str());
		std::ifstream source_input;
		source_input.open(source_file_name.c_str());
		std::ifstream target_input;
		target_input.open(target_file_name.c_str());

		std::ifstream source_input_2;
		if(multi_source) {
			source_input_2.open(mult_source_file.c_str());
			source_input_2.clear();
			BZ_CUDA::logger << "opening output file: " << tmp_output_name_ms << "\n";
			final_output_2.open(tmp_output_name_ms.c_str());
		}

		//first get the number of lines the the files and check to be sure they are the same
		int source_len = 0;
		int source_len_2 = 0;
		int target_len = 0;
		std::string src_str;
		std::string src_str_2;
		std::string tgt_str;

		source_input.clear();
		target_input.clear();

		source_input.seekg(0, std::ios::beg);
		while(std::getline(source_input, src_str)) {
			source_len++;
		}

		if(multi_source) {
			source_input_2.seekg(0, std::ios::beg);
			while(std::getline(source_input_2, src_str_2)) {
				source_len_2++;
			}

			if(source_len_2!=source_len) {
				BZ_CUDA::logger << "ERROR FOR MULTI SOURCE THE SOURCE FILE ARE NOT THE SAME LENGTH\n";
				exit (EXIT_FAILURE);
			}
		}

		target_input.seekg(0, std::ios::beg);
		while(std::getline(target_input, tgt_str)) {
			target_len++;
		}

		VISUAL_num_segment_pairs = target_len;

		//do check to be sure the two files are the same length
		if(source_len!=target_len) {
			BZ_CUDA::logger << "ERROR: Input files are not the same length\n";
			exit (EXIT_FAILURE);
		}

		if(multi_source) {
			source_input_2.clear();
			source_input_2.seekg(0, std::ios::beg);
		}

		source_input.clear();
		target_input.clear();
		source_input.seekg(0, std::ios::beg);
		target_input.seekg(0, std::ios::beg);
		for(int i=0; i<source_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> src_sentence_2;
			std::vector<std::string> tgt_sentence;
			std::getline(source_input, src_str);

			if(multi_source) {
				std::getline(source_input_2, src_str_2);
			}

			std::getline(target_input, tgt_str);

			std::istringstream iss_src(src_str, std::istringstream::in);
			std::istringstream iss_src_2(src_str_2, std::istringstream::in);
			std::istringstream iss_tgt(tgt_str, std::istringstream::in);
			while(iss_src >> word) {
				src_sentence.push_back(word);
			}

			if(multi_source) {
				while(iss_src_2 >> word) {
					src_sentence_2.push_back(word);
				}
			}

			while(iss_tgt >> word) {
				tgt_sentence.push_back(word);
			}

			if(!multi_source) {
				if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2) ) {
					data.push_back(comb_sent_info(src_sentence,tgt_sentence));

					VISUAL_num_source_word_tokens+=src_sentence.size();
					VISUAL_num_target_word_tokens+=tgt_sentence.size();

					if(VISUAL_source_longest_sent < src_sentence.size()) {
						VISUAL_source_longest_sent = src_sentence.size();
					}
					if(VISUAL_target_longest_sent < tgt_sentence.size()) {
						VISUAL_target_longest_sent = tgt_sentence.size();
					}
				}
				else {
					VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
				}
			}
			else {
				if( !(src_sentence.size()+1>=max_sent_cutoff-2 || tgt_sentence.size()+1>=max_sent_cutoff-2 || src_sentence_2.size()+1>=max_sent_cutoff-2  ) ) {
					data_ms.push_back(comb_sent_info_ms(src_sentence,src_sentence_2,tgt_sentence));

					VISUAL_num_source_word_tokens+=src_sentence.size();
					VISUAL_num_target_word_tokens+=tgt_sentence.size();

					if(VISUAL_source_longest_sent < src_sentence.size()) {
						VISUAL_source_longest_sent = src_sentence.size();
					}
					if(VISUAL_target_longest_sent < tgt_sentence.size()) {
						VISUAL_target_longest_sent = tgt_sentence.size();
					}
				}
				else {
					VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
				}
			}
		}	

		if( (minibatch_size!=1) ) {
            if(BZ_CUDA::shuffle_data) {
			    std::random_shuffle(data.begin(),data.end());
            }
		}

		//sort the data based on minibatch
		if( (minibatch_size!=1) ) {
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
		}

		if(!multi_source) {
			if(data.size()%minibatch_size!=0) {
				//std::random_shuffle(data.begin(),data.end());
				int num_to_remove = data.size()%minibatch_size;
				for(int i=0; i<num_to_remove; i++) {
					data.pop_back();
				}
			}
		}
		else {
			if(data_ms.size()%minibatch_size!=0) {
				//std::random_shuffle(data_ms.begin(),data_ms.end());
				int num_to_remove = data_ms.size()%minibatch_size;
				for(int i=0; i<num_to_remove; i++) {
					data_ms.pop_back();
				}
			}
		}

		if(data.size()==0 && data_ms.size()==0) {
			BZ_CUDA::logger << "ERROR: file size is zero, could be wrong input file or all lines are above max sent length\n";
			exit (EXIT_FAILURE);
		}

		//now integerize
		for(int i=0; i<std::max(data.size(),data_ms.size()); i++) {
			std::vector<int> src_int;
			std::vector<int> src_int_2;
			std::vector<int> tgt_int;

			if(!multi_source) {
				for(int j=0; j<data[i].src_sent.size(); j++) {
					if(src_mapping.count(data[i].src_sent[j])==0) {
						//src_int.push_back(source_vocab_size-1);
						src_int.push_back(src_mapping["<UNK>"]);
					}
					else {
						src_int.push_back(src_mapping[data[i].src_sent[j]]);
					}	
				}
			}
			else {
				for(int j=0; j<data_ms[i].src_sent.size(); j++) {
					if(src_mapping.count(data_ms[i].src_sent[j])==0) {
						//src_int.push_back(source_vocab_size-1);
						src_int.push_back(src_mapping["<UNK>"]);
					}
					else {
						src_int.push_back(src_mapping[data_ms[i].src_sent[j]]);
					}	
				}
			}

			std::reverse(src_int.begin(), src_int.end());

			if(!multi_source) {
				data[i].src_sent.clear();
				data[i].src_sent_int = src_int;
			}
			else {
				data_ms[i].src_sent.clear();
				data_ms[i].src_sent_int = src_int;
			}


			// if(!attention_model && !multi_source) {
			// 	data[i].src_sent_int.insert(data[i].src_sent_int.begin(),0);
			// }

			if(!multi_source) {
				while(data[i].minus_two_source.size()!=data[i].src_sent_int.size()) {
					data[i].minus_two_source.push_back(-2);
				}
			}
			else {
				while(data_ms[i].minus_two_source.size()!=data_ms[i].src_sent_int.size()) {
					data_ms[i].minus_two_source.push_back(-2);
				}
			}

			if(multi_source) {
				for(int j=0; j<data_ms[i].src_sent_2.size(); j++) {
					if(src_mapping_2.count(data_ms[i].src_sent_2[j])==0) {
						//src_int.push_back(source_vocab_size-1);
						src_int_2.push_back(src_mapping_2["<UNK>"]);
					}
					else {
						src_int_2.push_back(src_mapping_2[data_ms[i].src_sent_2[j]]);
					}	
				}
				std::reverse(src_int_2.begin(), src_int_2.end());
				data_ms[i].src_sent_2.clear();
				data_ms[i].src_sent_int_2 = src_int_2;

				while(data_ms[i].minus_two_source_2.size()!=data_ms[i].src_sent_int_2.size()) {
					data_ms[i].minus_two_source_2.push_back(-2);
				}
			}

			int max_iter = 0;
			if(!multi_source) {
				max_iter = data[i].tgt_sent.size();
			}
			else {
				max_iter = data_ms[i].tgt_sent.size();
			}
			for(int j=0; j<max_iter; j++) {

				if(!multi_source) {
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
				else {
					if(tgt_mapping.count(data_ms[i].tgt_sent[j])==0) {
						if(tgt_mapping.count("<UNK>")==0) {
							tgt_int.push_back(tgt_mapping["<UNK>NULL"]);
						}
						else {
							tgt_int.push_back(tgt_mapping["<UNK>"]);
						}
					}
					else {
						tgt_int.push_back(tgt_mapping[data_ms[i].tgt_sent[j]]);
					}
				}
			}

			if(!multi_source) {
				data[i].tgt_sent.clear();
				data[i].tgt_sent_int_i = tgt_int;
				data[i].tgt_sent_int_o = tgt_int;
				data[i].tgt_sent_int_i.insert(data[i].tgt_sent_int_i.begin(),0);
				data[i].tgt_sent_int_o.push_back(1);
			}
			else {
				data_ms[i].tgt_sent.clear();
				data_ms[i].tgt_sent_int_i = tgt_int;
				data_ms[i].tgt_sent_int_o = tgt_int;
				data_ms[i].tgt_sent_int_i.insert(data_ms[i].tgt_sent_int_i.begin(),0);
				data_ms[i].tgt_sent_int_o.push_back(1);
			}
		}

		//now pad
		if(!multi_source) {
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

					while(data[i].src_sent_int.size()<max_source_minibatch) {
						data[i].src_sent_int.insert(data[i].src_sent_int.begin(),-1);
						data[i].minus_two_source.insert(data[i].minus_two_source.begin(),-1);
					}

					while(data[i].tgt_sent_int_i.size()<max_target_minibatch) {
						data[i].tgt_sent_int_i.push_back(-1);
						data[i].tgt_sent_int_o.push_back(-1);
					}
				}
				curr_index+=minibatch_size;
			}
		}
		else {
			int curr_index = 0;
			while(curr_index < data_ms.size()) {
				int max_source_minibatch=0;
				int max_source_minibatch_2=0;
				int max_target_minibatch=0;

				for(int i=curr_index; i<std::min((int)data_ms.size(),curr_index+minibatch_size); i++) {
					if(data_ms[i].src_sent_int.size()>max_source_minibatch) {
						max_source_minibatch = data_ms[i].src_sent_int.size();
					}

					if(data_ms[i].src_sent_int_2.size()>max_source_minibatch_2) {
						max_source_minibatch_2 = data_ms[i].src_sent_int_2.size();
					}

					if(data_ms[i].tgt_sent_int_i.size()>max_target_minibatch) {
						max_target_minibatch = data_ms[i].tgt_sent_int_i.size();
					}
				}


				for(int i=curr_index; i<std::min((int)data_ms.size(),curr_index+minibatch_size); i++) {

					while(data_ms[i].src_sent_int.size()<max_source_minibatch) {
						data_ms[i].src_sent_int.insert(data_ms[i].src_sent_int.begin(),-1);
						data_ms[i].minus_two_source.insert(data_ms[i].minus_two_source.begin(),-1);
					}

					while(data_ms[i].src_sent_int_2.size()<max_source_minibatch_2) {
						data_ms[i].src_sent_int_2.insert(data_ms[i].src_sent_int_2.begin(),-1);
						data_ms[i].minus_two_source_2.insert(data_ms[i].minus_two_source_2.begin(),-1);
					}

					while(data_ms[i].tgt_sent_int_i.size()<max_target_minibatch) {
						data_ms[i].tgt_sent_int_i.push_back(-1);
						data_ms[i].tgt_sent_int_o.push_back(-1);
					}
				}
				curr_index+=minibatch_size;
			}
		}

		if(!multi_source) {
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
		}
		else {
			for(int i=0; i<data_ms.size(); i++) {
				for(int j=0; j<data_ms[i].src_sent_int.size(); j++) {
					final_output << data_ms[i].src_sent_int[j];
					if(j!=data_ms[i].src_sent_int.size()) {
						final_output << " ";
					}
				}
				final_output << "\n";

				for(int j=0; j<data_ms[i].minus_two_source.size(); j++) {
					final_output << data_ms[i].minus_two_source[j];
					if(j!=data_ms[i].minus_two_source.size()) {
						final_output << " ";
					}
				}
				final_output << "\n";


				for(int j=0; j<data_ms[i].src_sent_int_2.size(); j++) {
					final_output_2 << data_ms[i].src_sent_int_2[j];
					if(j!=data_ms[i].src_sent_int_2.size()) {
						final_output_2 << " ";
					}
				}
				final_output_2 << "\n";

				for(int j=0; j<data_ms[i].minus_two_source_2.size(); j++) {
					final_output_2 << data_ms[i].minus_two_source_2[j];
					if(j!=data_ms[i].minus_two_source_2.size()) {
						final_output_2 << " ";
					}
				}
				final_output_2 << "\n";


				for(int j=0; j<data_ms[i].tgt_sent_int_i.size(); j++) {
					final_output << data_ms[i].tgt_sent_int_i[j];
					if(j!=data_ms[i].tgt_sent_int_i.size()) {
						final_output << " ";
					}
				}
				final_output << "\n";


				for(int j=0; j<data_ms[i].tgt_sent_int_o.size(); j++) {
					final_output << data_ms[i].tgt_sent_int_o[j];
					if(j!=data_ms[i].tgt_sent_int_o.size()) {
						final_output << " ";
					}
				}
				final_output << "\n";
			}

			final_output_2.close();

		}


		weights_file.close();
		final_output.close();
		source_input.close();
		target_input.close();


		//print file stats:
		BZ_CUDA::logger << "----------------------------source dev file info-----------------------------\n";
		BZ_CUDA::logger << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		BZ_CUDA::logger << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
		//print file stats:
		BZ_CUDA::logger << "----------------------------target dev file info-----------------------------\n\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff (source + target): " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
	}


	void integerize_file_LM_carve(std::string output_weights_name,std::string target_file_name,std::string tmp_output_name,
		int max_sent_cutoff,int minibatch_size,bool dev,int &hiddenstate_size,int &target_vocab_size,int &num_layers) 
	{


		int VISUAL_num_target_word_tokens =0;
		int VISUAL_num_segment_pairs=0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;

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

		// if(file_input_vec.size()!=3) {
		// 	BZ_CUDA::logger << "ERROR: Neural network file format has been corrupted\n";
		// 	//exit (EXIT_FAILURE);
		// }

		num_layers = std::stoi(file_input_vec[0]);
		hiddenstate_size = std::stoi(file_input_vec[1]);
		target_vocab_size = std::stoi(file_input_vec[2]);

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

		VISUAL_num_segment_pairs = target_len;

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

			if( !(tgt_sentence.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));

				VISUAL_num_target_word_tokens+=tgt_sentence.size();

				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}


		if(data.size()%minibatch_size!=0) {
            if(BZ_CUDA::shuffle_data) {
    			std::random_shuffle(data.begin(),data.end());
            }
			int num_to_remove = data.size()%minibatch_size;
			for(int i=0; i<num_to_remove; i++) {
				data.pop_back();
			}
		}

		if(data.size()==0) {
			BZ_CUDA::logger << "ERROR: file size is zero, could be wrong input file or all lines are above max sent length\n";
			exit (EXIT_FAILURE);
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


		weights_file.close();
		final_output.close();
		target_input.close();

		BZ_CUDA::logger << "----------------------------target dev file info-----------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff: " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n";
	}



	void integerize_file_LM(std::string output_weights_name,std::string target_file_name,std::string tmp_output_name,
		int max_sent_cutoff,int minibatch_size,bool dev,int &hiddenstate_size,int &target_vocab_size,int &num_layers) 
	{


		int VISUAL_num_target_word_tokens =0;
		int VISUAL_num_segment_pairs=0;
		int VISUAL_target_longest_sent=0;

		int VISUAL_num_tokens_thrown_away=0;

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

		// if(file_input_vec.size()!=3) {
		// 	BZ_CUDA::logger << "ERROR: Neural network file format has been corrupted\n";
		// 	//exit (EXIT_FAILURE);
		// }

		num_layers = std::stoi(file_input_vec[0]);
		hiddenstate_size = std::stoi(file_input_vec[1]);
		target_vocab_size = std::stoi(file_input_vec[2]);

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

		VISUAL_num_segment_pairs = target_len;

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

			if( !(tgt_sentence.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));

				VISUAL_num_target_word_tokens+=tgt_sentence.size();

				if(VISUAL_target_longest_sent < tgt_sentence.size()) {
					VISUAL_target_longest_sent = tgt_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}


		// if(data.size()%minibatch_size!=0) {
		// 	std::random_shuffle(data.begin(),data.end());
		// 	int num_to_remove = data.size()%minibatch_size;
		// 	for(int i=0; i<num_to_remove; i++) {
		// 		data.pop_back();
		// 	}
		// }

		if(data.size()==0) {
			BZ_CUDA::logger << "ERROR: file size is zero, could be wrong input file or all lines are above max sent length\n";
			exit (EXIT_FAILURE);
		}

		compare_nonLM comp;
		int curr_index = 0;
		
		if(minibatch_size!=1) {
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

			data[i].tgt_sent.clear();
			data[i].tgt_sent_int_i = tgt_int;
			data[i].tgt_sent_int_o = tgt_int;
			data[i].tgt_sent_int_i.insert(data[i].tgt_sent_int_i.begin(),0);
			data[i].tgt_sent_int_o.push_back(1);
		}

		//now pad based on minibatch
		//if(dev) {
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
		//}


		//now add in all -1's to make the last minibatch complete
		int num_extra_to_add = minibatch_size - data.size()%minibatch_size;
		if(num_extra_to_add==minibatch_size) {
			num_extra_to_add = 0;
		}
		int target_sent_len = data.back().tgt_sent_int_i.size();
		for(int i=0; i<num_extra_to_add; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			data.push_back(comb_sent_info(src_sentence,tgt_sentence));

			std::vector<int> tgt_int_m1;
			for(int j=0; j<target_sent_len; j++) {
				tgt_int_m1.push_back(-1);
			}
			data.back().tgt_sent_int_i = tgt_int_m1;
			data.back().tgt_sent_int_o = tgt_int_m1;

		}



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


		weights_file.close();
		final_output.close();
		target_input.close();

		BZ_CUDA::logger << "----------------------------target dev file info-----------------------------\n";
		BZ_CUDA::logger << "Number of target word tokens: " << VISUAL_num_target_word_tokens <<"\n";
		BZ_CUDA::logger << "Number of target segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Longest target segment (after removing long sentences for training): " << VISUAL_target_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff: " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n";
	}

	void integerize_file_kbest(std::string output_weights_name,std::string source_file_name,std::string tmp_output_name,
		int max_sent_cutoff,int &target_vocab_size,bool multi_src_model,std::string multi_src_mapping_file) 
	{
		data.clear();
		int VISUAL_num_source_word_tokens =0;
		int VISUAL_num_segment_pairs=0;
		int VISUAL_source_longest_sent=0;
		int VISUAL_num_tokens_thrown_away=0;

		//int hiddenstate_size = -1;
		//int source_vocab_size = -1;
		//int num_layers = -1;

		std::ifstream weights_file;
		weights_file.open(output_weights_name.c_str());


		//for multi-source only
		std::ifstream multi_src_weights_file;
		if(multi_src_model) {
			multi_src_weights_file.open(multi_src_mapping_file.c_str());
		}

		std::vector<std::string> file_input_vec;
		std::string str;
		std::string word;

		std::getline(weights_file, str);
		std::istringstream iss(str, std::istringstream::in);
		while(iss >> word) {
			file_input_vec.push_back(word);
		}

		// if(file_input_vec.size()!=4) {
		// 	BZ_CUDA::logger << "ERROR: Neural network file format has been corrupted\n";
			
		// 	//exit (EXIT_FAILURE);
		// }

		//num_layers = std::stoi(file_input_vec[0]);
		//hiddenstate_size = std::stoi(file_input_vec[1]);
		target_vocab_size = std::stoi(file_input_vec[2]);
		//source_vocab_size = std::stoi(file_input_vec[3]);


		if(!multi_src_model) {
			//now get the source mappings
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
				src_mapping[word] = tmp_index;
			}
		}
		else {
			std::getline(multi_src_weights_file, str); //get this line, since all equals
			while(std::getline(multi_src_weights_file, str)) {
				int tmp_index;

				if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
					break; //done with target mapping
				}

				std::istringstream iss(str, std::istringstream::in);
				iss >> word;
				tmp_index = std::stoi(word);
				iss >> word;
				src_mapping[word] = tmp_index;
			}
		}

		//now that we have the mappings, integerize the file
		std::ofstream final_output;
		final_output.open(tmp_output_name.c_str());
		std::ifstream source_input;
		source_input.open(source_file_name.c_str());

		//first get the number of lines the the files and check to be sure they are the same
		int source_len = 0;
		std::string src_str;

		source_input.clear();

		source_input.seekg(0, std::ios::beg);
		while(std::getline(source_input, src_str)) {
			source_len++;
		}

		
		VISUAL_num_segment_pairs = source_len;

		source_input.clear();
		source_input.seekg(0, std::ios::beg);
		for(int i=0; i<source_len; i++) {
			std::vector<std::string> src_sentence;
			std::vector<std::string> tgt_sentence;
			std::getline(source_input, src_str);

			std::istringstream iss_src(src_str, std::istringstream::in);
			while(iss_src>> word) {
				src_sentence.push_back(word);
			}

			std::reverse(src_sentence.begin(),src_sentence.end());
			if( !(src_sentence.size()+1>=max_sent_cutoff-2) ) {
				data.push_back(comb_sent_info(src_sentence,tgt_sentence));
				VISUAL_num_source_word_tokens+=src_sentence.size();
				if(VISUAL_source_longest_sent < src_sentence.size()) {
					VISUAL_source_longest_sent = src_sentence.size();
				}
			}
			else {
				VISUAL_num_tokens_thrown_away+=src_sentence.size() + tgt_sentence.size();
			}
		}

		//now integerize
		for(int i=0; i<data.size(); i++) {
			std::vector<int> src_int;

			for(int j=0; j<data[i].src_sent.size(); j++) {
				if(src_mapping.count(data[i].src_sent[j])==0) {
					src_int.push_back(src_mapping["<UNK>"]);
				}
				else {
					src_int.push_back(src_mapping[data[i].src_sent[j]]);
				}	
			}
			data[i].src_sent.clear();
			data[i].src_sent_int= src_int;
		}


		for(int i=0; i<data.size(); i++) {
			for(int j=0; j<data[i].src_sent_int.size(); j++) {
				final_output << data[i].src_sent_int[j];
				if(j!=data[i].src_sent_int.size()) {
					final_output << " ";
				}
			}
			final_output << "\n";
		}

		weights_file.close();
		final_output.close();
		source_input.close();

		BZ_CUDA::logger << "----------------------------source kbest file info-----------------------------\n";
		BZ_CUDA::logger << "Number of source word tokens: " << VISUAL_num_source_word_tokens <<"\n";
		BZ_CUDA::logger << "Number of source segment pairs (lines in training file): " << VISUAL_num_segment_pairs<<"\n";
		BZ_CUDA::logger << "Longest source segment (after removing long sentences for training): " << VISUAL_source_longest_sent << "\n";
		BZ_CUDA::logger << "Total word tokens thrown out due to sentence cutoff: " << VISUAL_num_tokens_thrown_away <<"\n";
		BZ_CUDA::logger << "-------------------------------------------------------------------------------\n\n";
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

    void load_word_index_mapping(std::string output_weights_name,bool LM,bool decoder){
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
                int tmp_index;
                if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
                    break; //done with source mapping
                }
                std::istringstream iss(str, std::istringstream::in);
                iss >> word;
                tmp_index = std::stoi(word);
                iss >> word;
                src_reverse_mapping[tmp_index] = word;
                src_mapping[word] = tmp_index;
                
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
            tgt_mapping[word] = tmp_index;
        }
        
        
        weights_file.close();
        
    }

    

	void unint_alignments(std::string output_weights_name,std::string int_alignments_file,std::string final_alignment_file) {
		std::ifstream weights_file;
		weights_file.open(output_weights_name.c_str());
		weights_file.clear();
		weights_file.seekg(0, std::ios::beg);

		std::string str;
		std::string word;

		std::getline(weights_file, str); //info from first sentence
		std::getline(weights_file, str); //======== stuff

		while(std::getline(weights_file, str)) {
			int tmp_index;

			if(str.size()>3 && str[0]=='=' && str[1]=='=' && str[2]=='=') {
				break; //done with target mapping
			}
			std::istringstream iss(str, std::istringstream::in);
			iss >> word;
			tmp_index = std::stoi(word);
			iss >> word;
			src_reverse_mapping[tmp_index] = word;
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
		unint.open(int_alignments_file.c_str());

		std::ofstream final_output;
		final_output.open(final_alignment_file.c_str());

		//goes source the target, so get source from the loop
		while(std::getline(unint, str)) {

			std::istringstream iss(str, std::istringstream::in);
			std::vector<int> sent_int_src;
			std::vector<int> sent_int_tgt;

			while(iss >> word) {
				sent_int_src.push_back(std::stoi(word));
			}

			//now the stuff for the target
			std::getline(unint, str);
			std::istringstream iss_2(str, std::istringstream::in);


			while(iss_2 >> word) {
				sent_int_tgt.push_back(std::stoi(word));
			}

			for(int i=0; i<sent_int_src.size(); i++) {
				final_output << src_reverse_mapping[sent_int_src[i]];
				if(i!=sent_int_src.size()-1) {
					final_output << " ";
				}
			}
			final_output << "\n";

			for(int i=0; i<sent_int_tgt.size(); i++) {
				final_output << tgt_reverse_mapping[sent_int_tgt[i]];
				if(i!=sent_int_tgt.size()-1) {
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
