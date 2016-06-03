#ifndef DECODER_H
#define DECODER_H

#include <queue>
#include <vector>
#include <fstream>
#include <utility> 
#include <float.h>

//The decoder object type
template<typename dType>
struct dec_global_obj {

	dType val;
	int beam_index;
	int vocab_index;

	int viterbi_alignment;

	dec_global_obj(dType _val,int _beam_index,int _vocab_index,int _viterbi_alignment) {
		val = _val;
		beam_index = _beam_index;
		vocab_index = _vocab_index;
		viterbi_alignment = _viterbi_alignment;
	}
};

template<typename dType>
struct dec_obj {

	dType val;
	int vocab_index;

	int viterbi_alignment;

	dec_obj(dType _val,int _vocab_index,int _viterbi_alignment) {
		val = _val;
		vocab_index = _vocab_index;
		viterbi_alignment = _viterbi_alignment;
	}
};


template<typename dType>
struct k_best {
	dType score;
	dType index;
	k_best(dType _score,dType _index) {
		score = _score;
		index = _index;
	}
};

template<typename dType>
struct eigen_mat_wrapper {
	Eigen::Matrix<int, Eigen::Dynamic,1> hypothesis;
	Eigen::Matrix<int, Eigen::Dynamic,1> viterbi_alignments;

	dType score; //log prob score along with a penalty

	eigen_mat_wrapper(int size) {
		hypothesis.resize(size);
		viterbi_alignments.resize(size);
	}
};

bool compare_pq(dec_obj<float> &a,dec_obj<float> &b)
{
   return (a.val < b.val);
}

struct pq_compare_functor {
	template<typename dType>
  	bool operator() (dec_obj<dType> &a,dec_obj<dType> &b) const { return (a.val < b.val); }
};

struct pq_global_compare_functor {
	template<typename dType>
  	bool operator() (dec_global_obj<dType> &a,dec_global_obj<dType> &b) const { return (a.val < b.val); }
};

struct k_best_compare_functor {
	template<typename dType>
  	bool operator() (k_best<dType> &a,k_best<dType> &b) const { return (a.score < b.score); }
};

template<typename dType>
struct decoder {

	//global
	int beam_size;
	int vocab_size;
	int start_symbol;
	int end_symbol;
	int max_decoding_length; //max size of a translation
	dType min_decoding_ratio; //min size of a translation
	int current_index; //The current length of the decoded target sentence
	int num_hypotheses; //The number of hypotheses to be output for each translation
	dType penalty;//penality term to encourage longer hypotheses, tune for bleu score
	std::string output_file_name;
	std::ofstream output;
	bool print_score;


	std::priority_queue<dec_obj<dType>,std::vector<dec_obj<dType>>, pq_compare_functor> pq;
	std::priority_queue<dec_global_obj<dType>,std::vector<dec_global_obj<dType>>, pq_global_compare_functor> pq_global;

	std::vector<eigen_mat_wrapper<dType>> hypotheses; //Stores all hypotheses

	//CPU
	Eigen::Matrix<int, Eigen::Dynamic,1> current_indices; //stores the current indicies for the beam size

	//Size of (beam size)x(beam size)
	//Each row is the indicies for one old hypothesis
	//Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> top_words;

	//size (beam size)x(max decoder length)
	Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic> top_sentences;
	Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic> top_sentences_temp; //Temp to copy old ones into this

	//size (beam size)x(max decoder length)
	Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic> top_sentences_viterbi;
	Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic> top_sentences_temp_viterbi; //Temp to copy old ones into this

	//size (beam size)x1, score are stored as log probabilities
	Eigen::Matrix<dType,Eigen::Dynamic, 1> top_sentences_scores;
	Eigen::Matrix<dType,Eigen::Dynamic, 1> top_sentences_scores_temp;

	Eigen::Matrix<int,Eigen::Dynamic, 1> new_indicies_changes; //used to swap around hidden and cell states based on new beam results

	//GPU
	int *h_current_indices;
	int *d_current_indices;

	decoder(int beam_size,int vocab_size,int start_symbol,int end_symbol,int max_decoding_length,dType min_decoding_ratio,
		dType penalty,std::string output_file_name,int num_hypotheses,bool print_score) 
	{
		this->beam_size = beam_size;
		this->vocab_size = vocab_size;
		this->start_symbol = start_symbol;
		this->end_symbol = end_symbol;
		this->max_decoding_length = max_decoding_length;
		this->min_decoding_ratio = min_decoding_ratio;
		this->penalty = penalty;
		this->output_file_name = output_file_name;
		this->num_hypotheses = num_hypotheses;
		this->print_score = print_score;
		BZ_CUDA::logger << "Output file name for decoder: " << output_file_name << "\n";
		output.open(output_file_name.c_str());

		current_indices.resize(beam_size);
		//top_words.resize(beam_size,beam_size);
		top_sentences.resize(beam_size,max_decoding_length);
		top_sentences_temp.resize(beam_size,max_decoding_length);
		top_sentences_viterbi.resize(beam_size,max_decoding_length);
		top_sentences_temp_viterbi.resize(beam_size,max_decoding_length);
		top_sentences_scores.resize(beam_size);
		top_sentences_scores_temp.resize(beam_size);
		new_indicies_changes.resize(beam_size);

		h_current_indices = (int *)malloc(beam_size*1*sizeof(int));
		//cudaMalloc((void**)&d_current_indices,beam_size*1*sizeof(int));//put void**
	}

	~decoder() {
		output.close();
		//cudaFree(d_current_indices);
	}

	void empty_queue_pq() {
		while(!pq.empty()) {
			pq.pop();
		}
	}

	void empty_queue_global() {
		while(!pq_global.empty()) {
			pq_global.pop();
		}
	}

	template<typename Derived>
	void finish_current_hypotheses(const Eigen::MatrixBase<Derived> &outputDist,std::vector<int> &viterbi_alignments) {

		for(int i=0; i<beam_size; i++) {
			top_sentences(i,current_index+1) = end_symbol;
			top_sentences_scores(i) += std::log(outputDist(1,i)) + penalty;
			hypotheses.push_back(eigen_mat_wrapper<dType>(current_index+2));
			hypotheses.back().hypothesis = top_sentences.block(i,0,1,current_index+2).transpose();//.row(temp.beam_index);
			hypotheses.back().viterbi_alignments = top_sentences_viterbi.block(i,0,1,current_index+2).transpose();
			hypotheses.back().viterbi_alignments(current_index+1) = viterbi_alignments[i];
			hypotheses.back().score = top_sentences_scores(i);
		}
		current_index+=1;
	}

	template<typename Derived>
	void expand_hypothesis(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments) {
		

		if(viterbi_alignments.size()!=0 && viterbi_alignments.size()!=beam_size) {
			BZ_CUDA::logger << "VITERBI ALIGNMENT ERROR\n";
			exit (EXIT_FAILURE);
		}

		if(viterbi_alignments.size()==0) {

			for(int i=0; i<beam_size; i++) {
				viterbi_alignments.push_back(-1);
			}
		}

		int cols=outputDist.cols();
		if(index==0) {
			cols = 1;
		}

		empty_queue_global();
		for(int i=0; i<cols; i++) {
			empty_queue_pq();
			for(int j=0; j<outputDist.rows(); j++) {
				if(pq.size() < beam_size + 1) {
					pq.push( dec_obj<dType>(-outputDist(j,i),j,viterbi_alignments[i]) );
				}
				else {
					if(-outputDist(j,i) < pq.top().val) {
						pq.pop();
						pq.push( dec_obj<dType>(-outputDist(j,i),j,viterbi_alignments[i]) );
					}
				}
			}
			//Now have the top elements
			while(!pq.empty()) {
				dec_obj<dType> temp = pq.top();
				 pq.pop();
				//pq_global.push( dec_global_obj<dType>(-temp.val,i,temp.vocab_index) );
				pq_global.push( dec_global_obj<dType>(std::log(-temp.val) + top_sentences_scores(i),i,temp.vocab_index,temp.viterbi_alignment) );
			}
		}

		//Now have global heap with (beam size*beam size) elements
		//Go through until (beam size) new hypotheses.
		int i = 0;
		while(i < beam_size) {
			dec_global_obj<dType> temp = pq_global.top();
			pq_global.pop();
			//std::cout <<"Value: " << temp.val << " Index: " << temp.vocab_index << " Beam: "<< temp.beam_index << "\n\n";
			if(temp.vocab_index!=start_symbol) {
				if(temp.vocab_index==end_symbol) {
					hypotheses.push_back(eigen_mat_wrapper<dType>(current_index+2));
					hypotheses.back().hypothesis = top_sentences.block(temp.beam_index,0,1,current_index+2).transpose();//.row(temp.beam_index);
					hypotheses.back().hypothesis(current_index+1) = end_symbol;

					hypotheses.back().viterbi_alignments = top_sentences_viterbi.block(temp.beam_index,0,1,current_index+2).transpose();//.row(temp.beam_index);
					hypotheses.back().viterbi_alignments(current_index+1) = temp.viterbi_alignment;
					//hypotheses.back().score = std::log(temp.val) /*+ top_sentences_scores(temp.beam_index)*/ + penalty;
					hypotheses.back().score = temp.val + penalty;
				}
				else {
					top_sentences_temp.row(i) = top_sentences.row(temp.beam_index);
					top_sentences_temp(i,current_index+1) = temp.vocab_index;

					top_sentences_temp_viterbi.row(i) = top_sentences_viterbi.row(temp.beam_index);
					top_sentences_temp_viterbi(i,current_index+1) = temp.viterbi_alignment;

					current_indices(i) = temp.vocab_index;
					new_indicies_changes(i) = temp.beam_index;
					// if(top_sentences_scores(temp.beam_index)!=0) {
					// 	//top_sentences_scores_temp(i) = std::log(temp.val) + top_sentences_scores(temp.beam_index) + penalty;
					// 	top_sentences_scores_temp(i) = temp.val + penalty;
					// }
					// else {
					// 	//top_sentences_scores_temp(i) = std::log(temp.val) + penalty;
					// 	top_sentences_scores_temp(i) =temp.val + penalty;
					// }
					top_sentences_scores_temp(i) =temp.val + penalty;
					i++;
				}
			}
		}

		top_sentences = top_sentences_temp;
		top_sentences_viterbi = top_sentences_temp_viterbi;
		top_sentences_scores = top_sentences_scores_temp;
		current_index += 1;

		// std::cout << "--------------- top sentences viterbi ---------------\n";
		// for(int i=0; i<beam_size; i++) {
		// 	std::cout << top_sentences_viterbi.row(i) << "\n\n\n\n\n\n";
		// 	std::cout << top_sentences.row(i) << "\n\n\n\n\n\n";
		// }

		for(int i=0; i<beam_size; i++) {
			h_current_indices[i] = current_indices(i);
		}
		//cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice);
	}
	
	void init_decoder() {

		current_index = 0;
		top_sentences_scores.setZero();
		hypotheses.clear();

		for(int i=0; i<beam_size; i++) {
			current_indices(i) = start_symbol;
			h_current_indices[i] = start_symbol;
		}

		for(int i=0; i<beam_size; i++) {
			for(int j=0; j<max_decoding_length; j++) {
				top_sentences(i,j) = start_symbol;
				top_sentences_viterbi(i,j) = -20;
			}
		}

		//cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice);
	}

	void print_current_hypotheses() {

		// std::cout << "Printing out current indicies"<<std::endl;
		// for(int i=0; i< current_indices.size(); i++) {
		// 	std::cout << current_indices[i] << " ";
		// }
		// std::cout << "\n\n";

		BZ_CUDA::logger << "Printing out finished hypotheses" << "\n";
		BZ_CUDA::logger << "Number of hyptheses: " << hypotheses.size() << "\n";
		for(int i=0; i<hypotheses.size(); i++) {
			BZ_CUDA::logger << "Score of hypothesis " << hypotheses[i].score << "\n";
			BZ_CUDA::logger << hypotheses[i].hypothesis.transpose() << "\n\n\n";
		}

		BZ_CUDA::logger << "Printing out in-progress hypotheses: " << "\n";
		for(int i=0; i<top_sentences.rows();i++) {
			for(int j=0; j <= current_index; j++) {
				BZ_CUDA::logger << top_sentences(i,j) << " ";
			}
			BZ_CUDA::logger << "\n";
		}
		BZ_CUDA::logger << "\n";

		// std::cout << "Printing out beam changes\n";
		// std::cout << new_indicies_changes << "\n\n";
	}


	void output_k_best_hypotheses(int source_length) {

		std::priority_queue<k_best<dType>,std::vector<k_best<dType>>, k_best_compare_functor> best_hypoth;

		//dType max_val = -DBL_MAX;
		//dType max_val = -FLT_MAX;
		//int max_index = -1;
		dType len_ratio;
		for(int i=0; i<hypotheses.size(); i++) {
			len_ratio = ((dType)hypotheses[i].hypothesis.size())/source_length;
			if(len_ratio > min_decoding_ratio) {
				if(best_hypoth.size() < num_hypotheses) {
					best_hypoth.push( k_best<dType>(-hypotheses[i].score,i) );
				}
				else {
					if(-1*best_hypoth.top().score < hypotheses[i].score) {
						best_hypoth.pop();
						best_hypoth.push( k_best<dType>(-hypotheses[i].score,i) );
					}
				}
			}
		}
		//for making k-best list descending 
		std::priority_queue<k_best<dType>,std::vector<k_best<dType>>, k_best_compare_functor> best_hypoth_temp;
		while(!best_hypoth.empty()) {
			best_hypoth_temp.push( k_best<dType>(-1*best_hypoth.top().score,best_hypoth.top().index) );
			best_hypoth.pop();
		}
		
		output << "------------------------------------------------\n";
		while(!best_hypoth_temp.empty()) {

			if(print_score) {
				output << "-Score: " <<hypotheses[best_hypoth_temp.top().index].score << "\n";
			}
			for(int j=0; j<hypotheses[best_hypoth_temp.top().index].hypothesis.size(); j++) {
				output << hypotheses[best_hypoth_temp.top().index].hypothesis(j) << " ";
			}
			output << "\n";

			if(BZ_CUDA::unk_replacement) {
				for(int j=0; j<hypotheses[best_hypoth_temp.top().index].hypothesis.size(); j++) {
					if(hypotheses[best_hypoth_temp.top().index].hypothesis(j) == 2) {
						BZ_CUDA::unk_rep_file_stream << hypotheses[best_hypoth_temp.top().index].viterbi_alignments(j+1) << " ";
					}
				}
				BZ_CUDA::unk_rep_file_stream << "\n";
				BZ_CUDA::unk_rep_file_stream.flush();
			}
			best_hypoth_temp.pop();
		}
		output << "\n";

		output.flush();



		// output << "------------------------------------------------\n";
		// while(!best_hypoth.empty()) {

		// 	if(print_score) {
		// 		output << "Score: " <<-1*hypotheses[best_hypoth.top().index].score << "\n";
		// 	}
		// 	for(int j=0; j<hypotheses[best_hypoth.top().index].hypothesis.size(); j++) {
		// 		output << hypotheses[best_hypoth.top().index].hypothesis(j) << " ";
		// 	}
		// 	output << "\n";
		// 	best_hypoth.pop();
		// }
		// if(num_hypotheses>1) {
		// 	output << "------------------------------------------------\n\n";
		// }
	}

	// void output_k_best_hypotheses(int source_length) {
	// 	dType max_val = -DBL_MAX;
	// 	int max_index = -1;
	// 	dType len_ratio;
	// 	for(int i=0; i<hypotheses.size(); i++) {
	// 		len_ratio = ((dType)hypotheses[i].hypothesis.size())/source_length;
	// 		if( (hypotheses[i].score > max_val) && (len_ratio > min_decoding_ratio) ) {
	// 			max_val = hypotheses[i].score;
	// 			max_index = i;
	// 		}
	// 	}
	// 	for(int i=0; i<hypotheses[max_index].hypothesis.size(); i++) {
	// 		output << hypotheses[max_index].hypothesis(i) << " ";
	// 	}
	// 	output << "\n\n";
	// }
};

#endif
