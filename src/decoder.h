#ifndef DECODER_H
#define DECODER_H

#include <queue>
#include <vector>
#include <fstream>
#include <utility> 
#include <float.h>
#include "fsa.hpp"
#include "format.h"
#include "memory_util.h"
#include "custom_kernels.h"
#include "BZ_CUDA_UTIL.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>

//MARK:FSA related obj

template<typename dType>
class neuralMT_model;

//MARK: decoding related obj

//The decoder object type
template<typename dType>
struct dec_global_obj {

	dType val;
    dType score;
	int beam_index;
	int vocab_index;
    state *s;
	int viterbi_alignment;

	dec_global_obj(dType _val,int _beam_index,int _vocab_index,int _viterbi_alignment) {
		val = _val;
		beam_index = _beam_index;
		vocab_index = _vocab_index;
		viterbi_alignment = _viterbi_alignment;
    }
    
    dec_global_obj(){}
    
    bool operator==(const dec_global_obj &other) const
    {
        return (val == other.val
                && score == other.score
                && vocab_index == other.vocab_index
                && s->name == other.s->name );
	}
};

template<typename dType>
struct dec_obj {

	dType val;
    dType score;
	int vocab_index;
    state *s;
	int viterbi_alignment;

	dec_obj(dType _val,int _vocab_index,int _viterbi_alignment) {
		val = _val;
		vocab_index = _vocab_index;
		viterbi_alignment = _viterbi_alignment;
	}
    
    bool operator==(const dec_obj &other) const
    {
        return (val == other.val
                && score == other.score
                && vocab_index == other.vocab_index
                && s->name == other.s->name );
    }
    
};

namespace std {
    
    template<typename dType>
    struct hash<dec_obj<dType>>
    {
        size_t operator()(const dec_obj<dType>& k) const
        {
            return (hash<float>()(k.val))
            ^ (hash<float>()(k.score) << 1)
            ^ (hash<int>()(k.vocab_index) << 2)
            ^ (hash<string>()(k.s->name) << 3) ;
        }
    };
    
    
    template<typename dType>
    struct hash<dec_global_obj<dType>>
    {
        size_t operator()(const dec_global_obj<dType>& k) const
        {
            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:
            return (hash<float>()(k.val)) ^
            (hash<float>()(k.score) << 1) ^
            (hash<int>()(k.vocab_index) << 2) ^
            (hash<string>()(k.s->name) << 3) ;
            
        }
    };
    
}

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

    // ct[0], ht[0], ct[1], ht[1]; just for one model;
    std::vector<Eigen::Matrix<dType, Eigen::Dynamic, 1>> chts;
    
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
    const int invalid_symbol = -1;
	int max_decoding_length; //max size of a translation
	dType min_decoding_ratio; //min size of a translation
	int current_index; //The current length of the decoded target sentence
	int num_hypotheses; //The number of hypotheses to be output for each translation
	dType penalty;//penality term to encourage longer hypotheses, tune for bleu score
	std::string output_file_name;
	std::ofstream output;
	bool print_score;
    bool print_beam = false;
    neuralMT_model<dType> *model = NULL;
    //ThreadPool *pool;

    //--- for fsa integration ---
    float fsa_weight = 0.0;
    fsa* fsa_model;
    bool with_fsa = false;
    bool with_fsa_compress = false;
    bool fsa_can_prune = false;
    bool fsa_log = true;
    bool merge_state = true;
    int invalid_number = 0;
    bool end_transfer = false; // true if in fsa_line mode

    //Timer
    Timer timer;
    
    // other weight
    float alliteration_weight = 0.0;
    float wordlen_weight = 0.0;
    
    std::unordered_map<std::string,int> tgt_mapping;
    
    std::vector<state*> current_states;
    
    
	std::priority_queue<dec_obj<dType>,std::vector<dec_obj<dType>>, pq_compare_functor> pq;
	std::priority_queue<dec_global_obj<dType>,std::vector<dec_global_obj<dType>>, pq_global_compare_functor> pq_global;
    std::priority_queue<dec_global_obj<dType>,std::vector<dec_global_obj<dType>>, pq_global_compare_functor> pqg; // used in expand_pq_global_gpu

    std::unordered_set<dec_obj<dType>> pq_set;
    std::unordered_set<dec_global_obj<dType>> pq_global_set;
    std::unordered_set<dec_global_obj<dType>> pqg_set;
    
    
    
    
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
    
    
    
    //for repeat_penalty
    bool penalize_repeat = true; // always true, but controlled by the weights;
    precision repeat_penalty = 0.0;
    float interactive_repeat_penalty = 0.0;
    precision adjacent_repeat_penalty = 0.0;
    std::vector<std::unordered_map<int,int>> sentence_sets;
    std::vector<std::unordered_map<int,int>> temp_sentence_sets;

    //for source_length
    int source_length = 0;
    
	//GPU
	int *h_current_indices;
	int *d_current_indices;
    
    // for vocab_shrink_2
    int *h_current_indices_original;
    int target_vocab_policy = 0;
    // for LSH_WTA
    int nnz = 0;
    int *h_rowIdx;

    dType *h_outputdist; // [vocab_size, beam_size] point to models[0].h_outputdist;
    dType *d_outputdist; // need to allocate;
    dType *d_outputdist_topk; // [vocab_size, beam_size] need to allocate;
    dType *h_outputdist_topk; // need to allocate;
    
    // to save vocab_index
    int *d_dict;    // [vocab_size, beam_size] need to allocate;
    int *h_dict;    // need to allocate;
    
    // to save pointer
    int *d_pointers;
    int *h_pointers;
    // to save beam_index
    int *d_beams;
    int *h_beams;
    // to save valid_vocab_size;
    int *d_valid_vocab_sizes;
    int *h_valid_vocab_sizes;
    
    // to save top_sentence_score;
    dType *d_sentence_scores;
    dType *h_sentence_scores;
    // to save wordlen
    int *d_wordlen; //[vocab_size]
    int *h_wordlen;
    // to save alliteration information, same word would have save bin number;
    int *d_vocab_bin; // [vocab_size]
    int *h_vocab_bin;
    
    // to save repeat informaiton;
    int *d_sentence_set; // beam_size * max_decoding_length * 3 [vocab, beam_index, occur_times]
    int *h_sentence_set;
    
    // to save encourange list
    dType *h_encourage;
    dType *d_encourage;
    
    
    std::vector<cudaStream_t> streams;
    
	decoder(int beam_size,int vocab_size,int start_symbol,int end_symbol,int max_decoding_length,dType min_decoding_ratio,
            dType penalty,std::string output_file_name,int num_hypotheses,bool print_score, global_params &params)
	{
		this->beam_size = beam_size;
		this->vocab_size = vocab_size;
		this->start_symbol = start_symbol;
		this->end_symbol = end_symbol;
		this->max_decoding_length = max_decoding_length;
		this->min_decoding_ratio = min_decoding_ratio;
		this->penalty = penalty;
        this->repeat_penalty = params.repeat_penalty;
        this->adjacent_repeat_penalty =  params.adjacent_repeat_penalty;
        this->wordlen_weight = params.wordlen_weight;
        this->alliteration_weight = params.alliteration_weight;
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
		cudaMalloc((void**)&d_current_indices,beam_size*1*sizeof(int));//put void**

        // repeat_penalty
        for (int i = 0; i<beam_size; i++){
            std::unordered_map<int,int>* sentence_set = new std::unordered_map<int,int>();
            sentence_sets.push_back(*sentence_set);
            temp_sentence_sets.push_back(*sentence_set);
        }
        
        // fsa
        this->fsa_model = NULL;
        
        
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_dict, vocab_size*beam_size*1*sizeof(int)),"d_dict allocation failed\n");
        h_dict = (int *)malloc(vocab_size*beam_size*1*sizeof(int));
        //CUDA_ERROR_WRAPPER(cudaHostRegister(h_dict, vocab_size * beam_size * sizeof(int), cudaHostRegisterPortable),"h_dict pinned memeory error!");
        
        
        // d_pointers
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_pointers, vocab_size*beam_size*1*sizeof(int)),"d_pointers allocation failed\n");
        h_pointers = (int *)malloc(vocab_size*beam_size*1*sizeof(int));
        //CUDA_ERROR_WRAPPER(cudaHostRegister(h_pointers, vocab_size * beam_size * sizeof(int), cudaHostRegisterPortable),"h_pointers pinned memeory error!");
        
        // d_beams
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_beams, vocab_size*beam_size*1*sizeof(int)),"d_beams allocation failed\n");
        h_beams = (int *)malloc(vocab_size*beam_size*1*sizeof(int));
        //CUDA_ERROR_WRAPPER(cudaHostRegister(h_beams, vocab_size * beam_size * sizeof(int), cudaHostRegisterPortable),"h_beams pinned memeory error!");

        // d_valid_vocab_sizes
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_valid_vocab_sizes, (beam_size+1)*1*sizeof(int)),"d_valid_vocab_sizes allocation failed\n");
        h_valid_vocab_sizes = (int *)malloc((beam_size+1)*1*sizeof(int));
        //CUDA_ERROR_WRAPPER(cudaHostRegister(h_valid_vocab_sizes, (beam_size+1) * sizeof(int), cudaHostRegisterPortable),"h_valid_vocab_sizes pinned memeory error!");
        
        // d_sentence_scores
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_sentence_scores, beam_size*sizeof(dType)),"d_sentence_scores in decoder allocation failed\n");
        h_sentence_scores = (dType *)malloc(beam_size*1*sizeof(dType));
        
        // d_outputdist
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist, beam_size*vocab_size*sizeof(dType)),"d_outputdist in decoder allocation failed\n");

        // d_outputdist_topk
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist_topk, vocab_size*beam_size*sizeof(dType)),"d_outputdist_topk in decoder allocation failed\n");

        h_outputdist_topk = (dType *)malloc(vocab_size*beam_size*1*sizeof(dType));
        //CUDA_ERROR_WRAPPER(cudaHostRegister(h_outputdist_topk, vocab_size * beam_size * sizeof(dType), cudaHostRegisterPortable),"h_outputdist_topk pinned memeory error!");
        
        // d_wordlen
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_wordlen, vocab_size*1*sizeof(int)),"d_wordlen allocation failed\n");
        h_wordlen = (int *)malloc(vocab_size*1*sizeof(int));
        
        // d_vocab_bin
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_vocab_bin, vocab_size*1*sizeof(int)),"d_vocab_bin allocation failed\n");
        h_vocab_bin = (int *)malloc(vocab_size*1*sizeof(int));
        
        // d_sentence_set
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_sentence_set, beam_size * max_decoding_length*3*sizeof(int)),"d_sentence_set allocation failed\n");
        h_sentence_set = (int *)malloc(beam_size * max_decoding_length*3*sizeof(int));
        
        // d_encourage
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_encourage, vocab_size*1*sizeof(dType)),"d_encourage allocation failed\n");
        h_encourage = (dType *)malloc(vocab_size*1*sizeof(dType));
        
        
        // for cuda streams at least 10 beams;
        for (int i=0; i < std::max(beam_size,10); i++){
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams.push_back(stream);
        }
        
        // for target_vocab_shrink_policy
        this->target_vocab_policy = params.target_vocab_policy;
        if (this->target_vocab_policy == 2){
            this->h_current_indices_original = (int*) malloc(beam_size * sizeof(int));
        }
        
    }
    
	~decoder() {
		output.close();
        free(h_current_indices);
        
        cudaFree(d_outputdist);
        
        free(h_outputdist_topk);
        cudaFree(d_outputdist_topk);

        free(h_dict);
        cudaFree(d_dict);
        
        
        free(h_pointers);
        cudaFree(d_pointers);
        
        free(h_beams);
        cudaFree(d_beams);
        
        free(h_valid_vocab_sizes);
        cudaFree(d_valid_vocab_sizes);
        
        free(h_sentence_scores);
        cudaFree(d_sentence_scores);
        
        free(h_wordlen);
        cudaFree(d_wordlen);
        
        free(h_vocab_bin);
        cudaFree(d_vocab_bin);
        
        free(h_sentence_set);
        cudaFree(d_sentence_set);
        
        free(h_encourage);
        cudaFree(d_encourage);
        
        
        //for streams
        for (int i=0; i < std::max(beam_size,10); i++){
            cudaStreamDestroy(streams[i]);
        }
        
        if (this->target_vocab_policy == 2){
            free(h_current_indices_original);
        }

        
    }
    // for encourage list
    
    void init_encourage_lists(std::vector<std::string> fns, std::vector<float> weights){

        // even though len(fns) == 0, we still init h_encourage and d_encourage;
        
        for (int i = 0; i < vocab_size; i ++ ){
            h_encourage[i] = 0.0;
        }

        for (int i = 0; i< fns.size(); i++){
            std::string encourage_file = fns[i];
            float weight = weights[i];
            this->init_encourage_list(encourage_file, weight);
            
        }
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_encourage, h_encourage,
                                      vocab_size*sizeof(dType),
                                      cudaMemcpyHostToDevice),
                           " h_encourage to d_encourage\n");


    }
    
    void init_encourage_list(std::string fn, float weight){
        
        // should call after init_fsa();
        std::ifstream fin(fn.c_str());
        std::string line;
        int n_nounk = 0;
        while(std::getline(fin,line)){
            
            std::vector<std::string> ll = split(line,' ');
            float i_weight = 1.0;
            std::string word = ll[0];
            if (ll.size() == 2){
                i_weight = std::stof(ll[1]);
            }
            int index = 2 ; // <UNK>
            if (this->tgt_mapping.count(word) > 0){
                index = this->tgt_mapping[word];
            }
            
            if (index != 2){
                //std::cout << word << " " << index << " " << i_weight << "\n";
                h_encourage[index] += weight * i_weight;
                n_nounk += 1;
            }
            
        }
        fin.close();
        
        BZ_CUDA::logger<< "Encourage Weight: " << weight <<"\n";
        BZ_CUDA::logger<< "Encourage List Size: " << n_nounk <<"\n";
    }
    
    // for single fsa file
    void init_fsa_inner(global_params &params){
        this->fsa_weight = params.fsa_weight;
        this->fsa_log = params.fsa_log;
        this->with_fsa = true;
        if (this->fsa_weight >=0){
            // also, the log(weight) on fsa's edge should < 0.0;
            this->fsa_can_prune = true;
        }
    }
    
    void init_fsa(fsa *fsa_model, std::unordered_map<std::string,int> &tgt_mapping,global_params &params){
        
        this->init_fsa_inner(params);
        
        this->tgt_mapping = tgt_mapping;
        this->fsa_model = fsa_model;
        if (!params.interactive && !params.interactive_line){ // if it's interactive, this fsa_model is non_valid
            this->load_fsa_model();
        }
        
        std::unordered_map<std::string, int> bins;
        int bin_index =0 ;
        // prepare vocab_bin and wordlen
        for (const auto & item : tgt_mapping){
            std::string word = item.first;
            std::string key = std::string(1,word[0]);
            int vocab_index = item.second;
            h_wordlen[vocab_index] = word.size();
            if (bins.count(key) == 0){
                bins[key] = bin_index;
                bin_index += 1;
            }
            h_vocab_bin[vocab_index] = bins[key];
        }
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_wordlen, h_wordlen,
                                      vocab_size*sizeof(int),
                                      cudaMemcpyHostToDevice),
                           " h_wordlen to d_wordlen\n");
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_vocab_bin, h_vocab_bin,
                                      vocab_size*sizeof(int),
                                      cudaMemcpyHostToDevice),
                           " h_vocab_bin to d_vocab_bin\n");

        
        
    }
    
    void load_fsa_model()
    {
        this->fsa_model->convert_name_to_index(this->tgt_mapping);
        this->fsa_model->log_space = this->fsa_log;
        this->fsa_model->load_fsa();
    }
    
    
    void init_fsa_interactive(std::string fn_fsa){
        if (this->fsa_model != NULL){
            delete this->fsa_model;
            this->fsa_model = NULL;
        }
        
        this->fsa_model = new fsa(fn_fsa);
        this->load_fsa_model();
        
    }
    
    void empty_queue_pq(std::priority_queue<dec_obj<dType>,std::vector<dec_obj<dType>>, pq_compare_functor> &pq, std::unordered_set<dec_obj<dType>> &pq_set) {
        while(!pq.empty()) {
            pq.pop();
        }
        if (merge_state){
            pq_set.clear();
        }
    }
    
	void empty_queue_pq() {
		while(!pq.empty()) {
			pq.pop();
		}
        if (merge_state){
            pq_set.clear();
	}
    }

    
    void empty_queue_global(std::priority_queue<dec_global_obj<dType>,std::vector<dec_global_obj<dType>>, pq_global_compare_functor> &pq, std::unordered_set<dec_global_obj<dType>> &pq_set) {
        while(!pq.empty()) {
            pq.pop();
        }
        if (merge_state){
            pq_set.clear();
        }
    }

    
	void empty_queue_global() {
		while(!pq_global.empty()) {
			pq_global.pop();
        }
        if (merge_state){
            pq_global_set.clear();
		}
	}

	template<typename Derived>
	void finish_current_hypotheses(const Eigen::MatrixBase<Derived> &outputDist,std::vector<int> &viterbi_alignments) {
        if (this->with_fsa){
            for(int i=0; i<beam_size; i++) {
                
                int symbol = this->current_indices(i);
                if (symbol == this->invalid_symbol){
                    continue;
                }
                
                state* istate = this->current_states[i];
                std::vector<sw> sws;
                this->fsa_model->next_states(istate,end_symbol,sws);
                for (auto const & s: sws){
                    if ( s.s->name == this->fsa_model->end_state->name){
                        
                        // here start_symbol means it didn't end naturally.
                        top_sentences(i,current_index+1) = start_symbol;
                        
                        dType base_score = std::log(outputDist(end_symbol,i));
                        
                        dType fsa_score = 0.0;
                        
                        fsa_score = this->fsa_weight * s.weight;
                        
                        base_score += fsa_score;
                        
                        if (!end_transfer){
                            top_sentences_scores(i) += base_score + penalty;
                        }
                        
                        hypotheses.push_back(eigen_mat_wrapper<dType>(current_index+2));
                        hypotheses.back().hypothesis = top_sentences.block(i,0,1,current_index+2).transpose();//.row(temp.beam_index);
                        hypotheses.back().viterbi_alignments = top_sentences_viterbi.block(i,0,1,current_index+2).transpose();
                        hypotheses.back().viterbi_alignments(current_index+1) = viterbi_alignments[i];
                        hypotheses.back().score = top_sentences_scores(i);
                        
                        if (end_transfer){
                            // [HERE]
                            model->get_chts(hypotheses.back().chts, i, beam_size);
                        }
                        
                        break;
                    }
                }
            }
        } else{
            
            for(int i=0; i<beam_size; i++) {
                
                int symbol = this->current_indices(i);
                if (symbol == this->invalid_symbol){
                    continue;
                }
                
                top_sentences(i,current_index+1) = end_symbol;
                top_sentences_scores(i) += std::log(outputDist(1,i)) + penalty;
                hypotheses.push_back(eigen_mat_wrapper<dType>(current_index+2));
                hypotheses.back().hypothesis = top_sentences.block(i,0,1,current_index+2).transpose();//.row(temp.beam_index);
                hypotheses.back().viterbi_alignments = top_sentences_viterbi.block(i,0,1,current_index+2).transpose();
                hypotheses.back().viterbi_alignments(current_index+1) = viterbi_alignments[i];
                hypotheses.back().score = top_sentences_scores(i);
            }
        }
		current_index+=1;
	}

    // expand_hypothesis
    template<typename Derived>
    void expand_hypothesis(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments, dType *h_outputdist) {
        this->h_outputdist = h_outputdist;
        if (this->with_fsa){
            expand_hypothesis_with_fsa(outputDist,index, viterbi_alignments);
        } else {
            expand_hypothesis_without_fsa(outputDist,index, viterbi_alignments);
        }
    }

    template<typename Derived>
    void print_matrix_msg(Derived *mat, int size,std::string name){
        std::cout<<name<<"\n";
        for (int i = 0; i< size; i++){
            std:: cout << mat[i] << " ";
        }
        std:: cout<< "\n";
    }
    
    template<typename Derived>
    void expand_hypothesis_with_fsa(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments) {
        
        ///timer.clear();
        
        ///timer.start("expand_with_fsa");
        
        // viterbi_alignments check
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
        
        if(true){ // gpu expand
            if(index==0) {
                cols = 1;
            } else {
                cols = 0;
                for (int i = 0; i <beam_size; i++){
                    int symbol = this->current_indices(i);
                    if (symbol == this->invalid_symbol){
                        break;
                    }
                    cols += 1;
                }
            }
            std::cout<< "cols " << cols << "\n";
            
            
            this->expand_pq_global_gpu(pqg,pqg_set, viterbi_alignments, cols);
            
            this->invalid_number = 0;
            empty_queue_global();
            
            while(!pqg.empty()) {
                dec_global_obj<dType> temp = pqg.top();
                pqg.pop();
                temp.val = -temp.val;
                
                if (merge_state){
                    if (pq_global_set.count(temp) == 0){
                        pq_global.push(temp);
                        pq_global_set.insert(temp);
                    }
                }
                else {
                    pq_global.push( temp );
                }
            }
        }
        
                // filter the pq_global
        // so that dec_global_obj in pq_global has unique (history, vocab_index, state.name)
        // this is necessary: if two dec_global_obj have the same (history, vocab_index, state.name)
        // then they will have the same future, but different history. If we don't merge this, the whole beam will be occuped by
        // dec_global_obj with same (history, vocab_index, state.name).
        
        ///timer.start("filter");
        
        std::cout<<"before: "<<pq_global.size()<<"\n";
        
        int hash_index = 0;
        std::unordered_map<std::string,int> history_to_hash;
        std::unordered_map<int,int> beam_index_to_hash;
        for(int i=0; i<cols; i++) {
            std::string history = "";
            for (int j = 0; j<= current_index; j++){
                history += std::to_string(top_sentences(i,j)) + " ";
            }
            if (history_to_hash.count(history) == 0){
                history_to_hash[history] = hash_index;
                //std::cout <<"history: " <<i << " "<< history << " " <<hash_index << "\n";
                hash_index += 1;
            }
            beam_index_to_hash[i] = history_to_hash[history];
        }
        
        std::unordered_map<std::string,dec_global_obj<dType>> filtered_queue;
        while (!pq_global.empty()){
            dec_global_obj<dType> dgobj = pq_global.top();
            pq_global.pop();
            std::string key = "";
            key += std::to_string(beam_index_to_hash[dgobj.beam_index])+" ";
            key += std::to_string(dgobj.vocab_index) + " ";
            key += dgobj.s->name;
            //BZ_CUDA::logger<<"key "<<key<<"\n";
            if (filtered_queue.count(key) == 0){
                filtered_queue[key] = dgobj;
            } else {
                dec_global_obj<dType> old_dgobj = filtered_queue[key];
                //std::cout << "[-----]" << key << " "<< dgobj.val << " " << old_dgobj.val <<"\n";
                if (dgobj.val > old_dgobj.val){
                    filtered_queue[key] = dgobj;
                }
            }
        }
        
        for (auto const & item: filtered_queue){
            pq_global.push(item.second);
        }
        
        std::cout<<"after: "<<pq_global.size()<<"\n";
        
        ///timer.end("filter");
        
        //Now have global heap with (beam size*beam size) elements
        //Go through until (beam size) new hypotheses.
        
        if (print_beam){
            BZ_CUDA::logger<<"---------"<<index<<"------------\n";
        }
        
        if (pq_global.size() == 0){
            this->invalid_number = this->beam_size;
        }
        
        
        ///timer.start("for_loop_2");
        
        int i = 0;
        while(i < beam_size) {
            if (pq_global.size() == 0)
            {
                //grow the sentences in the beam;
                top_sentences_temp.row(i) = top_sentences.row(0);
                top_sentences_temp(i,current_index+1) = start_symbol;
                
                // vertibi
                top_sentences_temp_viterbi.row(i) = top_sentences_viterbi.row(0);
                //top_sentences_temp_viterbi(i,current_index+1);
                
                // for current indices
                current_indices(i) = invalid_symbol;
                new_indicies_changes(i) = 0;
                
                // sentence score
                top_sentences_scores_temp(i) = 0;
                // current state
                current_states[i] = this->fsa_model->end_state;

                i++;
            }
            else {
                dec_global_obj<dType> temp = pq_global.top();
                pq_global.pop();
                //BZ_CUDA::logger <<"Value: " << temp.val << " Index: " << temp.vocab_index << " Beam: "<< temp.beam_index << "\n\n";
                if(temp.vocab_index!=start_symbol) {
                    if(temp.vocab_index==end_symbol) {
                        
                        if (print_beam){
                            BZ_CUDA::logger<<"[*]Cell:"<<i<<"\tF-Cell:"<<temp.beam_index<<"\tState:"<<temp.s->name<<"\tWord:"<<this->fsa_model->index2words[temp.vocab_index]<<"["<<temp.vocab_index<<"]"<<"\tScore:"<<temp.val<<"\n";
                        }
                        
                        hypotheses.push_back(eigen_mat_wrapper<dType>(current_index+2));
                        hypotheses.back().hypothesis = top_sentences.block(temp.beam_index,0,1,current_index+2).transpose();//.row(temp.beam_index);
                        hypotheses.back().hypothesis(current_index+1) = end_symbol;
                        
                        //viterbi
                        hypotheses.back().viterbi_alignments = top_sentences_viterbi.block(temp.beam_index,0,1,current_index+2).transpose();//.row(temp.beam_index);
                        hypotheses.back().viterbi_alignments(current_index+1) = temp.viterbi_alignment;

                        
                        hypotheses.back().score = temp.score + penalty;
                        
                        if (end_transfer){
                            hypotheses.back().score = temp.score - outputDist(end_symbol,temp.beam_index);
                        }
                        
                        if (end_transfer){
                            // [HERE]
                            model->get_chts(hypotheses.back().chts, temp.beam_index, beam_size);
                        }
                    }
                    else {
                        if (print_beam){
                            BZ_CUDA::logger<<"Cell:"<<i<<"\tF-Cell:"<<temp.beam_index<<"\tState:"<<temp.s->name<<"\tWord:"<<this->fsa_model->index2words[temp.vocab_index]<<"["<<temp.vocab_index<<"]"<<"\tScore:"<<temp.val<<" "<<temp.score<<"\n";
                        }
                        top_sentences_temp.row(i) = top_sentences.row(temp.beam_index);
                        top_sentences_temp(i,current_index+1) = temp.vocab_index;
                        
                        // vertibi
                        top_sentences_temp_viterbi.row(i) = top_sentences_viterbi.row(temp.beam_index);
                        top_sentences_temp_viterbi(i,current_index+1) = temp.viterbi_alignment;
                        
                        // repeat penalty
                        if (penalize_repeat){
                            temp_sentence_sets[i] = sentence_sets[temp.beam_index];
                            if (temp_sentence_sets[i].count(temp.vocab_index) == 0){
                                temp_sentence_sets[i][temp.vocab_index] = 0;
                            }
                            temp_sentence_sets[i][temp.vocab_index] += 1;
                        }
                        
                        current_indices(i) = temp.vocab_index;
                        new_indicies_changes(i) = temp.beam_index;
                        top_sentences_scores_temp(i) = temp.score + penalty;
                        current_states[i] = temp.s;
                        
                        i++;
                        
                        
                    }
                }
            }
        }
        
        ///timer.end("for_loop_2");
        
        top_sentences = top_sentences_temp;
        top_sentences_scores = top_sentences_scores_temp;
        
        if (penalize_repeat){
            sentence_sets = temp_sentence_sets;
        }
        
        current_index += 1;
        
        for(int i=0; i<beam_size; i++) {
            h_current_indices[i] = current_indices(i);
        }
        
        ///timer.end("expand_with_fsa");
        ///timer.report();
        
        /*
         total_end= std::chrono::system_clock::now();
         total_dur = total_end - total_start;
         BZ_CUDA::logger<<"Epq: "<<epq_dur.count()<<"s \n";
         BZ_CUDA::logger<<"Expand: "<<expand_dur.count()<<"s \n";
         BZ_CUDA::logger<<"total: "<<total_dur.count()<<"s \n";
         */
        
        //cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice);
    }

    //template<typename Derived>
    void expand_pq_global_gpu(
        std::priority_queue<dec_global_obj<dType>,std::vector<dec_global_obj<dType>>, pq_global_compare_functor> & pqg, std::unordered_set<dec_global_obj<dType>>& pqg_set, std::vector<int> &viterbi_alignments, int cols){
        
        empty_queue_global(pqg,pqg_set);
        
        if (cols <=0){
            return;
        }
        
        ///timer.start("next_word_loop");
        // calculate next_word_indicies;
        for(int i=0; i<cols; i++) {
            this->current_states[i]->next_word_indicies();
        }
        ///timer.end("next_word_loop");
        
        //cudaProfilerStart();
        ///timer.start("gpusort_loop");
        
        // transfer d_dict;
        int total_valid_size = 0;
        for(int i=0; i<cols; i++) {
            state * istate = this->current_states[i];
            int valid_vocab_size = istate->next_word_index_set->size();
            int * d_dict_local = d_dict + total_valid_size;
            h_valid_vocab_sizes[i] = total_valid_size;
            
            // dict2array
            CUDA_ERROR_WRAPPER(cudaMemcpyAsync(d_dict_local, istate->h_dict,
                                               valid_vocab_size*sizeof(int),
                                               cudaMemcpyHostToDevice, streams[i]),
                               "expand_pq 1 h_dict to d_dict\n");
            total_valid_size += valid_vocab_size;
        }
        h_valid_vocab_sizes[cols] = total_valid_size;
        std::cout << "total_valid_size "<<total_valid_size << "\n";
        
        // sync
        for(int i=0; i<cols; i++) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // prepare d_current_indices
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice),"expand_pq 1 h_current_indices to d_current_indices");
        
        // prepare d_outputdist;
        ///timer.start("h_outputdist_to_gpu");
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_outputdist, h_outputdist,
                                      vocab_size*beam_size*sizeof(dType),
                                      cudaMemcpyHostToDevice),
                           "expand_pq 1 h_outputdist to d_outputdist\n");
        ///timer.end("h_outputdist_to_gpu");

        // prepare top_sentence_scores;
        for (int i = 0; i < cols; i ++){
            h_sentence_scores[i] = top_sentences_scores(i);
        }
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_sentence_scores, h_sentence_scores,
                                      cols*sizeof(dType),
                                      cudaMemcpyHostToDevice),
                           "expand_pq 1 h_sentence_scores to d_sentence_scores\n");

        
        // add_features;
        add_features<<<cols, 256>>>(d_outputdist, d_sentence_scores, d_encourage, d_current_indices, adjacent_repeat_penalty, d_wordlen, wordlen_weight, d_vocab_bin, alliteration_weight, beam_size, vocab_size);

        
        //prepare sentence_sets
        int sentence_set_size = 0;
        for (int i = 0; i < cols; i ++ ){
            for (const auto & item : sentence_sets[i]){
                int vocab_index = item.first;
                int occur_times = item.second;
                h_sentence_set[sentence_set_size * 3] = vocab_index;
                h_sentence_set[sentence_set_size * 3+1] = i;
                h_sentence_set[sentence_set_size * 3+2] = occur_times;
                //std::cout << "vocab beam_index occur_time : " <<  vocab_index << " " << i <<" " << occur_times << "\n";
                sentence_set_size += 1;
            }
        }
        std::cout<<"sentence_set_size "<< sentence_set_size << "\n";
        
        if (sentence_set_size > 0){
            CUDA_ERROR_WRAPPER(cudaMemcpy(d_sentence_set, h_sentence_set,
                                          sentence_set_size*3*sizeof(int),
                                          cudaMemcpyHostToDevice),
                               "expand_pq 1 h_sentence_set to d_sentence_set\n");
            
            // add_feature_repeat;
            //std::cout << "repeat_penalty: " <<repeat_penalty << " " << interactive_repeat_penalty << "\n";
            
            add_feature_repeat<<<std::min(256,(sentence_set_size + 256 - 1)/256),256>>>(d_outputdist, d_sentence_set, repeat_penalty + interactive_repeat_penalty, sentence_set_size, vocab_size);
        }
        
        
        // prepare valid_vocab_sizes;
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_valid_vocab_sizes, h_valid_vocab_sizes,
                                      (beam_size+1)*sizeof(int),
                                      cudaMemcpyHostToDevice),
                           "expand_pq 1 h_valid_vocab_sizes to d_valid_vocab_sizes\n");
        
        // log kernel;
        top_k<<<cols,256>>>(d_outputdist, d_outputdist_topk, d_pointers, d_dict, d_beams, d_valid_vocab_sizes, vocab_size);
        
        thrust::sort_by_key(thrust::cuda::par, d_outputdist_topk, d_outputdist_topk + total_valid_size, d_pointers, thrust::greater<dType>());
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(h_outputdist_topk, d_outputdist_topk,
                                      total_valid_size*sizeof(dType),
                                      cudaMemcpyDeviceToHost),
                           "expand_pq 1 d_outputdist_topk to h_outputdist_topk\n");
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(h_pointers, d_pointers,
                                      total_valid_size*sizeof(int),
                                      cudaMemcpyDeviceToHost),
                           "expand_pq 1 d_pointers to h_pointers\n");
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(h_dict, d_dict,
                                      total_valid_size*sizeof(int),
                                      cudaMemcpyDeviceToHost),
                           "expand_pq 1 d_dict to h_dict\n");
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(h_beams, d_beams,
                                      total_valid_size*sizeof(int),
                                      cudaMemcpyDeviceToHost),
                           "expand_pq 1 d_beams to h_beams\n");
        
        ///timer.end("gpusort_loop");

        //cudaProfilerStop();
        
        
        ///timer.start("for_loop_1_new");
        if (false && total_valid_size < 10000){
            print_matrix_msg(h_valid_vocab_sizes, beam_size + 1, "h_valid_vocab_sizes");
            print_matrix_msg(h_beams, total_valid_size, "h_beams");
            print_matrix_msg(h_pointers, total_valid_size, "h_pointers");
            print_matrix_msg(h_outputdist_topk, total_valid_size, "h_outputdist_topk");
        }
        
        int pq_size_limit = beam_size * cols;
        int nprune = 0;
        for (int ipointer = 0; ipointer < total_valid_size; ipointer ++){
            
            dType base_score = h_outputdist_topk[ipointer];
            // base_score already includes the top_sentences_scores;
            if (fsa_can_prune){
                if (pqg.size() >= pq_size_limit){
                    dType upper_bound = base_score;
                    if ( - upper_bound > pqg.top().val){
                        nprune += 1;
                        break;
                    }
                }
            }
            
            int p = h_pointers[ipointer];
            int beam_index = h_beams[p];
            int vocab_index = h_dict[p];
            int viterbi_alignment = viterbi_alignments[beam_index];
            
            state * istate = this->current_states[beam_index];
            std::vector<sw> sws;
            this->fsa_model->next_states(istate,vocab_index,sws);
            
            for (auto const & s:sws){
                dType score = base_score;
                dType fsa_score = 0.0;
                fsa_score = this->fsa_weight * s.weight;
                score += fsa_score;
                
                //if (total_valid_size < 10000){
                //std::cout<< p << " " << beam_index << " " << vocab_index << " " << viterbi_alignment << " " << score << "\n";
                //}
                
                dec_global_obj<dType> dgobj =  dec_global_obj<dType>(-score,beam_index, vocab_index, viterbi_alignment);
                dgobj.s = s.s;
                dgobj.score = score;
                
                if(pqg.size() >= pq_size_limit ) {
                    pqg.pop();
                }
                
                if (merge_state){
                    if (pqg_set.count(dgobj) == 0){
                        pqg.push(dgobj);
                        pqg_set.insert(dgobj);
                    }
                }
                else {
                    pqg.push( dgobj );
                }
            }
        }
        //std::cout<< "nprune / pq_limit " << nprune << "/" << pq_size_limit << "\n";
        
        ///timer.end("for_loop_1_new");
        
    }
    

    
	template<typename Derived>
	void expand_hypothesis_without_fsa(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments) {
		
        int n_rows = outputDist.rows();
        if (this->target_vocab_policy == 3){
            n_rows = nnz;
            //std::cout<<"nnz: "<<nnz<<"\n";
            //std::cout<<"h_rowIdx inside: "<< h_rowIdx << "\n";
            //print_matrix(h_rowIdx, 1, nnz);
        }
    
        
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
            
            int symbol = this->current_indices(i);
            if (symbol == this->invalid_symbol){
                break;
            }
            
			for(int j=0; j<n_rows; j++) {
                int _word_index = j;
                if (this->target_vocab_policy == 3){ // LSH_WTA
                    _word_index = h_rowIdx[j];
                }

				if(pq.size() < beam_size + 1) {
					pq.push( dec_obj<dType>(-outputDist(j,i),_word_index,viterbi_alignments[i]) );
				}
				else {
					if(-outputDist(j,i) < pq.top().val) {
                        pq.pop();
						pq.push( dec_obj<dType>(-outputDist(j,i),_word_index,viterbi_alignments[i]) );
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

        if (print_beam){
            BZ_CUDA::logger<<"---------"<<index<<"------------\n";
        }
        
		//Now have global heap with (beam size*beam size) elements
		//Go through until (beam size) new hypotheses.
		int i = 0;
		while(i < beam_size) {
            
            if (pq_global.size() == 0){
                //grow the sentences in the beam;
                top_sentences_temp.row(i) = top_sentences.row(0);
                top_sentences_temp(i,current_index+1) = start_symbol;

                // vertibi
                top_sentences_temp_viterbi.row(i) = top_sentences_viterbi.row(0);
                //top_sentences_temp_viterbi(i,current_index+1) = temp.viterbi_alignment;

                // for current indices
                current_indices(i) = invalid_symbol;
                new_indicies_changes(i) = 0;
                
                // sentence score
                top_sentences_scores_temp(i) = 0;
                
                i ++;
                continue;

            }
            
			dec_global_obj<dType> temp = pq_global.top();
			pq_global.pop();

            if(temp.vocab_index!=start_symbol) {
				if(temp.vocab_index==end_symbol) {
                    
                    if (print_beam){
                        BZ_CUDA::logger<<"[*]Cell:"<<i<<" F-Cell:"<<temp.beam_index<<" Vocab:"<<temp.vocab_index<<" Score:"<<temp.val<<"\n";
                    }
                    
					hypotheses.push_back(eigen_mat_wrapper<dType>(current_index+2));
					hypotheses.back().hypothesis = top_sentences.block(temp.beam_index,0,1,current_index+2).transpose();//.row(temp.beam_index);
					hypotheses.back().hypothesis(current_index+1) = end_symbol;

					hypotheses.back().viterbi_alignments = top_sentences_viterbi.block(temp.beam_index,0,1,current_index+2).transpose();//.row(temp.beam_index);
					hypotheses.back().viterbi_alignments(current_index+1) = temp.viterbi_alignment;
					//hypotheses.back().score = std::log(temp.val) /*+ top_sentences_scores(temp.beam_index)*/ + penalty;
					hypotheses.back().score = temp.val + penalty;
				}
				else {
                    
                    if (print_beam){
                        BZ_CUDA::logger<<"Cell:"<<i<<" F-Cell:"<<temp.beam_index<<" Vocab:"<<temp.vocab_index<<" Score:"<<temp.val<<"\n";
                    }
                    // grow the sentences in the beam
					top_sentences_temp.row(i) = top_sentences.row(temp.beam_index);
					top_sentences_temp(i,current_index+1) = temp.vocab_index;

                    // vertibi
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

		// BZ_CUDA::logger << "--------------- top sentences viterbi ---------------\n";
		// for(int i=0; i<beam_size; i++) {
		// 	BZ_CUDA::logger << top_sentences_viterbi.row(i) << "\n\n\n\n\n\n";
		// 	BZ_CUDA::logger << top_sentences.row(i) << "\n\n\n\n\n\n";
		// }

		for(int i=0; i<beam_size; i++) {
			h_current_indices[i] = current_indices(i);
		}
		//cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice);
	}
	
	void init_decoder(int source_length = 0, bool init_h_start_symbol = true) {

        this->source_length = source_length;

		current_index = 0;
		top_sentences_scores.setZero();
		hypotheses.clear();

		for(int i=0; i<beam_size; i++) {
			current_indices(i) = start_symbol;
            if (init_h_start_symbol){
                h_current_indices[i] = start_symbol;
            }
		}

		for(int i=0; i<beam_size; i++) {
			for(int j=0; j<max_decoding_length; j++) {
				top_sentences(i,j) = start_symbol;
				top_sentences_viterbi(i,j) = -20;
			}
		}

        
        if (penalize_repeat){
            for (int i = 0;i < sentence_sets.size(); i++){
                sentence_sets[i].clear();
            }
        }
        
        if (this->with_fsa){
            current_states.clear();
            for (int i=0; i<beam_size;i++){
                current_states.push_back(this->fsa_model->start_state);
            }
        }
        
        
		//cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice);
	}

	void print_current_hypotheses() {

		// BZ_CUDA::logger << "Printing out current indicies"<<std::endl;
		// for(int i=0; i< current_indices.size(); i++) {
		// 	BZ_CUDA::logger << current_indices[i] << " ";
		// }
		// BZ_CUDA::logger << "\n\n";

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

		// BZ_CUDA::logger << "Printing out beam changes\n";
		// BZ_CUDA::logger << new_indicies_changes << "\n\n";
	}


	void output_k_best_hypotheses(int source_length, int *h_new_vocab_index = NULL, bool target_vocab_shrink = false) {

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
        
        bool is_first = true;
        
		while(!best_hypoth_temp.empty()) {

			if(print_score) {
				output << "-Score: " <<hypotheses[best_hypoth_temp.top().index].score << "\n";
			}
			for(int j=0; j<hypotheses[best_hypoth_temp.top().index].hypothesis.size(); j++) {
                int vocab_index =hypotheses[best_hypoth_temp.top().index].hypothesis(j);
                if (target_vocab_shrink){
                    int original_vocab_index = h_new_vocab_index[vocab_index];
                    output << original_vocab_index << " ";
                } else {
                    output << vocab_index << " ";
                }
			}
            
            if (is_first)
            {
                if (end_transfer){
                    model->set_chts(hypotheses[best_hypoth_temp.top().index].chts, beam_size);
                    // set h_current_indices
                    Eigen::Matrix<int, Eigen::Dynamic,1> &word_indices = hypotheses[best_hypoth_temp.top().index].hypothesis;
                    int last_symbol = word_indices(word_indices.rows()-2);
                    for(int i=0; i<beam_size; i++) {
                        h_current_indices[i] = last_symbol;
                    }
                }
                is_first = false;
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
