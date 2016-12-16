#ifndef DECODER_H
#define DECODER_H

#include <queue>
#include <vector>
#include <fstream>
#include <utility> 
#include <float.h>
#include "fsa.hpp"
#include "format.h"


//MARK:FSA related obj

template<typename dType>
class neuralMT_model;

struct state_obj{
    state s;
    float score;
    
    std::vector<boost::dynamic_bitset<>> masks;
    
    state_obj(){}
    
    state_obj(state _s, float _score){
        s = _s;
        score = _score;
    }
};




template<typename dType>
struct fsa_obj{
    dType val;
    int beam_index;
    int vocab_index;
    std::vector<state_obj> state_objs;
    
    std::string state_objs_str(bool source_mask = false, int source_length = 0, bool is_end = false){
        std::string s = "[";
        for (int i = 0; i< std::min(int(state_objs.size()),5); i += 1){
            state_obj so = state_objs[i];
            if (source_mask){
                std::string bit_string;
                boost::to_string(so.masks[0], bit_string);
                
                if (is_end){
                    // if is end, gnenerate the longest one and rest 4;
                    int longest = 0;
                    boost::dynamic_bitset<> long_bits;
                    for (int j = 0; j<so.masks.size(); j++){
                        if (longest < so.masks[j].count()){
                            longest = so.masks[j].count();
                            long_bits = so.masks[j];
                        }
                    }
                    
                    boost::to_string(long_bits, bit_string);
                    
                    bit_string += ", ";
                    std::string bs = "";
                    
                    for (int j = 1; j< std::min(5,int(so.masks.size())); j++){
                        boost::to_string(so.masks[j], bs);
                        bit_string += bs + ", ";
                    }
                }
                /*
                 std::string count_str = "";
                 for (auto bits: so.masks){
                 count_str += fmt::format("{}",bits.count());
                 }
                 */
                s += fmt::format("{}({}/{}) [{}] , ",so.s.name,so.masks.size(),source_length, bit_string);
            } else {
                s += so.s.name + ", ";
            }
            
        }
        if (state_objs.size() > 5){
            s += fmt::format("...{}...",state_objs.size());
        }
        s += "]";
        return s;
    }
    
};


//MARK: decoding related obj

//The decoder object type
template<typename dType>
struct dec_global_obj {

	dType val;
    dType score;
	int beam_index;
	int vocab_index;
    state s;
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
                && s.name == other.s.name );
	}
};

template<typename dType>
struct dec_obj {

	dType val;
    dType score;
	int vocab_index;
    state s;
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
                && s.name == other.s.name );
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
            ^ (hash<string>()(k.s.name) << 3) ;
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
            (hash<string>()(k.s.name) << 3) ;
            
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

struct fsa_compare_functor{
    template<typename dType>
    bool operator() (fsa_obj<dType> &a, fsa_obj<dType> &b) const { return (a.val < b.val); }
};

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
    bool fsa_log = false;
    bool merge_state = true;
    int invalid_number = 0;
    bool end_transfer = false; // true if in fsa_line mode

    // other weight
    float alliteration_weight = 0.0;
    float wordlen_weight = 0.0;
    
    // for encourage list
    bool encourage = false;
    std::unordered_map<int,float> *encourage_list = NULL;
    //float encourage_weight = 0;
    
    std::unordered_map<std::string,int> tgt_mapping;
    
    std::vector<state> current_states;
    
    std::vector<fsa_obj<dType>> current_fsa_objs;
    
    
	std::priority_queue<dec_obj<dType>,std::vector<dec_obj<dType>>, pq_compare_functor> pq;
    
    //std::vector<std::priority_queue<dec_obj<dType>,std::vector<dec_obj<dType>>, pq_compare_functor>> pqs;
    
	std::priority_queue<dec_global_obj<dType>,std::vector<dec_global_obj<dType>>, pq_global_compare_functor> pq_global;

    std::unordered_set<dec_obj<dType>> pq_set;
    //std::vector<std::unordered_set<dec_obj<dType>>> pq_sets;
    
    
    std::unordered_set<dec_global_obj<dType>> pq_global_set;
    
    std::priority_queue<fsa_obj<dType>, std::vector<fsa_obj<dType>>, fsa_compare_functor> fsa_pq;
    std::priority_queue<fsa_obj<dType>, std::vector<fsa_obj<dType>>, fsa_compare_functor> fsa_global_pq;
    
    
    
    
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
    bool penalize_repeat = false;
    precision repeat_penalty = 0.0;
    float interactive_repeat_penalty;
    bool penalize_adjacent_repeat = false;
    precision adjacent_repeat_penalty = 0.0;
    std::vector<std::unordered_map<int,int>> sentence_sets;
    std::vector<std::unordered_map<int,int>> temp_sentence_sets;

    //for source_length
    int source_length = 0;
    
	//GPU
	int *h_current_indices;
	int *d_current_indices;

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
		//cudaMalloc((void**)&d_current_indices,beam_size*1*sizeof(int));//put void**

        // repeat_penalty
        if (repeat_penalty != 0.0){
            penalize_repeat = true;
            for (int i = 0; i<beam_size; i++){
                std::unordered_map<int,int>* sentence_set = new std::unordered_map<int,int>();
                sentence_sets.push_back(*sentence_set);
                temp_sentence_sets.push_back(*sentence_set);
            }
        }
        // adjacent_repeat_penalty
        if (adjacent_repeat_penalty != 0.0){
            penalize_adjacent_repeat = true;
        }
        
        // fsa
        this->fsa_model = NULL;
    }
    
	~decoder() {
		output.close();
        //delete this->pool;
		//cudaFree(d_current_indices);
    }
    // for encourage list
    void init_encourage_list(std::string fn, float weight){
        
        
        // should call after init_fsa();
        if (fn == ""){
            encourage = false;
            //encourage_weight = 0;
            if (this->encourage_list != NULL){
                delete this->encourage_list;
                this->encourage_list = NULL;
            }
            return;
        }

        encourage = true;
        //encourage_weight = weight;
        if (this->encourage_list != NULL){
            delete this->encourage_list;
        }
        this->encourage_list = new std::unordered_map<int,float>();
        
        std::ifstream fin(fn.c_str());
        std::string line;
        while(std::getline(fin,line)){
            int index = 2 ; // <UNK>
            if (this->tgt_mapping.count(line) > 0){
                index = this->tgt_mapping[line];
            }
            if (index != 2){
                (*(this->encourage_list))[index] = weight;
            }
        }
        fin.close();
        
        BZ_CUDA::logger<< "Encourage Weight: " << weight <<"\n";
        BZ_CUDA::logger<< "Encourage List Size: " <<(int)(encourage_list->size()) <<"\n";
    }
    
    void init_encourage_list_additional(std::string fn, float weight){
        // if there's more than one encourage list, use this function to init the encourage_lists except the first one.
        std::ifstream fin(fn.c_str());
        std::string line;
        while(std::getline(fin,line)){
            int index = 2 ; // <UNK>
            if (this->tgt_mapping.count(line) > 0){
                index = this->tgt_mapping[line];
            }
            if (index != 2){
                if (this->encourage_list->count(index) == 0){
                    (*(this->encourage_list))[index] = weight;
                } else {
                    (*(this->encourage_list))[index] += weight;
                }
            }
            
        }
        fin.close();
        BZ_CUDA::logger<< "Encourage Weight: "<< weight <<"\n";
        BZ_CUDA::logger<< "Encourage List Size: "<<(int)(encourage_list->size())<<"\n";
        
    }
    
    
    // for single fsa file
    void init_fsa_inner(global_params &params){
        this->fsa_weight = params.fsa_weight;
        this->fsa_log = params.fsa_log;
        this->with_fsa = true;
        if (this->fsa_weight >=0){
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
                
                state istate = this->current_states[i];
                std::vector<sw> sws;
                this->fsa_model->next_states(istate,end_symbol,sws);
                for (auto const & s: sws){
                    if ( s.s.name == this->fsa_model->end_state->name){
                        
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
    void expand_hypothesis(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments) {
        if (this->with_fsa){
            expand_hypothesis_with_fsa(outputDist,index, viterbi_alignments);
        } else {
            expand_hypothesis_without_fsa(outputDist,index, viterbi_alignments);
        }
    }

    template<typename Derived>
    void expand_hypothesis_with_fsa(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments) {
        
        
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
        if(index==0) {
            cols = 1;
        }
        
        this->invalid_number = 0;
        empty_queue_global();
        
        for(int i=0; i<cols; i++) {
            
            int symbol = this->current_indices(i);
            if (symbol == this->invalid_symbol){
                break;
            }
            
            this->expand_pq(pq,pq_set,outputDist,i,viterbi_alignments);
            
            //Now have the top elements
            while(!pq.empty()) {
                dec_obj<dType> temp = pq.top();
                pq.pop();
                dec_global_obj<dType> dgobj =  dec_global_obj<dType>(-temp.val + top_sentences_scores(i),i,temp.vocab_index, temp.viterbi_alignment);
                dgobj.s = temp.s;
                dgobj.score = temp.score + top_sentences_scores(i);
                
                if (merge_state){
                    if (pq_global_set.count(dgobj) == 0){
                        pq_global.push(dgobj);
                        pq_global_set.insert(dgobj);
                    }
                }
                else {
                    pq_global.push( dgobj );
                }
            }
        }
        
        
        // filter the pq_global
        // so that dec_global_obj in pq_global has unique (history, vocab_index, state.name)
        
        int hash_index = 0;
        std::unordered_map<std::string,int> history_to_hash;
        std::unordered_map<int,int> beam_index_to_hash;
        for(int i=0; i<cols; i++) {
            std::string history = "";
            for (int j = 0; j< current_index; j++){
                history += std::to_string(top_sentences(i,j)) + " ";
            }
            if (history_to_hash.count(history) == 0){
                history_to_hash[history] = hash_index;
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
            key += dgobj.s.name;
            //BZ_CUDA::logger<<"key "<<key<<"\n";
            if (filtered_queue.count(key) == 0){
                filtered_queue[key] = dgobj;
            } else {
                dec_global_obj<dType> old_dgobj = filtered_queue[key];
                //BZ_CUDA::logger << "[-----]" << key << " "<< dgobj.val << " " << old_dgobj.val <<"\n";
                if (dgobj.val > old_dgobj.val){
                    //BZ_CUDA::logger << "[-----]" << key << " "<< dgobj.val << " " << old_dgobj.val <<"\n";
                    filtered_queue[key] = dgobj;
                }
            }
        }
        
        for (auto const & item: filtered_queue){
            pq_global.push(item.second);
        }

        //Now have global heap with (beam size*beam size) elements
        //Go through until (beam size) new hypotheses.
        
        if (print_beam){
            BZ_CUDA::logger<<"---------"<<index<<"------------\n";
        }
        
        if (pq_global.size() == 0){
            this->invalid_number = this->beam_size;
        }
        
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
                current_states[i] = *(this->fsa_model->end_state);

                i++;
            }
            else {
                dec_global_obj<dType> temp = pq_global.top();
                pq_global.pop();
                //BZ_CUDA::logger <<"Value: " << temp.val << " Index: " << temp.vocab_index << " Beam: "<< temp.beam_index << "\n\n";
                if(temp.vocab_index!=start_symbol) {
                    if(temp.vocab_index==end_symbol) {
                        
                        if (print_beam){
                            BZ_CUDA::logger<<"[*]Cell:"<<i<<"\tF-Cell:"<<temp.beam_index<<"\tState:"<<temp.s.name<<"\tWord:"<<this->fsa_model->index2words[temp.vocab_index]<<"["<<temp.vocab_index<<"]"<<"\tScore:"<<temp.val<<"\n";
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
                            BZ_CUDA::logger<<"Cell:"<<i<<"\tF-Cell:"<<temp.beam_index<<"\tState:"<<temp.s.name<<"\tWord:"<<this->fsa_model->index2words[temp.vocab_index]<<"["<<temp.vocab_index<<"]"<<"\tScore:"<<temp.val<<" "<<temp.score<<"\n";
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
        
        top_sentences = top_sentences_temp;
        top_sentences_scores = top_sentences_scores_temp;
        
        if (penalize_repeat){
            sentence_sets = temp_sentence_sets;
        }
        
        current_index += 1;
        
        for(int i=0; i<beam_size; i++) {
            h_current_indices[i] = current_indices(i);
        }
        
        /*
         total_end= std::chrono::system_clock::now();
         total_dur = total_end - total_start;
         BZ_CUDA::logger<<"Epq: "<<epq_dur.count()<<"s \n";
         BZ_CUDA::logger<<"Expand: "<<expand_dur.count()<<"s \n";
         BZ_CUDA::logger<<"total: "<<total_dur.count()<<"s \n";
         */
        
        //cudaMemcpy(d_current_indices,h_current_indices,beam_size*1*sizeof(int),cudaMemcpyHostToDevice);
    }

    template<typename Derived>
    void expand_pq(std::priority_queue<dec_obj<dType>,std::vector<dec_obj<dType>>, pq_compare_functor>& pq,     std::unordered_set<dec_obj<dType>>& pq_set,
                   const Eigen::MatrixBase<Derived> &outputDist, int beam_index, std::vector<int> &viterbi_alignments){
        
        empty_queue_pq(pq,pq_set);
        
        const state& istate = this->current_states[beam_index];
        
        // there's no encournage_partially option now;
        
        int nprune = 0;
        int n = 0;
        int nrows = outputDist.rows();
        std::unordered_set<int> next_indicies;
        istate.next_word_indicies(next_indicies);

        for (auto const & j : next_indicies){
            if (j == -1 || j>=outputDist.rows()) {continue;}
            
            dType base_score = std::log(outputDist(j,beam_index));
            
            if (encourage){
                if (this->encourage_list->count(j) > 0){
                    base_score += (*(this->encourage_list))[j];
                }
            }
            
            if (penalize_repeat){
                if ((sentence_sets[beam_index]).count(j) > 0){
                    //BZ_CUDA::logger<<"Beam: "<<i<<" Repeat: "<<j<<" "<<base_score;
                    base_score += sentence_sets[beam_index][j] * (repeat_penalty + interactive_repeat_penalty);
                    //BZ_CUDA::logger<<" "<<base_score<<"\n";
                }
            }
            
            if (penalize_adjacent_repeat){
                if (this->current_indices(beam_index) == j){
                    base_score += adjacent_repeat_penalty;
                }
            }
            
            
            std::string word_current = fsa_model->index2words[this->current_indices(beam_index)];
            std::string word_next = fsa_model->index2words[j];
            
            // alliteration_weight;
            if (word_current[0] == word_next[0]){
                base_score += alliteration_weight;
            }
            
            // wordlen_weight;
            base_score += wordlen_weight * word_next.size() * word_next.size();
            
            n += 1;
            if (pq.size() >= beam_size){
                if (fsa_can_prune){
                    dType upper_bound = base_score;
                    if ( - upper_bound > pq.top().val){
                        nprune += 1;
                        continue;
                    }
                }
            }
            
            /*
             if (istate.name=="0" && j == 1){
             BZ_CUDA::logger<<"Ready\n";
             int k = j;
             BZ_CUDA::logger<<"Here is"<<k<<"\n";
             }
             */
            
            std::vector<sw> sws;
            this->fsa_model->next_states(istate,j,sws);
            
            for (auto const & s:sws){
                dType score = base_score;
                dType fsa_score = 0.0;
                fsa_score = this->fsa_weight * s.weight;
                score += fsa_score;
                
                
                
                if(pq.size() < beam_size ) {
                    dec_obj<dType> dobj = dec_obj<dType>(-score,j, viterbi_alignments[beam_index] );
                    dobj.score = score;
                    dobj.s = s.s;
                    
                    if (merge_state){
                        if (pq_set.count(dobj) == 0){
                            pq.push(dobj);
                            pq_set.insert(dobj);
                        }
                    }
                    else {
                        pq.push( dobj );
                    }
                }
                else {
                    if(-score < pq.top().val) {
                        pq.pop();
                        dec_obj<dType> dobj = dec_obj<dType>(-score,j, viterbi_alignments[beam_index] );
                        dobj.score = score;
                        dobj.s = s.s;
                        
                        if (merge_state){
                            if (pq_set.count(dobj) == 0){
                                pq.push(dobj);
                                pq_set.insert(dobj);
                            }
                        }
                        else {
                            pq.push( dobj );
                        }
                    }
                }
            }
        }
        
        //BZ_CUDA::logger<<beam_index<<" "<< nprune << "/" << n << "\n";
        
    }

    
	template<typename Derived>
	void expand_hypothesis_without_fsa(const Eigen::MatrixBase<Derived> &outputDist,int index,std::vector<int> &viterbi_alignments) {
		

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
                current_states.push_back(*(this->fsa_model->start_state));
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
        
        bool is_first = true;
        
		while(!best_hypoth_temp.empty()) {

			if(print_score) {
				output << "-Score: " <<hypotheses[best_hypoth_temp.top().index].score << "\n";
			}
			for(int j=0; j<hypotheses[best_hypoth_temp.top().index].hypothesis.size(); j++) {
				output << hypotheses[best_hypoth_temp.top().index].hypothesis(j) << " ";
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
