#ifndef FSA_H
#define FSA_H
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/regex.hpp>
#include "format.h"
#include "memory_util.h"

/*
 fsa weight should always (0,1];
 if fsa weight is in log space, then fsa-log = 1;
 */




struct sw;

class state{
public:
    std::string name;
    //std::unordered_map<std::string,std::unordered_set<state> > *links;
    //weights are all stored in log space; weights should be in (0,1]
    std::unordered_map<int,std::unordered_map<std::string,std::pair<state*,float>>> *weights;
    
    // for next word indices;
    std::unordered_set<int> *next_word_index_set;
    int *h_dict;
    bool next_word_index_set_ready;
    
    state(){}
    
    // copy 
    state(const state &s);

    state(std::string name);
    
    void process_link(state *d, int word, float weight, bool log_space);

    std::string toString() const;
    
    void next_states(std::vector<sw>& results, int word);
    
    std::unordered_set<int>* next_word_indicies();

    bool operator==(const state &anotherState) const{
        return (name == anotherState.name);
    }
    
    state& operator=(const state &other){
        this->name = other.name;
        this->weights = other.weights;
        this->next_word_index_set = other.next_word_index_set;
        this->next_word_index_set_ready = other.next_word_index_set_ready;
        return *this;
    }
    
};

struct sw{
    state* s;
    float weight;
};


namespace std{
    template <>
    struct hash<state>
    {
        size_t operator()(const state& k) const{
            return (std::hash<std::string>()(k.name));
        }
    };
    
}

class fsa {
public:
    
    std::string fsa_filename;
    state* start_state;
    state* end_state;
    std::unordered_map<std::string,state*> states;
    std::unordered_map<int,std::string> index2words;
    std::unordered_map<std::string,int> word2index;
    bool log_space = true;
    
    fsa(std::string filename){
        this->fsa_filename = filename;
    }
    
    ~fsa(){
        for (auto &item: states){
            state *s = item.second;
            delete s->weights;
            delete s->next_word_index_set;
            free(s->h_dict);
            delete s;
        }
    }
    
    
    void print_fsa();
    void load_fsa();
    void convert_name_to_index(std::unordered_map<std::string,int> &dict);
    
    void next_states(state* current_state,int index, std::vector<sw>& results);
    
};

/*
class fourObj{
public:
    std::string s;
    std::string d;
    int word_index;
    float weight;
    
    fourObj(){
        s = "";
        d = "";
        word_index = -1;
        weight = 0.0;
    }
};
*/

#endif
