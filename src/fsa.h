#ifndef FSA_H
#define FSA_H
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/regex.hpp>
#include "format.h"
#include <boost/dynamic_bitset.hpp>

/*
 fsa weight should always (0,1];
 if fsa weight is in log space, then fsa-log = 1;
 */




struct sw;
struct swb;

class state{
public:
    std::string name;
    //std::unordered_map<std::string,std::unordered_set<state> > *links;
    //weights are all stored in log space; weights should be in (0,1]
    std::unordered_map<int,std::unordered_map<state,float> > *weights;

    
    // specific for the empty info
    std::vector<std::pair<int,int>> *empty_info;
    
    state(){}
    
    // copy 
    state(const state &s);

    state(std::string name);
    
    void process_link(state &d, int word, float weight, bool log_space, std::string info="");

    std::string toString() const;
    
    void next_states(std::vector<sw>& results, int word);
    
    void next_word_indicies(std::unordered_set<int> &results) const;

    bool operator==(const state &anotherState) const{
        return (name == anotherState.name);
    }
    
    state& operator=(const state &other){
        this->name = other.name;
        //this->links = other.links;
        this->weights = other.weights;
        this->empty_info = other.empty_info;
        return *this;
    }
    
};

struct sw{
    state s;
    float weight;
};

struct swb{
    state s;
    float weight;
    std::vector<boost::dynamic_bitset<>> masks;
};



namespace std{
    template <>
    struct hash<state>
    {
        size_t operator()(const state& k) const{
            return (std::hash<std::string>()(k.name));
        }
    };
    
    template <typename B, typename A>
    struct hash<boost::dynamic_bitset<B,A>>{
        size_t operator()(const boost::dynamic_bitset<B,A>& bs) const{
            return boost::hash_value(bs.m_bits);
        }
    };
}

class fsa {
public:
    
    std::string fsa_filename;
    state* start_state;
    state* end_state;
    std::unordered_map<std::string,state> states;
    std::unordered_map<int,std::string> index2words;
    std::unordered_map<std::string,int> word2index;
    bool log_space = false;
    
    fsa(std::string filename){
        this->fsa_filename = filename;
    }
    
    ~fsa(){
        for (auto &item: states){
            state &s = item.second;
            delete s.weights;
            delete s.empty_info;
          
        }
    }
    
    
    void print_fsa();
    void load_fsa();
    void convert_name_to_index(std::unordered_map<std::string,int> &dict);
    
    void next_states(state current_state,int index, std::vector<sw>& results);
    
};

#endif
