#ifndef FSA_HPP
#define FSA_HPP

#include "fsa.h"

//------------------- state ----------------


// the copy constructer;
state::state(const state& s)
{
    this->name = s.name;
    //this->links = s.links;
    this->weights = s.weights;

    this->next_word_index_set = s.next_word_index_set;
    this->next_word_index_set_ready = s.next_word_index_set_ready;
}



state::state(std::string name){
    this->name = name;
    //this->links = new std::unordered_map<std::string,std::unordered_set<state> >();
    //this->weights = new std::unordered_map<std::string,std::unordered_map<state,float> >();
    this->weights = new std::unordered_map<int,std::unordered_map<std::string, std::pair<state*,float> > >();

    this->next_word_index_set = new std::unordered_set<int>();
    this->next_word_index_set_ready = false;
}

void state::process_link(state* d, int word, float weight, bool log_space){
    if (!log_space)
    {
        weight = std::log(weight);
    }
    if (weights->count(word) == 0){
        (*weights)[word] = std::unordered_map<std::string, std::pair<state*,float>>();
        
    }
    (*weights)[word][d->name] = std::pair<state*,float>(d,weight);
}


std::string state::toString() const{
    std::string s = "";
    s += "Name: "+this->name+ "\n";
    s += "Links:\n";
    for (auto const &i:*(this->weights)){
        
        s += "--"+ std::to_string(i.first)+"--> ";
        for (auto const &j:i.second){
            state* st = j.second.first;
            float weight = j.second.second;
            s += fmt::format("{} {}",st->name,weight);
        }
        
        
        s += '\n';
    }
    return s;
}

std::unordered_set<int>* state::next_word_indicies() {
    if (this->next_word_index_set_ready){
        return this->next_word_index_set;
    } else {
        for (auto & item : *(this->weights)){
            int index = item.first;
            if (index == -1){ // word = *e*
                for (auto & state_item : item.second){
                    //item.second is unordered_map<string, pair<state*,float> >;
                    std::unordered_set<int>* temp_word_index_set = state_item.second.first->next_word_indicies();
                    for (int index : *(temp_word_index_set))
                    {
                        this->next_word_index_set->insert(index);
                    }
                }
            } else {
                this->next_word_index_set->insert(index);
            }
        }
        
        this->h_dict = (int *)malloc(this->next_word_index_set->size()*1*sizeof(int));
        //CUDA_ERROR_WRAPPER(cudaHostRegister(this->h_dict, this->next_word_index_set->size()*1*sizeof(int), cudaHostRegisterPortable),"h_dict in fsa.hpp pinned memeory error!");
        int i = 0;
        for (int index : *(this->next_word_index_set)){
            this->h_dict[i] = index;
            i+=1;
        }
        this->next_word_index_set_ready = true;
        return this->next_word_index_set;
    }
}



void state::next_states(std::vector<sw>& results, int word ){
    // the fsa should not contains a *e* circle.
    
    int c = this->weights->count(word);
    if (c > 0){
        for (auto const &s: this->weights->at(word)){
            sw temp_sw;
            temp_sw.s = s.second.first;
            temp_sw.weight = s.second.second;
            results.push_back(temp_sw);
        }
    }
    int empty = -1;
    c = this->weights->count(empty);
    if (c > 0){
        for (auto const & s: this->weights->at(empty)){
            float weight = s.second.second;
            state* st = s.second.first;
            std::vector<sw> sws;
            st->next_states(sws, word);
            for (auto const & i:sws){
                sw temp_sw;
                temp_sw.s = i.s;
                temp_sw.weight = weight + i.weight;
                results.push_back(temp_sw);
            }
        }
    }
}

//------------------- fsa ----------------

void fsa::print_fsa(){
    std::cout << "start_state: " << this->start_state->name<<"\n" ;
    std::cout << "end_state: " << this->end_state->name<<"\n" ;
    std::cout << "\n";
    for (auto const & s: this->states){
        std::cout << s.second->toString() <<'\n';
    }
    
    std::cout << this->index2words[0] <<"\n";
    std::cout << this->index2words[1] <<"\n";
    std::cout << this->index2words[2] <<"\n";
    
    std::vector<sw> sws;
    this->next_states(this->start_state,1,sws);
    for (auto const & s:sws){
        std::cout << "have: " << s.s->name << "\n";
    }
    
    
    
}

void fsa::load_fsa(){
    //Timer timer;
    
    //timer.start("load_fsa");
    std::chrono::time_point<std::chrono::system_clock> total_start= std::chrono::system_clock::now();
    
    //for (0 (1 "k"))
    boost::regex e3q{"\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+\"(.*)\"[ ]*\\)\\)"};
    
    //for (0 (1 sf))
    boost::regex e3{"\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+([^ ]+)[ ]*\\)\\)"};
    
    //for (0 (1 "k" 0.5))
    boost::regex e4q{"\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+\"(.*)\"[ ]+([^ ]+)[ ]*\\)\\)"};
    //for (0 (1 sf 0.5))
    boost::regex e4{"\\(([^ ]+)[ ]+\\(([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)[ ]*\\)\\)"};
    
    boost::regex regexes[4] = {e3q,e3,e4q,e4};

    
    std::ifstream fin(this->fsa_filename.c_str());
    std::string line;
    // for the end_state;
    std::getline(fin, line);
    states[line] = new state(line);
    end_state = states[line];
    
    bool is_first_link = true;
    int i =0 ;
    int num_links = 0;
    
    float default_weight = 1.0;
    if (this->log_space){
        default_weight = 0.0;
    }
    
    /*
    timer.start("read_lines");
    std::vector<std::string> lines;

    
    while (std::getline(fin,line)) {
        //std::cout<<line<<'\n';
        if (line.size() == 0 || line[0] == '#'){
            continue;
        }
        lines.push_back(line);
    }
    
    timer.end("read_lines");
    
    timer.start("read_lines2");

    //fourObj ** fours = (fourObj **)malloc(lines.size()*sizeof(fourObj*));
    
    
    #pragma omp parallel for
    for (int i =0 ;i < lines.size(); i ++ ){
        fourObj fo;
        line = lines[i];
        boost::smatch sm;
        
        bool matched = false;
        for (int r=0;r<1;r++){
            boost::regex e = regexes[r];
            
            if (boost::regex_search(line,sm,e)){
                //std::cout<<sm.size()<<" "<<r<<"\n";
                fo.s = sm[1];
                fo.d = sm[2];
                
                std::string word = sm[3];
                int word_index = 2;
                if (word == "*e*"){
                    word_index = -1;
                } else {
                    if (this->word2index.count(word)>0){
                        word_index = this->word2index[word];
                    } else {
                        std::cout<<fmt::format("{} is not in vocab set\n",word);
                    }
                }

                fo.word_index = word_index;
                matched = true;
                float weight = default_weight;
                if (sm.size() == 5){
                    weight = std::stof(sm[4]);
                }
                if (sm.size() == 6) {
                    weight = std::stof(sm[5]);
                }
                fo.weight = weight;
                //break;
            }
        }
        if (!matched) {
            std::cerr<<"Error in Line "<<i+2<<": "<<line<<"\n";
            //throw("Error when parsing fsa.");
        }
        //fours[i] = fo;
    }
    */
    /*
    for (int i = 0 ; i < lines.size(); i ++){
        delete fours[i];
    }
    
    free(fours);
     
    
    timer.end("read_lines2");

    */
    
    
    while (std::getline(fin,line)) {
        std::string s;
        std::string d;
        std::string word;
        int word_index = -2;
        float weight = default_weight;
        
        
        //std::cout<<line<<'\n';
        if (line.size() == 0 || line[0] == '#'){
            continue;
        }
        
        //timer.start("regex_match");
        boost::smatch sm;
        bool matched = false;
        for (int r=0;r<4;r++){
            boost::regex e = regexes[r];
            if (boost::regex_search(line,sm,e)){
                //std::cout<<sm.size()<<" "<<r<<"\n";
                s = sm[1];
                d = sm[2];
                word = sm[3];
                matched = true;
                if (sm.size() == 5){
                    weight = std::stof(sm[4]);
                }
                if (sm.size() == 6) {
                    weight = std::stof(sm[5]);
                }
                break;
            }
        }
        
        if (!matched){
            std::cerr<<"Error in Line "<<i+2<<": "<<line<<"\n";
            throw("Error when parsing fsa.");
        }
        
        //timer.end("regex_match");

        //std::cout<<s<<" "<<d<<" "<<word<<" "<<weight<<"\n";
        
        if (states.count(s) == 0){
            states[s] = new state(s);
        }
        if (states.count(d) == 0){
            states[d] = new state(d);
        }
        
        if (is_first_link){
            // for start symbol;
            this->start_state = states[s];
            is_first_link = false;
        }
        
        if (word == "*e*"){
            word_index = -1;
        } else {
            if (this->word2index.count(word)>0){
                word_index = this->word2index[word];
            } else {
                std::cout<<fmt::format("{} is not in vocab set\n",word);
            }
        }
        
        //timer.start("process_link");
        
        if (word_index != -2){
            states[s]->process_link(states[d],word_index,weight,this->log_space);
            num_links += 1;
        }
        
        //timer.end("process_link");
        
        
        i+=1;
    }
    
    if (this->states.count("<EOF>") == 0)
    {
        this->end_state->process_link(this->end_state,this->word2index["<EOF>"],default_weight, this->log_space);
    }
    
    std::chrono::time_point<std::chrono::system_clock> total_end=std::chrono::system_clock::now();
    std::chrono::duration<double> total_dur = total_end - total_start;
    
    std::cout<<"------------------------FSA Info------------------------\n";
    std::cout<<"Number of States: "<< this->states.size() <<"\n";
    std::cout<<"Number of Links: "<< num_links <<"\n";
    std::cout<<"Start State: "<< this->start_state->name <<"\n";
    std::cout<<"End State: "<< this->end_state->name <<"\n";
    std::cout<<"Loading with "<<total_dur.count()<<"s \n";
    std::cout<<"--------------------------------------------------------\n";

    //timer.end("load_fsa");
    
    //timer.report();
    
}

void fsa::convert_name_to_index(std::unordered_map<std::string,int> &dict){
    for (auto const & i:dict){
        //std::cout<<i.first<<" "<<i.second<<"\n";
        this->index2words[i.second] = i.first;
        this->word2index[i.first] = i.second;
        //std::cout<<this->index2words.size()<<"\n";
    }
}


void fsa::next_states(state* current_state,int index, std::vector<sw>& results){
    if (this->index2words.count(index) > 0){
        current_state->next_states(results, index);
    }
}


#endif

