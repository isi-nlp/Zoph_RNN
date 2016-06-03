//function for adding in the model information
#ifndef ADD_MODEL_INFO_H
#define ADD_MODEL_INFO_H

#include <fstream>
#include <string>
void add_model_info(int num_layers,int LSTM_size,int target_vocab_size,int source_vocab_size,bool attention_model, bool feed_input,bool multi_source,bool combine_LSTM,bool char_cnn,std::string filename) {
    std::ifstream input(filename.c_str());
    std::string output_string = std::to_string(num_layers) +" "+ std::to_string(LSTM_size) +" "+ std::to_string(target_vocab_size) +" "+ std::to_string(source_vocab_size) +" "+ std::to_string(attention_model) + " " + std::to_string(feed_input) + " " + std::to_string(multi_source) + " " + std::to_string(combine_LSTM) +" "+ std::to_string(char_cnn);
    std::string str; 
    std::vector<std::string> file_lines;
    std::getline(input,str);//first line that is being replace
    file_lines.push_back(output_string);
    while(std::getline(input,str)) {
        file_lines.push_back(str);
    }
    input.close();
    std::ofstream output(filename.c_str());
    for(int i=0; i<file_lines.size(); i++) {
        output << file_lines[i] << "\n";
    }
    output.close();
} 
#endif
