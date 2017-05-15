//This is the file reader for the beam decoder
#ifndef FILE_INPUT_DECODER
#define FILE_INPUT_DECODER

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "Eigen_Util.h"
#include "file_helper_char_decoder.h"


struct file_helper_decoder {
  std::string file_name; //Name of input file
  std::ifstream input_file; //Input file stream
  int current_line_in_file = 1;
  int num_lines_in_file;
	
  //Used for computing the maximum sentence length of previous minibatch
  int words_in_sent;
  int sentence_length; //The legnth of the current source sentence

  int max_sent_len; //The max length for a source sentence

  //num rows is the length of minibatch, num columns is len of longest sentence
  //unused positions are padded with -1, since that is not a valid token
  Eigen::Matrix<int,1,Eigen::Dynamic> minibatch_tokens_source_input;


  //-----------------------------------------GPU Parameters---------------------------------------------

  int *h_input_vocab_indicies_source; //This is the pointer to memory on the CPU
  int *h_batch_info;

  bool char_cnn = false;
  file_helper_char_decoder *fhc;

  file_helper_decoder() { }
  

  //Constructor
  file_helper_decoder(std::string file_name,int &num_lines_in_file,int max_sent_len,
                      char_cnn_params &cnp,std::string char_file) {
    this->file_name = file_name;
    input_file.open(file_name.c_str(),std::ifstream::in); //Open the stream to the file
    //this->num_lines_in_file = num_lines_in_file;
    this->max_sent_len = max_sent_len;

    //GPU allocation
    h_input_vocab_indicies_source = (int *)malloc(max_sent_len * sizeof(int));
    h_batch_info = (int *)malloc(2 * sizeof(int));

    int total_words;
    int target_words;
    get_file_stats(num_lines_in_file,total_words,input_file,target_words);
    this->num_lines_in_file = num_lines_in_file;

    if(cnp.char_cnn) {
      this->char_cnn = cnp.char_cnn;
      fhc = new file_helper_char_decoder(max_sent_len,cnp.longest_word,
                                         char_file);
    }
  }

  ~file_helper_decoder() {
    free(h_input_vocab_indicies_source);
    free(h_batch_info);
    input_file.close();
  }

  //Read in the next minibatch from the file
  //returns bool, true is same epoch, false if now need to start new epoch
  bool read_sentence() {

    if(char_cnn) {
      fhc->read_minibatch();
    }

    bool more_lines_in_file = true; //returns false when the file is finished
    words_in_sent=0; //For throughput calculation
    std::vector<int> temp_input_sentence_source; //This stores the source sentence

    std::string temp_input_source; //Temp string for getline to put into
    std::getline(input_file, temp_input_source); //Get the line from the file
    std::istringstream iss_input_source(temp_input_source, std::istringstream::in);
    std::string word; //The temp word

    int current_temp_source_input_index = 0;
    while( iss_input_source >> word ) {
      temp_input_sentence_source.push_back(std::stoi(word));
      h_input_vocab_indicies_source[current_temp_source_input_index] = std::stoi(word);
      current_temp_source_input_index+=1;
    }
		
    words_in_sent = temp_input_sentence_source.size();

    //std::cout << "Current input source length: " << input_source_length << "\n";

    //Now increase current line in file because we have seen two more sentences
    current_line_in_file+=1;

    if(current_line_in_file > num_lines_in_file) {
      current_line_in_file = 1;
      input_file.clear();
      input_file.seekg(0, std::ios::beg);
      more_lines_in_file = false;
      if(char_cnn) {
        fhc->reset_file();
      }
    }

    //Now fill in the minibatch_tokens_input and minibatch_tokens_output
    minibatch_tokens_source_input.resize(1,words_in_sent);
    sentence_length = words_in_sent;
    for(int i=0; i < temp_input_sentence_source.size(); i++) {
      minibatch_tokens_source_input(i) = temp_input_sentence_source[i];
    }

    //get vocab indicies in correct memory layout on the host
    // std::cout << "-------------------source input check--------------------\n";
    // for(int i=0; i < words_in_sent; i++) {
    // 	std::cout << h_input_vocab_indicies_source[i] << "   " << minibatch_tokens_source_input(i) << "\n";
    // }

    h_batch_info[0] = sentence_length;
    h_batch_info[1] = 0;

    return more_lines_in_file;
  }

};

#endif
