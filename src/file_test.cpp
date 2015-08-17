#include <iostream>

#include "input_file_prep.h"

int main() {

	int minibatch_size = 4; //
	bool LM = false;//true is only target being used
	int max_sent_cutoff = 100; //throw away these sentences over this length
	int source_vocab_size = 1500;
	int target_vocab_size = 1000;
	bool shuffle = true;
	int hiddenstate_size = 200;

	std::string source_file_name = "english.txt";
	std::string target_file_name = "spanish.txt";
	std::string output_file_name = "output.txt";
	std::string output_weights_name = "output_weights.txt";

	input_file_prep test_input;
	//test_input.prep_files_train_nonLM(output_weights_name,hiddenstate_size);

	// test_input.prep_files_train_LM(minibatch_size,max_sent_cutoff,
	// 	target_file_name,
	// 	output_file_name,target_vocab_size,
	// 	shuffle,output_weights_name,hiddenstate_size);

	// test_input.prep_files_train_nonLM(minibatch_size,max_sent_cutoff,
	// 	source_file_name,target_file_name,output_file_name,source_vocab_size,target_vocab_size,
	// 	shuffle,output_weights_name,hiddenstate_size);


	source_file_name = "english_dev.txt";
	target_file_name = "spanish_dev.txt";
	std::string tmp_output = "ooo.txt";

	//test_input.integerize_file_nonLM(output_weights_name,source_file_name,target_file_name,tmp_output,max_sent_cutoff);

	//test_input.integerize_file_LM(output_weights_name,source_file_name,tmp_output,max_sent_cutoff);

	test_input.unint_file(output_weights_name,"ints.txt","ints_output.txt",1);

}