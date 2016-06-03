#ifndef FILE_INPUT_CHAR_DECODER
#define FILE_INPUT_CHAR_DECODER

//file helper for character stuff


struct file_helper_char_decoder {

	int num_unique_chars_source; //per minibatch
	int *h_char_vocab_indicies_source;

	
	std::ifstream input_file; //Input file stream

	file_helper_char_decoder(int longest_sent,int longest_word,
		std::string char_file) 
	{
		h_char_vocab_indicies_source = (int *)malloc(longest_sent * longest_word * sizeof(int));
		input_file.open(char_file.c_str());
	}

	void read_minibatch() {

		//read one lines per minibatch
		//first is the tokenized data

		std::string temp_line;
		std::string word;
		int index = 0;

		//std::cout << "Getting line from character file\n";
		std::getline(input_file, temp_line);
		std::istringstream iss_source1(temp_line, std::istringstream::in);
		while( iss_source1 >> word ) {
			h_char_vocab_indicies_source[index] = std::stoi(word);
			//std::cout << h_char_vocab_indicies_source[index] << " ";
			index+=1;
		}
		//std::cout << "\n";
	}

	void reset_file() {
		input_file.clear();
		input_file.seekg(0, std::ios::beg);
	}

};


#endif