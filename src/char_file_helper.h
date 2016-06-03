#ifndef FILE_INPUT_CHAR
#define FILE_INPUT_CHAR

//file helper for character stuff


struct file_helper_char {

	int num_unique_chars_source; //per minibatch
	int *h_unique_chars_source;
	int *h_char_vocab_indicies_source;

	int num_unique_chars_target; //per minibatch
	int *h_unique_chars_target;
	int *h_char_vocab_indicies_target;

	std::ifstream input_file; //Input file stream

	file_helper_char(int longest_sent,int longest_word,int minibatch_size,int total_unique_chars_source,
		int total_unique_chars_target,std::string char_file) 
	{
		h_unique_chars_source = (int *)malloc(total_unique_chars_source * sizeof(int));
		h_unique_chars_target = (int *)malloc(total_unique_chars_target * sizeof(int));

		h_char_vocab_indicies_source = (int *)malloc(minibatch_size * longest_sent * longest_word * sizeof(int));
		h_char_vocab_indicies_target = (int *)malloc(minibatch_size * longest_sent * longest_word * sizeof(int));

		BZ_CUDA::logger << "Character file name: " << char_file << "\n";
		BZ_CUDA::logger << "longest_sent in character file: " << longest_sent << "\n";
		BZ_CUDA::logger << "longest_word in character file: " << longest_word << "\n";
		BZ_CUDA::logger << "minibatch_size in character file: " << minibatch_size << "\n";
		input_file.open(char_file.c_str());
	}

	void read_minibatch() {

		//read four lines per minibatch
		//first is the tokenized data
		//next is the unique characters
		//then same for target

		std::string temp_line;
		std::string word;
		int index = 0;

		std::getline(input_file, temp_line);
		std::istringstream iss_source1(temp_line, std::istringstream::in);
		while( iss_source1 >> word ) {
			h_char_vocab_indicies_source[index] = std::stoi(word);
			index+=1;
		}

		//std::cout << "FROM CHAR HELPER ||| Source num index: " << index << "\n";

		index=0;
		std::getline(input_file, temp_line);
		std::istringstream iss_source2(temp_line, std::istringstream::in);
		while( iss_source2 >> word ) {
			h_unique_chars_source[index] = std::stoi(word);
			index+=1;
		}
		num_unique_chars_source = index;


		index = 0;
		std::getline(input_file, temp_line);
		std::istringstream iss_target1(temp_line, std::istringstream::in);
		while( iss_target1 >> word ) {
			h_char_vocab_indicies_target[index] = std::stoi(word);
			index+=1;
		}

		//std::cout << "FROM CHAR HELPER ||| Target num index: " << index << "\n";
		index=0;
		std::getline(input_file, temp_line);
		std::istringstream iss_target2(temp_line, std::istringstream::in);
		while( iss_target2 >> word ) {
			h_unique_chars_target[index] = std::stoi(word);
			index+=1;
		}
		num_unique_chars_target = index;

	}

	void reset_file() {
		input_file.clear();
		input_file.seekg(0, std::ios::beg);
	}

};


#endif