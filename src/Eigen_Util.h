
#ifndef EIGEN_UTIL_H
#define EIGEN_UTIL_H

#include <fstream>
//#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

// Functions that take non-const matrices as arguments
// are supposed to declare them const and then use this
// to cast away constness.
#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

#define UNCONST_DIAG(t,c,uc) Eigen::DiagonalBase<t> &uc = const_cast<Eigen::DiagonalBase<t>&>(c);

struct sigmoid_functor {
	template<typename dType>
  dType operator() (dType x) const { return 1/(1+std::exp(-x)); }
};

struct tanh_functor {
	template<typename dType>
  dType operator() (dType x) const { return std::tanh(x); }
};

struct tanh_sq_functor {
	template<typename dType>
  dType operator() (dType x) const { return std::tanh(x)*std::tanh(x); }
};


struct exp_functor {
	template<typename dType>
  dType operator() (dType x) const { return std::exp(x); }
};

template<typename Derived>
void readMatrix(const Eigen::MatrixBase<Derived> &matrixConst,std::ifstream &input) {
	UNCONST(Derived, matrixConst, matrixUnconst);

	std::string temp_string;
	std::string temp_token;
	//std::cout << matrixConst.rows() << "\n";
	//std::cout << matrixConst.cols() << "\n";
	for(int i=0; i<matrixConst.rows(); i++) {
		//std::string temp_string;
		std::getline(input, temp_string);
		//input.sync();
		//std::cout << temp_string << " ||| "<< i<<"\n";
		std::istringstream iss_input(temp_string, std::istringstream::in);
		for(int j=0; j<matrixConst.cols(); j++) {
			//std::string temp_token;
			iss_input >> temp_token;
			//std::cout << temp_token << "\n";
			//std::cout << temp_token << "\n";
			matrixUnconst(i,j) = std::stod(temp_token);
		}
	}
	//std::string temp_string;
	//get the final space
	std::getline(input, temp_string);
}



template<typename Derived>
void writeMatrix(const Eigen::MatrixBase<Derived> &matrix_const,std::ofstream &output) {

	for(int i=0; i<matrix_const.rows(); i++) {
		for(int j=0; j<matrix_const.cols(); j++) {
			output << matrix_const(i,j);
			if(j!=matrix_const.cols()-1) {
				output << " ";
			}
		}
		output << "\n";
	}
	output << "\n";
}


template<typename dType>
void read_matrix_GPU(dType *d_mat,int rows,int cols,std::ifstream &input) {

	//thrust::device_ptr<dType> d_ptr = thrust::device_pointer_cast(d_mat);
	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	
	std::string temp_string;
	std::string temp_token;
	//std::cout << matrixConst.rows() << "\n";
	//std::cout << matrixConst.cols() << "\n";
	for(int i=0; i<rows; i++) {
		//std::string temp_string;
		std::getline(input, temp_string);
		//input.sync();
		//std::cout << temp_string << " ||| "<< i<<"\n";
		std::istringstream iss_input(temp_string, std::istringstream::in);
		for(int j=0; j<cols; j++) {
			//std::string temp_token;
			iss_input >> temp_token;
			//std::cout << temp_token << "\n";
			temp_mat[IDX2C(i,j,rows)] = std::stod(temp_token);
		}
	}
	//std::string temp_string;
	//get the final space
	std::getline(input, temp_string);

	cudaMemcpy(d_mat,temp_mat,rows*cols*sizeof(dType),cudaMemcpyHostToDevice);
	free(temp_mat);
}


template<typename dType>
void read_matrix_GPU_T(dType *d_mat,int rows,int cols,std::ifstream &input) {

	//thrust::device_ptr<dType> d_ptr = thrust::device_pointer_cast(d_mat);
	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	
	std::string temp_string;
	std::string temp_token;
	//std::cout << matrixConst.rows() << "\n";
	//std::cout << matrixConst.cols() << "\n";
	for(int i=0; i<cols; i++) {
		//std::string temp_string;
		std::getline(input, temp_string);
		//input.sync();
		//std::cout << temp_string << " ||| "<< i<<"\n";
		std::istringstream iss_input(temp_string, std::istringstream::in);
		for(int j=0; j<rows; j++) {
			//std::string temp_token;
			iss_input >> temp_token;
			//std::cout << temp_token << "\n";
			temp_mat[IDX2C(j,i,rows)] = std::stod(temp_token);
		}
	}
	//std::string temp_string;
	//get the final space
	std::getline(input, temp_string);

	cudaMemcpy(d_mat,temp_mat,rows*cols*sizeof(dType),cudaMemcpyHostToDevice);
	free(temp_mat);
}


template<typename dType>
void write_matrix_GPU(dType *d_mat,int rows,int cols,std::ofstream &output) {
	//thrust::device_ptr<dType> d_ptr = thrust::device_pointer_cast(d_mat);
	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	cudaMemcpy(temp_mat,d_mat,rows*cols*sizeof(dType),cudaMemcpyDeviceToHost);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			output << temp_mat[IDX2C(i,j,rows)];
			if(j!=cols-1) {
				output << " ";
			}
		}
		output << "\n";
	}
	output << "\n";
	free(temp_mat);
}

template<typename dType>
void write_matrix_GPU_T(dType *d_mat,int rows,int cols,std::ofstream &output) {
	//thrust::device_ptr<dType> d_ptr = thrust::device_pointer_cast(d_mat);
	dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));
	cudaMemcpy(temp_mat,d_mat,rows*cols*sizeof(dType),cudaMemcpyDeviceToHost);
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			output << temp_mat[IDX2C(i,j,rows)];
			if(j!=rows-1) {
				output << " ";
			}
		}
		output << "\n";
	}
	output << "\n";
	free(temp_mat);
}


template<typename Derived,typename dType>
void clipNorm(const Eigen::MatrixBase<Derived> &gradient_const,dType norm,dType norm_clip) {
	UNCONST(Derived, gradient_const, gradient);
	gradient = (norm_clip/norm)*gradient;
}

//get the norm for matrix
template<typename Derived,typename dType>
void computeNorm(const Eigen::MatrixBase<Derived> &gradient,dType norm_clip) {
	dType norm = std::sqrt(gradient.array().square().sum());
	if(norm>norm_clip) {
		clipNorm(gradient,norm,norm_clip);
	}
}


//counts the total number of words in my file format, so you can halve learning rate at half epochs
//counts the total number of lines too
void get_file_stats(int &num_lines,int &num_words,std::ifstream &input,int &total_target_words) {
	std::string str; 
	std::string word;
	num_lines =0;
	num_words=0;
	total_target_words=0;
    while (std::getline(input, str)){
        num_lines++;
    }

    input.clear();
	input.seekg(0, std::ios::beg);
    // if(num_lines%4!=0) {
    // 	std::cout << "ERROR FILE BEING READ IN IS NOT CORRECT FORMAT\n";
    // 	exit (EXIT_FAILURE);
    // }

    for(int i=0; i<num_lines; i+=4) {
    	std::getline(input, str);//source input
    	std::istringstream iss_input_source(str, std::istringstream::in);
    	while( iss_input_source >> word ) {
    		if(std::stoi(word) !=-1) {
    			num_words+=1;
    		}
    	}
    	std::getline(input, str);//source output,dont use
    	std::getline(input, str);//target input
    	std::istringstream iss_input_target(str, std::istringstream::in);
    	while( iss_input_target >> word ) {
    		if(std::stoi(word) != -1) {
    			num_words+=1;
    			total_target_words++;
    		}
    	}
    	std::getline(input, str);//target output,done use
    }
    input.clear();
	input.seekg(0, std::ios::beg);
}

//for multisource MT
void get_file_stats_source(int &num_lines,std::ifstream &input) {
	std::string str; 
	std::string word;
	num_lines =0;
    while (std::getline(input, str)){
        num_lines++;
    }

    input.clear();
	input.seekg(0, std::ios::beg);
}


//for using the charCNN model
void extract_char_info(int &longest_word,int &num_unique_chars_source,int &num_unique_chars_target,
	int &source_vocab_size,int &target_vocab_size,std::string char_mapping_name,std::string word_mapping_name) 
{
	std::ifstream char_stream;
	std::ifstream word_stream;
	char_stream.open(char_mapping_name.c_str());
	word_stream.open(word_mapping_name.c_str());

	std::vector<std::string> params;
	std::string temp_line; //for getline
	std::string temp_word;

	std::getline(char_stream,temp_line);
	std::istringstream my_ss(temp_line, std::istringstream::in);
	while(my_ss >> temp_word) {
		params.push_back(temp_word);
	}

	num_unique_chars_source = std::stoi(params[0]);
	num_unique_chars_target = std::stoi(params[1]);
	longest_word = std::stoi(params[2]);

	params.clear();
	std::getline(word_stream,temp_line);
	std::istringstream my_sss(temp_line, std::istringstream::in);
	while(my_sss >> temp_word) {
		params.push_back(temp_word);
	}

	target_vocab_size = std::stoi(params[2]);
	source_vocab_size = std::stoi(params[3]);

	char_stream.close();
	word_stream.close();
}


//for charCNN decoding
void extract_charCNN(std::unordered_map<int,std::vector<int>> &word_to_char_map,std::string file_name) {
	
	std::ifstream mapping_stream;
	mapping_stream.open(file_name.c_str());
	std::string temp_line; //for getline
	std::string temp_word;

    while (std::getline(mapping_stream, temp_line)){
    	std::istringstream ss(temp_line, std::istringstream::in);
    	bool first = true;
    	std::vector<int> temp_vec;
    	int key = -1;
		while(ss >> temp_word) {
			if(first) {
				first = false;
				key = std::stoi(temp_word);
			}
			else {
				temp_vec.push_back(std::stoi(temp_word));
			}
		}
		if(word_to_char_map.count(key)!=0) {
			BZ_CUDA::logger << "ERROR IN extract_charCNN\n";
			exit (EXIT_FAILURE);
		}
		word_to_char_map[key] = temp_vec;
		//std::cout << "Word: " << key << "\n";
		// std::cout << "Char: ";
		// for(int i=0; i<temp_vec.size(); i++) {
		// 	std::cout << temp_vec[i] << " ";
		// }
		// std::cout << "\n\n";
    }
    mapping_stream.close();
}

//for charCNN decoding
void create_char_vocab(int *h_word_indicies,int num_words,int longest_word,int *h_char_indicies,
	std::unordered_map<int,std::vector<int>> &word_to_char_map) 
{
	int curr_char_index=0;
	for(int i=0; i<num_words; i++) {
		int word_index = h_word_indicies[i];
		//std::cout << "Word index: " << word_index << "\n";
		std::vector<int> char_vec = word_to_char_map[word_index];
		// std::cout << "char vec:\n";
		// for(int j=0; j<char_vec.size(); j++) {
		// 	std::cout << char_vec[j] << " ";
		// }
		// std::cout << "\n\n";

		int num_pad = longest_word - char_vec.size();
		for(int j=0; j<char_vec.size(); j++) {
			h_char_indicies[curr_char_index] = char_vec[j];
			curr_char_index++;
		}
		for(int j=0; j<num_pad; j++) {
			h_char_indicies[curr_char_index] = -1;
			curr_char_index++;
		}
	}
}

template<typename dType>
Eigen::Matrix<dType,Eigen::Dynamic, 1> readCol_GPU2Eigen(dType *d_mat, dType *temp_mat, int col_index, int rows, int cols){
    
    CUDA_ERROR_WRAPPER(cudaMemcpy(temp_mat,d_mat,rows*cols*sizeof(dType),cudaMemcpyDeviceToHost),"readCol_GPU2Eigen");
    
    Eigen::Matrix<dType,Eigen::Dynamic, 1> mat;
    mat.resize(rows,1);
    
    int start = col_index * rows;
    for (int i = 0; i < rows; i ++){
        mat(i,0) = temp_mat[start+i];
    }
    return mat;
}

template<typename dType>
void writeColBroadcast_Eigen2GPU(dType *d_mat, const Eigen::Matrix<dType,Eigen::Dynamic, 1> &mat, int rows, int cols){
    
    dType *temp_mat = (dType *)malloc(rows*cols*sizeof(dType));

    for (int col = 0; col < cols; col ++){
        for (int row = 0; row < rows; row ++){
            temp_mat[col*cols + row] = mat(row, 1);
        }

    }CUDA_ERROR_WRAPPER(cudaMemcpy(d_mat,temp_mat,rows*cols*sizeof(dType),cudaMemcpyHostToDevice),"writeColBroadcast_Eigen2GPU");

    free(temp_mat);

}



#endif
