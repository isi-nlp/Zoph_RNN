
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



#endif