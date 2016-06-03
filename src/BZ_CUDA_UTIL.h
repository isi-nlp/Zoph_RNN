//CUDA utilility for LSTM RNN

#ifndef BZ_CUDA_UTIL_H
#define BZ_CUDA_UTIL_H

#include <stdlib.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <stdio.h>   
#include <stdlib.h>

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <curand.h>
#include <thrust/iterator/constant_iterator.h>
#include "cuda_profiler_api.h"

//This is used since all cuBLAS storage is column major
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//std::ofstream HPC_output;


namespace deniz {
	bool source_side = false;
	bool train_source_input_embedding = true;
	bool train_target_input_embedding = true;
	bool train_target_output_embedding = true;
	bool train_source_RNN = true;
	bool train_target_RNN = true;
	bool train_attention_target_RNN = true;

	bool soft_regularizer = false;
	precision train_source_input_embedding_lambda = 0;
	precision train_target_input_embedding_lambda = 0;
	precision train_target_output_embedding_lambda = 0;
	precision train_source_RNN_lambda = 0;
	precision train_target_RNN_lambda = 0;
	precision train_attention_target_RNN_lambda = 0;
}



//for t-sne stuff for paper
namespace BZ_STATS {
	precision *h_dump_ht = NULL;
	bool tsne_dump = false;
	std::ofstream tsne_dump_stream;//("tsne_dump_COMB.txt");
}

//namespace to hold constants
namespace BZ_CUDA {

//for logging the output
//bool HPC_output = false;
OutputLogger logger;

bool cont_train = false;
bool shuffle_data=true;

//for ensembling pre-normalization
bool pre_norm = false;


//for dumping the best model
bool dump_every_best = false;
int curr_dump_num = 1;


//stuff for unk replacement using attention
bool unk_replacement = false;
std::string unk_rep_file_name;
std::ofstream unk_rep_file_stream;
std::vector<int> viterbi_alignments;
std::vector<int> all_viterbi_alignments;
std::vector<precision> alignment_scores; //for ensembling alignment values
int *h_align_indicies;
precision *h_alignment_values;

bool print_norms = false;

unsigned int curr_seed = 0;

//for not storing extra stuff during testing
bool force_decode = false;

//FOR BAD NCE DUMP
bool nce_legacy_dump = false;

boost::random::mt19937 gen;
double lower = -0.08;
double upper = 0.08;

bool global_clip_flag = false;
precision global_norm = 0; //the global norm for gradient clipping
precision global_norm_threshold;


//clip errors with respect to h_t and c_t
bool clip_cell = false;
precision cell_clip_threshold = 50;
precision error_clip_threshold = 1000;


//for stats on gradient norms
double recent_sum = 0;


//grad clipping
bool individual_grad_clip = false;
precision ind_norm_clip_thres = 0.1;

//for gettings only NCE scores (used for reranking, etc ...)
bool nce_score = false;

//for NCE stats for paper
bool dump_NCE_stats = false;
std::string NCE_file_dump_name = "ASHISH_DUMP.txt";
std::ofstream NCE_file_dump;//(NCE_file_dump_name.c_str());
precision *h_h_t_storage;
double *h_part_vals;
double *d_part_vals;

//partition function calculation for NCE
bool print_partition_function = false;
std::vector<double> full_partition_vals; //all the partition function values

void print_partition_stats() {
	double total_sum = 0;
	double mean = 0;
	double variance = 0;

	for(int i=0; i<full_partition_vals.size(); i++) {
		total_sum+=full_partition_vals[i];
	}
	mean = total_sum/full_partition_vals.size();

	for(int i=0; i<full_partition_vals.size(); i++) {
		variance+= (full_partition_vals[i] - mean)*(full_partition_vals[i] - mean);
	}

	variance = variance/full_partition_vals.size();

	BZ_CUDA::logger << "\n-------------------NCE PARTITION STATS------------------\n";
	BZ_CUDA::logger << "Partition mean: " << mean << "\n";
	BZ_CUDA::logger << "Partition function standard deviation: " << std::sqrt(variance) << "\n\n\n";

	full_partition_vals.clear();
}


} //BZ_CUDA namespace



#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}


#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}



#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

// void CUDA_ERROR_WRAPPER(cudaError_t cudaStat,std::string error_message) {
// 	if (cudaStat != cudaSuccess) {
// 		std::cout << error_message << std::endl;
// 		exit (EXIT_FAILURE);
// 	}
// }


void CUDA_ERROR_WRAPPER(cudaError_t cudaStat,std::string error_message) {

	if ( cudaSuccess != cudaStat ) {
		BZ_CUDA::logger << "Error\n";
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(cudaStat));
		BZ_CUDA::logger << error_message << "\n";
		exit (EXIT_FAILURE);
	}
}



std::string cublasErrorString(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}




void CUBLAS_ERROR_WRAPPER(cublasStatus_t cudaStat,std::string error_message) {
	//if (cudaStat != cudaSuccess) {
      if (cudaStat != CUBLAS_STATUS_SUCCESS) {
		std::string msg = cublasErrorString(cudaStat);

		std::cout << error_message << std::endl;
		BZ_CUDA::logger << msg << "\n";

		exit (EXIT_FAILURE);
	}
}


 void CUDA_GET_LAST_ERROR() {
	cudaError_t code = cudaGetLastError();
	if ( cudaSuccess != code ) {
		BZ_CUDA::logger << "Error in kernel\n";
		BZ_CUDA::logger << "NO MESSAGE\n";
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
		exit (EXIT_FAILURE);
	}
}

 void CUDA_GET_LAST_ERROR(std::string msg) {
	cudaError_t code = cudaGetLastError();
	if ( cudaSuccess != code ) {
		BZ_CUDA::logger << "Error in kernel\n";
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(code));
		BZ_CUDA::logger << msg << "\n";
		exit (EXIT_FAILURE);
	}
}

// void CUDA_GET_LAST_ERROR(std::string message) {
// 	if ( cudaSuccess != cudaGetLastError() ) {
// 		std::cout << "Error in kernel: " << message << "\n" ;
// 	}
// }

//Can be used for either double or float, use floats for performance, but doubles for gradient checking
template<typename dType>
void initialize_Matrix(dType *h_matrix,int rows,int cols) {
	boost::uniform_real<> distribution(BZ_CUDA::lower,BZ_CUDA::upper);
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			h_matrix[IDX2C(i,j,rows)] =  (dType)distribution(BZ_CUDA::gen);
		}
	}
}


template<typename dType>
void initialize_Matrix_GPU(dType *d_matrix,int rows,int cols) {
	boost::uniform_real<> distribution(BZ_CUDA::lower,BZ_CUDA::upper);
	thrust::device_ptr<dType> mat_ptr = thrust::device_pointer_cast(d_matrix);
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			mat_ptr[IDX2C(i,j,rows)] =  (dType)distribution(BZ_CUDA::gen);
		}
	}
}

template<typename dType>
void initialize_Matrix_ones(dType *h_matrix,int rows,int cols) {
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			h_matrix[IDX2C(i,j,rows)] =  1;
		}
	}
}

template<typename dType>
void initialize_Matrix_zeros(dType *h_matrix,int rows,int cols) {
	for(int j=0; j<cols; j++) {
		for(int i=0; i<rows; i++) {
			h_matrix[IDX2C(i,j,rows)] =  0;
		}
	}
}


template<typename dType>
void allocate_Matrix_CPU(dType **h_matrix,int rows,int cols) {
	*h_matrix = (dType *)malloc(rows*cols*sizeof(dType));
}

template<typename dType>
void allocate_Matrix_GPU(dType **d_matrix,int rows,int cols) {
	CUDA_ERROR_WRAPPER(cudaMalloc((void**)d_matrix, rows*cols*sizeof(dType)),"GPU memory allocation failed\n");
}

template<typename dType>
void set_matrix_cuBLAS(dType *h_matrix,dType *d_matrix,int rows,int cols) {
	CUBLAS_ERROR_WRAPPER(cublasSetMatrix(rows, cols, sizeof(dType), h_matrix, rows, d_matrix, rows),"cuBLAS set matrix failed\n");
}

template<typename dType>
void set_vector_cuBLAS(dType *h_vector,dType *d_vector,int rows) {
	CUBLAS_ERROR_WRAPPER(cublasSetVector(rows, sizeof(dType), h_vector, 1, d_vector, 1),"cuBLAS set vector failed\n");
}

template<typename dType>
void get_matrix_cuBLAS(dType *h_matrix,dType *d_matrix,int rows,int cols) {
	CUBLAS_ERROR_WRAPPER(cublasGetMatrix(rows, cols, sizeof(dType), d_matrix, rows, h_matrix, rows),"cuBLAS get matrix failed\n");
}

template<typename dType>
void get_vector_cuBLAS(dType *h_vector,dType *d_vector,int rows) {
	CUBLAS_ERROR_WRAPPER(cublasGetVector(rows, sizeof(dType), d_vector, 1, h_vector, 1),"cuBLAS get vector failed\n");
}

template<typename dType>
void full_matrix_setup(dType **h_matrix,dType **d_matrix,int rows,int cols) {
	allocate_Matrix_CPU(h_matrix,rows,cols);
	initialize_Matrix(*h_matrix,rows,cols);
	allocate_Matrix_GPU(d_matrix,rows,cols);
	set_matrix_cuBLAS(*h_matrix,*d_matrix,rows,cols);

	free(*h_matrix);
}


template<typename dType>
void full_matrix_setup_0(dType **h_matrix,dType **d_matrix,int rows,int cols) {
	allocate_Matrix_CPU(h_matrix,rows,cols);
	initialize_Matrix_zeros(*h_matrix,rows,cols);
	allocate_Matrix_GPU(d_matrix,rows,cols);
	set_matrix_cuBLAS(*h_matrix,*d_matrix,rows,cols);

	free(*h_matrix);
}


template<typename dType>
void full_vector_setup(dType **h_vector,dType **d_vector,int rows) {
	allocate_Matrix_CPU(h_vector,rows,1);
	initialize_Matrix(*h_vector,rows,1);
	allocate_Matrix_GPU(d_vector,rows,1);
	set_vector_cuBLAS(*h_vector,*d_vector,rows);


	free(*h_vector);
}

template<typename dType>
void full_vector_setup_ones(dType **h_vector,dType **d_vector,int rows) {
	allocate_Matrix_CPU(h_vector,rows,1);
	initialize_Matrix_ones(*h_vector,rows,1);
	allocate_Matrix_GPU(d_vector,rows,1);
	set_vector_cuBLAS(*h_vector,*d_vector,rows);

	free(*h_vector);
}

void initialize_vector_vocab(int *h_vector,int rows,int vocab_size) {
	boost::uniform_real<> distribution(0,1);
	for(int i=0; i<rows; i++) {
		h_vector[i] = (int)(vocab_size*distribution(BZ_CUDA::gen));
	}
}

void initialize_vector_vocab_01(int *h_vector,int rows) {
	 srand (time(NULL));
	for(int i=0; i<rows; i++) {
		h_vector[i] = (int)(rand()%2);
	}
}

void full_vector_setup_vocab(int **h_vector,int **d_vector,int rows,int vocab_size) {
	allocate_Matrix_CPU(h_vector,rows,1);
	initialize_vector_vocab(*h_vector,rows,vocab_size);
	allocate_Matrix_GPU(d_vector,rows,1);
	set_vector_cuBLAS(*h_vector,*d_vector,rows);

	free(*h_vector);
}

void full_vector_setup_vocab_01(int **h_vector,int **d_vector,int rows) {
	allocate_Matrix_CPU(h_vector,rows,1);
	initialize_vector_vocab_01(*h_vector,rows);
	allocate_Matrix_GPU(d_vector,rows,1);
	set_vector_cuBLAS(*h_vector,*d_vector,rows);

	free(*h_vector);
}

template<typename dType>
void print_matrix(dType *h_matrix,int rows,int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			BZ_CUDA::logger << h_matrix[IDX2C(i,j,rows)] << " ";
		}
		BZ_CUDA::logger << "\n";
	}
	BZ_CUDA::logger << "\n";
}


template<typename Derived>
void print_eigen_matrix(const Eigen::MatrixBase<Derived> &h_mat) {
	for(int i=0; i<h_mat.rows(); i++) {
		for(int j=0; j<h_mat.cols(); j++) {
			BZ_CUDA::logger << h_mat(i,j) << " ";
		}
		BZ_CUDA::logger << "\n";
	}
	BZ_CUDA::logger << "\n";
}

template<typename dType>
void print_thrust_matrix(thrust::host_vector<dType> &h_mat,int rows,int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			BZ_CUDA::logger << h_mat[IDX2C(i,j,rows)] << " ";
		}
		BZ_CUDA::logger << "\n";
	}
	BZ_CUDA::logger << "\n";
}



//returns true if eigen matrix is the same, false otherwise
template<typename Derived,typename dType>
bool eigen_check(const Eigen::MatrixBase<Derived> &h_eigen_mat,dType *h_cuda_matrix) {
	for(int i=0; i<h_eigen_mat.rows(); i++) {
		for(int j=0; j<h_eigen_mat.cols(); j++) {
			if(h_eigen_mat(i,j) != h_cuda_matrix[IDX2C(i,j,h_eigen_mat.rows())]) {
				return false;
			}
		}
	}
	return true;
}

//returns true if eigen matrix is the same, false otherwise
template<typename Derived,typename dType>
bool eigen_check_thres(const Eigen::MatrixBase<Derived> &h_eigen_mat,dType *h_cuda_matrix,dType threshold) {
	int num_bad = 0;
	dType max_fail = 0;
	dType average_fail = 0;
	for(int i=0; i<h_eigen_mat.rows(); i++) {
		for(int j=0; j<h_eigen_mat.cols(); j++) {
			if(  std::abs(h_eigen_mat(i,j) -h_cuda_matrix[IDX2C(i,j,h_eigen_mat.rows())]) > threshold  ) {
				//std::cout << "Eigen check failing at: " << i << " " << j << "\n";
				//std::cout << "Difference: " << std::abs(h_eigen_mat(i,j) - h_cuda_matrix[IDX2C(i,j,h_eigen_mat.rows())]) << "\n";
				dType diff = std::abs(h_eigen_mat(i,j)-h_cuda_matrix[IDX2C(i,j,h_eigen_mat.rows())] );
				average_fail+=diff;
				if(diff > max_fail) {
					max_fail = diff;
				}
				num_bad++;
			}
		}
	}
	
	if(num_bad > 0) {
		BZ_CUDA::logger << "Total that could fail: " << h_eigen_mat.rows()*h_eigen_mat.cols() << "\n";
		BZ_CUDA::logger << "Number in eigen check that failed: " << num_bad << "\n";
		BZ_CUDA::logger << "Max fail: " << max_fail << "\n";
		BZ_CUDA::logger << "average fail: " << average_fail/num_bad << "\n";
		return false;
	} 
	return true;
}

#include <set>
template<typename Derived,typename dType>
void eigen_check_thrust_ptr(const Eigen::MatrixBase<Derived> &h_eigen_mat,dType *d_ptr,std::string msg,dType threshold) {
	//thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(d_ptr);

	int tot_size = h_eigen_mat.rows()*h_eigen_mat.cols()*sizeof(dType);
	dType * h_temp = (dType *)malloc(tot_size);
	cudaMemcpy(h_temp, d_ptr, tot_size, cudaMemcpyDeviceToHost);
	int num_bad =0;
	dType max_fail = 0;
	dType average_fail = 0;
	std::set<int> myset;
	std::set<int> myset2;
	for(int i=0; i<h_eigen_mat.rows(); i++) {
		for(int j=0; j<h_eigen_mat.cols(); j++) {
			if(  std::abs(h_eigen_mat(i,j) -h_temp[IDX2C(i,j,h_eigen_mat.rows())]) > threshold  ) {
				dType diff = std::abs(h_eigen_mat(i,j)-h_temp[IDX2C(i,j,h_eigen_mat.rows())] );
				average_fail+=diff;
				if(diff > max_fail) {
					max_fail = diff;
				}
				myset.insert(j);
				num_bad++;
				myset2.insert(i);
			}
		}
	}
	
	if(num_bad > 0) {
		BZ_CUDA::logger << "Operation: " << msg << " failed\n";
		BZ_CUDA::logger << "Total that could fail: " << h_eigen_mat.rows()*h_eigen_mat.cols() << "\n";
		BZ_CUDA::logger << "Number in eigen check that failed: " << num_bad << "\n";
		BZ_CUDA::logger << "Max fail: " << max_fail << "\n";
		BZ_CUDA::logger << "average fail: " << average_fail/num_bad << "\n";
		for (auto it=myset.begin(); it!=myset.end(); ++it)
    		BZ_CUDA::logger << ' ' << *it;
    	BZ_CUDA::logger << "\n\n";

    	for (auto it=myset2.begin(); it!=myset2.end(); ++it)
    		BZ_CUDA::logger << ' ' << *it;
    	BZ_CUDA::logger << "\n\n";
    	//std::cout << h_eigen_mat << "\n\n\n\n";
  //   	for(int i=0; i<h_eigen_mat.rows(); i++) {
		// 	for(int j=0; j<h_eigen_mat.cols(); j++) {
		// 		std::cout << h_temp[IDX2C(i,j,h_eigen_mat.rows())] << " ";
		// 	}
		// 	std::cout << "\n";
		// }
		BZ_CUDA::logger << "\n";
		exit (EXIT_FAILURE);
	} 
	free(h_temp);
}

template<typename dType>
void check_GPU_GPU(dType *mat1,dType *mat2,dType threshold,int rows,int cols,std::string msg) {
	thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(mat1);
	thrust::device_ptr<dType> debug_ptr2 = thrust::device_pointer_cast(mat2);
	int num_bad =0;
	dType max_fail = 0;
	dType average_fail = 0;

	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			int idx = IDX2C(i,j,rows);
			if(  std::abs(debug_ptr2[idx] - debug_ptr[idx]) > threshold  ) {
				dType diff = std::abs( debug_ptr2[idx] - debug_ptr[idx] );
				average_fail+=diff;
				if(diff > max_fail) {
					max_fail = diff;
				}
				num_bad++;
			}
		}
	}

	if(num_bad > 0) {
		BZ_CUDA::logger << "Operation: " << msg << " failed\n";
		BZ_CUDA::logger << "Total that could fail: " << rows*cols << "\n";
		BZ_CUDA::logger << "Number in eigen check that failed: " << num_bad << "\n";
		BZ_CUDA::logger << "Max fail: " << max_fail << "\n";
		BZ_CUDA::logger << "average fail: " << average_fail/num_bad << "\n";
		exit (EXIT_FAILURE);
	} 
}


//returns true if eigen matrix is the same, false otherwise
template<typename Derived,typename dType>
bool eigen_check_thres_thrust(const Eigen::MatrixBase<Derived> &h_eigen_mat,thrust::host_vector<dType> &h_mat,dType threshold) {
	for(int i=0; i<h_eigen_mat.rows(); i++) {
		for(int j=0; j<h_eigen_mat.cols(); j++) {
			if(  std::abs(h_eigen_mat(i,j) -h_mat[IDX2C(i,j,h_eigen_mat.rows())]) > threshold  ) {
				return false;
			}
		}
	}
	return true;
}

//Copy a matrix in column major format to eigen
template<typename Derived,typename dType>
void copy_to_eigen(const Eigen::MatrixBase<Derived> &h_eigen_mat_const,dType *h_cuda_matrix) {
	UNCONST(Derived,h_eigen_mat_const,h_eigen_mat);
	for(int i=0; i<h_eigen_mat.rows(); i++) {
		for(int j=0; j<h_eigen_mat.cols(); j++) {
			h_eigen_mat(i,j) = h_cuda_matrix[IDX2C(i,j,h_eigen_mat.rows())];
		}
	}
}

//Copy a matrix in column major format to eigen
template<typename Derived,typename dType>
void copy_to_eigen_thrust(const Eigen::MatrixBase<Derived> &h_eigen_mat_const,
	thrust::host_vector<dType> &h_mat_thrust) 
{
	UNCONST(Derived,h_eigen_mat_const,h_eigen_mat);
	for(int i=0; i<h_eigen_mat.rows(); i++) {
		for(int j=0; j<h_eigen_mat.cols(); j++) {
			h_eigen_mat(i,j) = h_mat_thrust[IDX2C(i,j,h_eigen_mat.rows())];
		}
	}
}


//note there are no thrust matrices only vectors
template<typename dType>
void initialize_thrust_vector(thrust::host_vector<dType> &h_vec,int size) {
	boost::uniform_real<> distribution(BZ_CUDA::lower,BZ_CUDA::upper);
	for(int i=0; i<size; i++) {
		h_vec[i] = (dType)distribution(BZ_CUDA::gen);
	}
}

template<typename dType>
void print_GPU_Matrix(dType *d_ptr,int rows,int cols) {
	thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(d_ptr);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			BZ_CUDA::logger << debug_ptr[IDX2C(i,j,rows)] << " ";
		}
		BZ_CUDA::logger << "\n";
	}
	BZ_CUDA::logger << "\n";
}

template<typename dType>
void check_mem_loc(dType *d_ptr,int rows,int cols) {
	thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(d_ptr);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType temp = debug_ptr[IDX2C(i,j,rows)];
		}
	}
	CUDA_GET_LAST_ERROR("check_mem_loc");
}



//------------------------------------CUBLAS ERROR WRAPPERS--------------------------------------


///////////////////////////////////////////DOUBLE DEFINE BEGIN///////////////////////////////////////
inline cublasStatus_t cublas_gemm_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const float *alpha, const float *A, int lda, 
	 const float *B, int ldb, const float *beta, float *C, int ldc) 
{
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublas_gemm_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	 int m, int n, int k, const double *alpha, const double *A, int lda, 
	 const double *B, int ldb, const double *beta, double *C, int ldc) 
{
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
///////////////////////////////////////////DOUBLE DEFINE END///////////////////////////////////////


///////////////////////////////////////////DOUBLE DEFINE BEGIN///////////////////////////////////////
inline cublasStatus_t cublas_geam_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
	int m, int n, const float *alpha, const float *A, int lda, const float *beta, 
	const float *B, int ldb, float *C, int ldc) 
{
	return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);

}

inline cublasStatus_t cublas_geam_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, 
	int m, int n, const double *alpha, const double *A, int lda, const double *beta, 
	const double *B, int ldb, double *C, int ldc) 
{
	return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);

}
///////////////////////////////////////////DOUBLE DEFINE END///////////////////////////////////////


///////////////////////////////////////////DOUBLE DEFINE BEGIN///////////////////////////////////////
inline cublasStatus_t cublas_gemv_wrapper(cublasHandle_t handle, cublasOperation_t trans, int m, 
	int n, const float *alpha, const float *A, int lda, const float *x, int incx, 
	const float *beta, float *y, int incy) 
{
	return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t cublas_gemv_wrapper(cublasHandle_t handle, cublasOperation_t trans, int m, 
	int n, const double *alpha, const double *A, int lda, const double *x, int incx, 
	const double *beta, double *y, int incy) 
{
	return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
///////////////////////////////////////////DOUBLE DEFINE END///////////////////////////////////////



///////////////////////////////////////////DOUBLE DEFINE BEGIN///////////////////////////////////////
inline cublasStatus_t cublas_dgmm_wrapper(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, 
	const float *A, int lda, const float *x, int incx, float *C, int ldc)
{
	return cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

inline cublasStatus_t cublas_dgmm_wrapper(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, 
	const double *A, int lda, const double *x, int incx, double *C, int ldc)
{
	return cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

///////////////////////////////////////////DOUBLE DEFINE END///////////////////////////////////////


//atomic add for doubles,since undefined in cuda
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

//atomic add for doubles,since undefined in cuda
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


void curandGenerateUniform_wrapper(float *d_mask,int size,curandGenerator_t &generator) {
	curandGenerateUniform(generator,d_mask,size);
}

void curandGenerateUniform_wrapper(double *d_mask,int size,curandGenerator_t &generator) {
	curandGenerateUniformDouble(generator,d_mask,size);
}


__device__
inline float cuda_exp_wrapper(float x) {
	return expf(x);
}

__device__
inline double cuda_exp_wrapper(double x) {
	return exp(x);
}

__device__
inline float cuda_log_wrapper(float x) {
	return logf(x);
}

__device__
inline double cuda_log_wrapper(double x) {
	return log(x);
}


__device__
inline float pow_wrapper(float x,float y) {
	return powf(x,y);
}

__device__
inline double pow_wrapper(double x,double y) {
	return pow(x,y);
}


__device__
inline double tanh_wrapper(double x) {
	return tanh(x);
}


__device__
inline float tanh_wrapper(float x) {
	return tanhf(x);
}


__device__
inline bool nan_wrapper(float x) {
	return isnan((double)x);
}

__device__
inline bool nan_wrapper(double x) {
	return isnan(x);
}

__device__
inline bool nan_wrapper(int x) {
	return isnan((double)x);
}

__device__
inline bool isinf_wrapper(double x) {
	return isinf((float)x);
}

__device__
inline bool isinf_wrapper(float x) {
	return isinf(x);
}

__device__
inline double cuda_log1p_wrapper(double x) {
	return log1p(x);
}

__device__
inline float cuda_log1p_wrapper(float x) {
	return log1pf(x);
}

__device__
inline double cuda_max_wrapper(double x,double y) {
	return max(x,y);
}

__device__
inline float cuda_max_wrapper(float x,float y) {
	return fmaxf(x,y);
}



__device__
inline double cuda_min_wrapper(double x,double y) {
	return min(x,y);
}

__device__
inline float cuda_min_wrapper(float x,float y) {
	return fminf(x,y);
}


template<typename dType>
void get_cell_states(dType *d_ptr,int LSTM_size,int minibatch_size) {
	thrust::device_ptr<dType> debug_ptr = thrust::device_pointer_cast(d_ptr);
	int num_above_10 = 0;
	int num_below_10 = 0;
	int num_above_50 = 0;
	int num_below_50 = 0;
	int num_above_100 = 0;
	int num_below_100 = 0;
	int num_above_500 = 0;
	int num_below_500 = 0;

	for(int i=0; i<minibatch_size; i++) {
		for(int j=0; j< LSTM_size; j++) {
			dType val = debug_ptr[IDX2C(j,i,LSTM_size)];


			if(val>10) {
				num_above_10++;
			}
			if(val>50) {
				num_above_50++;
			}
			if(val>100) {
				num_above_100++;
			}
			if(val>500) {
				num_above_500++;
			}
			if(val<-10) {
				num_below_10++;
			}
			if(val<-50) {
				num_below_50++;
			}
			if(val<-100) {
				num_below_100++;
			}
			if(val<-500) {
				num_below_500++;
			}
		}
	}

	BZ_CUDA::logger << "CELL STATS\n";
	BZ_CUDA::logger << "Total cell states: " << LSTM_size*minibatch_size << "\n";
	BZ_CUDA::logger << "Num above 10: " << num_above_10 << "\n";
	BZ_CUDA::logger << "Num above 50: " << num_above_50 << "\n";
	BZ_CUDA::logger << "Num above 100: " << num_above_100 << "\n";
	BZ_CUDA::logger << "Num above 500: " << num_above_500 << "\n";
	BZ_CUDA::logger << "Num below -10: " << num_below_10 << "\n";
	BZ_CUDA::logger << "Num below -50: " << num_below_50 << "\n";
	BZ_CUDA::logger << "Num below -100: " << num_below_100 << "\n";
	BZ_CUDA::logger << "Num below -500: " << num_below_500 << "\n";
}

namespace gpu_info {
	std::vector<int> device_numbers;
}


//void devSynchAll() {
//	int origin_device;
//	cudaGetDevice(&origin_device);
//    std::cout << gpu_info::device_numbers.size() << "\n";
//	for(int i=0; i<gpu_info::device_numbers.size(); i++) {
//		cudaSetDevice(gpu_info::device_numbers[i]);
//		cudaDeviceSynchronize();
//	}
//	cudaSetDevice(origin_device);
//}


 void devSynchAll() {
 	int num_devices;
 	int origin_device;
 	cudaGetDevice(&origin_device);
 	cudaGetDeviceCount(&num_devices);
 	for(int i=0; i<num_devices; i++) {
 		cudaSetDevice(i);
 		cudaDeviceSynchronize();
 	}
 	cudaSetDevice(origin_device);
 }


template<typename dType>
__global__
void nan_check_kernel(dType *d_ptr,int rows,int cols,bool *d_check) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<rows*cols; i+=gridDim.x*blockDim.x) {
		if(nan_wrapper(d_ptr[i])) {
			d_check[0] = true;
		}
	}
}


bool *d_temp_bool=NULL;

template<typename dType>
bool check_nan(dType *d_ptr,int rows,int cols) {

	CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_temp_bool, 1*sizeof(bool)),"GPU memory allocation failed\n");
	thrust::device_ptr<bool> debug_ptr = thrust::device_pointer_cast(d_temp_bool);
	cudaMemset(d_temp_bool,0,1*sizeof(bool));

	nan_check_kernel<<<256,256>>>(d_ptr,rows,cols,d_temp_bool);
	devSynchAll();

	if(debug_ptr[0]) {
		BZ_CUDA::logger << "NAN check failed\n";
		return true;
	}

	cudaFree(d_temp_bool);
	return false;
}


template<typename dType>
__global__
void zero_check(dType *d_mat, int size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		assert(d_mat[i]==0);
	}
}


template<typename dType>
__global__
void check_nonseg(dType *d_ptr,int rows,int cols) {

	
}

void printIntroMessage(global_params &params) {

	if(params.train) {
		BZ_CUDA::logger << "\n\n------------------------Train Info------------------------\n";
		BZ_CUDA::logger << "Minibatch Size: " << params.minibatch_size << "\n";
		BZ_CUDA::logger << "Number of Epochs: " << params.num_epochs << "\n";
		BZ_CUDA::logger << "Learning Rate: " << params.learning_rate << "\n";
		if(params.clip_gradient) {
			BZ_CUDA::logger << "Gradient Clipping Threshold per matrix (Norm Ball): " << params.norm_clip << "\n";
		}
		if(params.individual_grad_clip) {
			BZ_CUDA::logger << "Gradient Clipping Threshold per element: " << params.ind_norm_clip_thres << "\n";
		}
		if(params.truncated_softmax) {
			BZ_CUDA::logger << "-------------------Truncated softmax info----------------------\n";
			BZ_CUDA::logger << "Shortlist Size: " << params.shortlist_size << "\n";
			BZ_CUDA::logger << "Sampled Size: " << params.sampled_size << "\n";
			BZ_CUDA::logger << "---------------------------------------------------------------\n\n";
		}
	}
	BZ_CUDA::logger << "------------------------Model Info------------------------\n";
	if(params.LM) {
		BZ_CUDA::logger << "Sequence model\n";
	}
	else {
		BZ_CUDA::logger << "Sequence to sequence model\n";
	}
	BZ_CUDA::logger << "Source Vocab Size: " << params.source_vocab_size << "\n";
	BZ_CUDA::logger << "Target Vocab Size: " << params.target_vocab_size << "\n";
	BZ_CUDA::logger << "Number of Hidden Units: " << params.LSTM_size << "\n";
	BZ_CUDA::logger << "Number of Layers: " << params.num_layers << "\n";
	if(params.attent_params.attention_model) {
		BZ_CUDA::logger << "Attention model set as true\n";
		BZ_CUDA::logger << "D = " << params.attent_params.D << "\n";
		if(params.attent_params.feed_input) {
			BZ_CUDA::logger << "Feed Input set as true\n";
		}
	}

	if(params.unk_replace) {
		BZ_CUDA::logger << "UNK replace is set to true\n";
	}
    if(params.NCE) {
        BZ_CUDA::logger << "Using NCE objective\n";
        BZ_CUDA::logger << "Number of noise samples for NCE: " << params.num_negative_samples << "\n";
    }
    else {
        BZ_CUDA::logger << "Using MLE objective\n";
    }

	BZ_CUDA::logger << "---------------------------------------------------------------\n\n";
	if(params.decode) {
		BZ_CUDA::logger << "------------------------Decode Info------------------------\n";
		BZ_CUDA::logger << "Beam size for kbest: " << params.beam_size << "\n";
		BZ_CUDA::logger << "Number of paths for kbest: " << params.num_hypotheses << "\n";
		BZ_CUDA::logger << "------------------------------------------------------------\n\n";
	}
	// if(stochastic_generation) {
	// 	BZ_CUDA::logger << "------------------------Stoch Generation Info------------------------\n";
	// 	BZ_CUDA::logger << "Number of tokens for stoch generation: " << sg_length << "\n";
	// 	BZ_CUDA::logger << "Stoch generation temperature: " << temperature << "\n";
	// 	BZ_CUDA::logger << "------------------------------------------------------------\n\n";
	// }

	//BZ_CUDA::logger << "Number of Lines in Training File: " << train_num_lines_in_file << "\n";
	BZ_CUDA::logger << "\n\n";
}






#endif
