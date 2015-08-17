//Custom Kernels
#ifndef CUSTOM_KERNELS_H
#define CUSTOM_KERNELS_H
#include <thrust/transform_reduce.h>
//------------------------------------Input data formatting kernels-----------------------------------------------

//transform vocab indices with -1's and numbers to all 0's and 1's
__global__ 
void vocab_to_01(int *d_vocab_indicies_01,int *d_vocab_indicies,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies[i]==-1) {
			d_vocab_indicies_01[i] = 0;
		}
		else {
			d_vocab_indicies_01[i] = 1;
		}
	}
}

//gets rid of all -1's and replaces them with index 0
__global__ 
void vocab_to_nonM1(int *d_vocab_indicies_nonM1,int *d_vocab_indicies,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies[i]==-1) {
			d_vocab_indicies_nonM1[i] = 0;
		}
		else {
			d_vocab_indicies_nonM1[i] = d_vocab_indicies[i];
		}
	}
}


//softmax kernel to preprocess data
template<typename dType>
__global__ 
void vocab_softmax(int *d_vocab_indicies,int *d_vocab_indicies_01,dType *d_vocab_indicies_01_float,int total_length) {
	for(int i= threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies[i]==-1) {
			d_vocab_indicies[i] = 0;
			d_vocab_indicies_01[i] = 0;
			d_vocab_indicies_01_float[i] = 0;
		}
		else {
			d_vocab_indicies_01[i] = 1;
			d_vocab_indicies_01_float[i] = 1;
		}
	}
}



//------------------------------------Forward prop kernels-------------------------------------------

///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__
void forward_sigmoid_kernel(float *d_final,float *temp1,float *temp2,float *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = 1.0f/(1.0f + expf(-1.0f*temp_val));
	}
}

__global__
void forward_sigmoid_kernel(double *d_final,double *temp1,double *temp2,double *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = 1.0/(1.0 + exp(-1.0*temp_val));
	}
}

///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////


///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__
void forward_tanh_kernel(float *d_final,float *temp1,float *temp2,float *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = tanhf(temp_val);
	}
}

__global__
void forward_tanh_kernel(double *d_final,double *temp1,double *temp2,double *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + temp2[index] + d_bias[idx];
		d_final[index] = tanh(temp_val);
	}
}
///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////


template<typename dType>
__global__
void forward_c_t_kernel(dType *d_c_t,dType *d_f_t, dType *d_c_t_prev,dType *d_i_t,dType *d_c_prime_t_tanh,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_c_t[index] = d_f_t[index] * d_c_t_prev[index] + d_i_t[index] * d_c_prime_t_tanh[index];
	}
}


///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__
void forward_h_t_kernel(float *d_h_t,float *d_o_t, float *d_c_t,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_h_t[index] = d_o_t[index] * tanhf(d_c_t[index]);
	}
}

__global__
void forward_h_t_kernel(double *d_h_t,double *d_o_t,double *d_c_t,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_h_t[index] = d_o_t[index] * tanh(d_c_t[index]);
	}
}

///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////


template<typename dType>
__global__ 
void zero_c_t_and_h_t(dType *d_h_t,dType *d_c_t,int *d_vocab_indices_01,int hiddenstate_size) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
  	int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_h_t[index] = d_h_t[index] * d_vocab_indices_01[blockIdx.x];
		d_c_t[index] = d_c_t[index] * d_vocab_indices_01[blockIdx.x];
	}
}













//-----------------------------------------backprop kernels-----------------------------------------


///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__ 
void d_ERRt_ct_kernel(float *d_d_ERRt_ct,float *d_d_ERRnTOt_ht,float *d_o_t,float *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
  		float val = tanhf(d_c_t[index]);
		d_d_ERRt_ct[index] = d_d_ERRnTOt_ht[index] * d_o_t[index] * (1.0f - val*val);
	}
}


__global__ 
void d_ERRt_ct_kernel(double *d_d_ERRt_ct,double *d_d_ERRnTOt_ht,double *d_o_t,double *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
  		double val = tanh(d_c_t[index]);
		d_d_ERRt_ct[index] = d_d_ERRnTOt_ht[index] * d_o_t[index] * (1.0f - val*val);
	}
}
///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////


///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__ 
void d_ERRnTOt_ot_kernel(float *d_d_ERRnTOt_ot,float *d_d_ERRnTOt_ht,float *d_o_t,float *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt_ot[index] = d_d_ERRnTOt_ht[index] *  tanhf(d_c_t[index]) * d_o_t[index] * (1 - d_o_t[index]);
	}
}


__global__ 
void d_ERRnTOt_ot_kernel(double *d_d_ERRnTOt_ot,double *d_d_ERRnTOt_ht,double *d_o_t,double *d_c_t,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt_ot[index] = d_d_ERRnTOt_ht[index] *  tanh(d_c_t[index]) * d_o_t[index] * (1 - d_o_t[index]);
	}
}

///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////

//For floats or doubles
template<typename dType>
__global__ 
void d_ERRnTOt_ft_it_kernel(dType *d_d_ERRnTOt,dType *d_d_ERRnTOt_ct,dType *d_single_err,dType *d_double_err,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt[index] = d_d_ERRnTOt_ct[index] * d_single_err[index] * d_double_err[index] * (1 - d_double_err[index]);
	}
}


template<typename dType>
__global__ 
void d_ERRnTOt_tanhcpt_kernel(dType *d_d_ERRnTOt_tanhcpt,dType *d_d_ERRnTOt_ct,dType *d_i_t,dType *d_c_prime_t_tanh,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_d_ERRnTOt_tanhcpt[index] = d_d_ERRnTOt_ct[index] * d_i_t[index] * (1 -d_c_prime_t_tanh[index]*d_c_prime_t_tanh[index]);
	}
}


template<typename dType>
__global__ 
void zero_columns_kernel(int hiddenstate_size, dType *d_mat,int *d_vec,dType *d_mat_final) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = \
		d_mat[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_vec[blockIdx.x];
	}
}


template<typename dType>
__global__ 
void add_four_matrices_kernel(dType *d_final,dType *d_mat1,dType *d_mat2,dType *d_mat3,dType *d_mat4,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < hiddenstate_size) {
  		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_final[index] = d_mat1[index] + d_mat2[index] + d_mat3[index] + d_mat4[index];
	}
}


template<typename dType>
__global__ 
void elementwise_mult_kernel(dType *d_mat1,dType *d_mat2,dType *d_final,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = d_mat1[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_mat2[IDX2C(idx,blockIdx.x,hiddenstate_size)];
	}
}


// template<typename dType>
// __global__ 
// void sparse_lookup_kernel(dType *d_lookup, dType *d_W,int *d_vocab_indices, int minibatch_size,int hiddenstate_size)
// {
// 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
// 	for(int idy = blockIdx.x; idy < minibatch_size; idy++) {
// 		if(idx < hiddenstate_size)
// 			d_lookup[IDX2C(idx,idy,hiddenstate_size)] = d_W[IDX2C(idx,d_vocab_indices[idy],hiddenstate_size)];
// 	}
// }

template<typename dType>
__global__ 
void sparse_lookup_kernel(dType *d_lookup, dType *d_W,int *d_vocab_indices, int minibatch_size,int hiddenstate_size)
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		d_lookup[IDX2C(idx,blockIdx.x,hiddenstate_size)] = d_W[IDX2C(idx,d_vocab_indices[blockIdx.x],hiddenstate_size)];
	}
}



///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__
void W_gradient_kernel(float *d_W_grad,int *d_vocab_indices,float *temp1,float *temp2,float *temp3,
	float *temp4,int hiddenstate_size) 
{
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index_cols = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float sum = temp1[index_cols] + temp2[index_cols] + temp3[index_cols] + temp4[index_cols];
		atomicAdd(&(d_W_grad[IDX2C(idx,d_vocab_indices[blockIdx.x],hiddenstate_size)]),sum);
	}
}


__global__
void W_gradient_kernel(double *d_W_grad,int *d_vocab_indices,double *temp1,double *temp2,double *temp3,
	double *temp4,int hiddenstate_size) 
{
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index_cols = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double sum = temp1[index_cols] + temp2[index_cols] + temp3[index_cols] + temp4[index_cols];
		atomicAddDouble(&(d_W_grad[IDX2C(idx,d_vocab_indices[blockIdx.x],hiddenstate_size)]),sum);
	}
}
///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////










//----------------------------------------------softmax kernels--------------------------------------

#define SOFTMAX_THREADS 256
#include <cfloat>

//for optimizing warps
//volatile must be used as register optimization will lead to wrong answers
template<typename dType>
__device__ 
void warpReduceSum(volatile dType* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template<typename dType>
__device__ 
void warpReduceMax(volatile dType* sdata, int tid) {
	sdata[tid] = (sdata[tid] > sdata[32 + tid]) ? sdata[tid] : sdata[32 + tid];
	sdata[tid] = (sdata[tid] > sdata[16 + tid]) ? sdata[tid] : sdata[16 + tid];
	sdata[tid] = (sdata[tid] > sdata[8 + tid]) ? sdata[tid] : sdata[8 + tid];
	sdata[tid] = (sdata[tid] > sdata[4 + tid]) ? sdata[tid] : sdata[4 + tid];
	sdata[tid] = (sdata[tid] > sdata[2 + tid]) ? sdata[tid] : sdata[2 + tid];
	sdata[tid] = (sdata[tid] > sdata[1 + tid]) ? sdata[tid] : sdata[1 + tid];
}


template<typename dType>
__global__
void train_perplexity_kernel(int *d_output_vocab_indices_single,int *d_output_vocab_indices_01_single,dType *d_outputdist,
	double *train_perplexity,int minibatch_size,int output_vocab_size) 
{
	for(int i= 0; i<minibatch_size; i++) {
		if(d_output_vocab_indices_01_single[i]==1) {
			train_perplexity[0]+= log( (double) d_outputdist[IDX2C(d_output_vocab_indices_single[i],i,output_vocab_size)]);
		}
	}
}



//This is called on the un-normalized distribution
//Note this is only called for float to deal with overflow issues with floats
/*
	-Each thread in a block gets a location in the buffer. Initially the max element is stored in this location
	-For buffer one extra slot is allocated to store the true max of the buffer
	-Each block does one outputdist column, so for a minibatch of 128, simply call this with dim = 20000 and blocks = 128
	-column major storage is necessary for this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	-adapted from torch
	-This does summing and exping all in one go, so no thrust or column of 1's needed
*/
template<typename dType>
__global__
void outputdist_overflow_prevention_kernel(dType *output, dType *input, int dim) {
	__shared__ dType buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	dType *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	//buffer[threadIdx.x] = -FLT_MAX;
	buffer[threadIdx.x] = -FLT_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		dType z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=SOFTMAX_THREADS/2; stride>0/*32*/; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	// if(tid<32) {
	// 	warpReduceMax(buffer,tid);
	// }

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	dType max_k = buffer[0];
	__syncthreads(); //THIS MUST BE HERE
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		dType z = cuda_exp_wrapper(input_k[i]-max_k);
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=SOFTMAX_THREADS/2; stride>0/*32*/; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	// if(tid<32) {
	// 	warpReduceSum(buffer,tid);
	// }

  	__syncthreads();

  	// normalize the softmax
	dType sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = output_k[i] / sum_k;
	}
}



template<typename dType>
__global__
void outputdist_perplexity_kernel(double *output, dType *input, int dim) {
	__shared__ double buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	double *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	buffer[threadIdx.x] = -DBL_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		double z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=SOFTMAX_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceMax(buffer,tid);
	}

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	double max_k = buffer[0];
	__syncthreads();
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		double z = cuda_exp_wrapper(input_k[i]-max_k);
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=SOFTMAX_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}

  	__syncthreads();

  	// normalize the softmax
	double sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = cuda_log_wrapper(output_k[i]) - cuda_log_wrapper(sum_k);
	}
}






template<typename dType>
__global__ 
void matrix_bias_kernel(int hiddenstate_size, dType *d_mat,dType *d_vec,dType *d_mat_final) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = \
	d_mat[IDX2C(idx,blockIdx.x,hiddenstate_size)] + d_vec[idx];
	}
}

struct exp_functor_gpu {
	__host__ __device__ void operator()(float &x) {
		x = expf(x);
	}
	__host__ __device__ void operator()(double &x) {
		x = exp(x);
	}
};

struct exp_functor_eigen {
	template<typename dType>
  dType operator() (dType x) const { return std::exp(x); }
};

//inverse each element in matrix
struct inv_functor_gpu {
	template<typename dType>
	__host__ __device__ void operator()(dType &x) {
		x = 1/x;
	}
};

template<typename dType>
__global__ 
void zero_columns_kernel_128(int hiddenstate_size, dType *d_mat,int *d_vec,dType *d_mat_final) 
{
 	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = \
	d_mat[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_vec[blockIdx.x];
	}
}


//This kernel adds a matrices rows to a matrices columns, which ones depend on the index
//hiddenstate_size refers to the number of rows in d_mat_final and also d_mat_col
template<typename dType>
__global__
void matrix_row_to_matrix_column_kernel(dType *d_mat_final,dType *d_mat_col,dType *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		d_mat_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] = d_mat_col[IDX2C(idx,blockIdx.x,hiddenstate_size)] + \
		d_mat_row[IDX2C(d_indices[blockIdx.x],idx,output_vocab_size)];
	}
}

//take binary matrix fo ints and convert it to floats
template<typename dType>
__global__ 
void copy_matrix_float_to_int_kernel(int *d_source,dType *d_destination,int size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  	if(idx < size) {
		d_destination[idx] = (dType)d_source[idx];
	}
}


//This kernel adds a matrices columns to a matrices rows, which ones depend on the index
//hiddenstate_size refers to the number of rows in d_mat_final and also d_mat_col
///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__
void matrix_column_to_matrix_row_kernel(float *d_mat_final,float *d_mat_col,float *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		atomicAdd(&d_mat_final[IDX2C(d_indices[blockIdx.x],idx,output_vocab_size)],d_mat_col[IDX2C(idx,blockIdx.x,hiddenstate_size)]);
	}
}

__global__
void matrix_column_to_matrix_row_kernel(double *d_mat_final,double *d_mat_col,double *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		atomicAddDouble(&d_mat_final[IDX2C(d_indices[blockIdx.x],idx,output_vocab_size)],d_mat_col[IDX2C(idx,blockIdx.x,hiddenstate_size)]);
	}
}
///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////


template<typename dType>
__global__
void matrix_column_to_matrix_row_kernel_2(dType *d_mat_final,dType *d_mat_col,dType *d_mat_row,int *d_indices,int hiddenstate_size,int output_vocab_size,int minibatch_size) {
	
	for(int i=0; i<minibatch_size; i++) {
		if(d_indices[i]==blockIdx.x) {
			int idx = threadIdx.x + blockIdx.y*blockDim.x;
			if(idx < hiddenstate_size) {
				d_mat_final[IDX2C(blockIdx.x,idx,output_vocab_size)] += d_mat_col[IDX2C(idx,i,hiddenstate_size)];
			}
		}
	}
}


//add ones to b_d bias unit
///////////////////////////////////////////DOUBLE DECLARATION BEGIN/////////////////////////////////////////////
__global__
void add_ones_b_d_grad(float *d_b_d_grad,int *d_output_vocab_indices_01,int *d_output_vocab_indices,int minibatch_size) {
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
	if(idx < minibatch_size && d_output_vocab_indices_01[idx]==1) {
		atomicAdd(&d_b_d_grad[d_output_vocab_indices[idx]],1);
	}
}

__global__
void add_ones_b_d_grad(double *d_b_d_grad,int *d_output_vocab_indices_01,int *d_output_vocab_indices,int minibatch_size) {
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
	if(idx < minibatch_size && d_output_vocab_indices_01[idx]==1) {
		atomicAddDouble(&d_b_d_grad[d_output_vocab_indices[idx]],1);
	}
}
///////////////////////////////////////////DOUBLE DECLARATION END/////////////////////////////////////////////



//-----------------------------------------updating parameters------------------------------------------

struct scale_functor {
	const int minibatch_size;

	scale_functor(int _minibatch_size) : minibatch_size(_minibatch_size) {}

	__host__ __device__ void operator()(float &x) {
		x = (1.0f/minibatch_size)*x;
	}
	__host__ __device__ void operator()(double &x) {
		x = (1.0/minibatch_size)*x;
	}
};


struct square {
    __host__ __device__
    float operator()(const float& x) const { 
        return x * x;
    }

    __host__ __device__
    double operator()(const double& x) const { 
        return x * x;
    }
};


template<typename dType>
struct re_scale_norm_functor {
	const dType norm_threshold;
	const dType norm;

	re_scale_norm_functor(dType _norm_threshold,dType _norm) : norm_threshold(_norm_threshold),norm(_norm) {}

	__host__ __device__ void operator()(dType &x) {
		x = (norm_threshold/norm)*x;
	}
};


#define NORM_THREADS 256
template<typename dType>
__global__
void basic_compute_norm_p1(dType *d_gradient,int size,dType *result) {
	__shared__ dType buffer[NORM_THREADS];
	int i_start = threadIdx.x+blockIdx.x*blockDim.x; //start at the thread index
	int i_end = size; //end at dim
	int i_step = blockDim.x*gridDim.x; //the block dimension (aka the number of threads in the block) is the step
	int tid = threadIdx.x;


	buffer[tid] = 0;
	for(int i= i_start; i<i_end; i+=i_step) {
		buffer[tid]+=(d_gradient[i]*d_gradient[i]);
	}
	__syncthreads();

	for(int stride=NORM_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}
	__syncthreads();

	if(tid==0) {
		result[blockIdx.x]=buffer[0];
	}
}


template<typename dType>
__global__
void basic_compute_norm_p2(dType *temp_result,dType *final_result) {
	__shared__ dType buffer[NORM_THREADS];

	int tid = threadIdx.x;
	buffer[tid] = temp_result[tid];
	__syncthreads();

	for(int stride=NORM_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}
	__syncthreads();

	if(tid==0) {
		final_result[0]=buffer[0];
	}
}



//clip the norm if it is greater than the threshold
template<typename dType>
void norm_clip_GPU(thrust::device_ptr<dType> &thrust_d_gradient,dType norm_threshold,int size) {
	dType norm = std::sqrt( thrust::transform_reduce(thrust_d_gradient, 
		thrust_d_gradient+size, square(), (dType)0, thrust::plus<dType>()) );
	if(norm > norm_threshold) {
		//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
		re_scale_norm_functor<dType> unary_op(norm_threshold,norm);
		thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	}
}

//clip the norm if it is greater than the threshold
template<typename dType>
void norm_clip_GPU_v2(thrust::device_ptr<dType> &thrust_d_gradient,dType *d_gradient,dType norm_threshold,int size,dType *d_temp_result,dType *d_result) {
	dType norm;
	basic_compute_norm_p1<<<NORM_THREADS,NORM_THREADS>>>(d_gradient,size,d_temp_result);
	basic_compute_norm_p2<<<1,NORM_THREADS>>>(d_temp_result,d_result);
	cudaMemcpy(&norm,d_result,1*sizeof(dType),cudaMemcpyDeviceToHost);
	norm = std::sqrt(norm);
	if(norm > norm_threshold) {
		//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
		re_scale_norm_functor<dType> unary_op(norm_threshold,norm);
		thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	}
}

//Kernel for getting scaling the gradient of W by 1/(minibatch size)
template<typename dType>
__global__
void scale_W_gradient(dType *d_W_gradient,int *d_vocab_indicies_m1,int hiddenstate_size,dType scale,int total_length) {
	for(int j=blockIdx.y; j<total_length; j+=gridDim.y) {
		const int idx = threadIdx.x + blockIdx.x*blockDim.x;
		if(idx < hiddenstate_size) {
			const int index = IDX2C(idx,d_vocab_indicies_m1[j],hiddenstate_size);
			d_W_gradient[index] = scale * d_W_gradient[index];
		}
	}
}


//compute l2 norm of W
template<typename dType>
__global__
void norm_W_compute_p1(dType *d_W_gradient,dType *global_tempsum,int *d_vocab_indicies,int hiddenstate_size,int total_length) {

	__shared__ dType buffer[NORM_THREADS];

	int i_start = threadIdx.x; //start at the thread index
	int i_end = hiddenstate_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	int tid = threadIdx.x;
	int j_start = blockIdx.x;
	int j_end = total_length;
	int j_step = gridDim.x;
	int bid = blockIdx.x;

	if(tid ==0) {
		global_tempsum[bid]=0;
	}

	for(int j= j_start; j<j_end; j+=j_step) {
		buffer[tid] = 0;
		for(int i= i_start; i<i_end; i+=i_step) {
			buffer[tid]+=(d_W_gradient[IDX2C(i,d_vocab_indicies[j],hiddenstate_size)]*d_W_gradient[IDX2C(i,d_vocab_indicies[j],hiddenstate_size)]);
		}
		__syncthreads();

		for(int stride=NORM_THREADS/2; stride>32; stride>>=1) {
			if(tid < stride) {
				buffer[tid] += buffer[stride + tid];
			}
			__syncthreads();
		}

		if(tid<32) {
			warpReduceSum(buffer,tid);
		}
		__syncthreads();
		if(tid ==0) {
			global_tempsum[bid]+=buffer[0];
		}
		__syncthreads();
	}
}

//compute l2 norm of W
//NOTE this should be launched with only 1 block
template<typename dType>
__global__
void norm_W_compute_p2(dType *global_tempsum) {
	__shared__ dType buffer[NORM_THREADS];
	int tid = threadIdx.x;

	buffer[tid] = global_tempsum[tid];

	__syncthreads();

	for(int stride=NORM_THREADS/2; stride>32; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	if(tid<32) {
		warpReduceSum(buffer,tid);
	}
	__syncthreads();

	if(tid==0) {
		global_tempsum[0] = buffer[0];
	}

	//Now in the zeroth spot is the sum(so memcpy it)
}


template<typename dType>
void norm_clip_W_GPU(thrust::device_ptr<dType> &thrust_d_gradient,dType * d_grad,
	int *d_vocab_indicies_m1 ,dType norm_threshold,int total_length,int hiddenstate_size,int size) 
{
	dType norm = std::sqrt( thrust::transform_reduce(thrust_d_gradient, 
		thrust_d_gradient+size, square(), (dType)0, thrust::plus<dType>()) );
	if(norm > norm_threshold) {
		//std::cout << "----ACTUALLY CLIPPING W NORM----\n";
		int threads_per_block = 256;
		int num_block = (hiddenstate_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		dType scalar = (norm_threshold/norm);
		scale_W_gradient<<<kernel,threads_per_block>>>(d_grad,d_vocab_indicies_m1,hiddenstate_size,scalar,total_length);
	}
}

//v2 with custom W gradient clipping
template<typename dType>
void norm_clip_W_GPU_v2(dType *d_global_W_sum,dType * d_grad,
	int *d_vocab_indicies_m1 ,dType norm_threshold,int total_length,int hiddenstate_size) 
{
	dType norm;
	norm_W_compute_p1<<<NORM_THREADS,NORM_THREADS>>>(d_grad,d_global_W_sum,d_vocab_indicies_m1,hiddenstate_size,total_length);
	norm_W_compute_p2<<<1,NORM_THREADS>>>(d_global_W_sum);
	cudaMemcpy(&norm,d_global_W_sum,1*sizeof(dType),cudaMemcpyDeviceToHost);
	norm = std::sqrt(norm);
	//std::cout << "NORM OF W: " << norm << "\n";

	if(norm > norm_threshold) {
		//std::cout << "----ACTUALLY CLIPPING W NORM----\n";
		int threads_per_block = 256;
		int num_block = (hiddenstate_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		dType scalar = (norm_threshold/norm);
		scale_W_gradient<<<kernel,threads_per_block>>>(d_grad,d_vocab_indicies_m1,hiddenstate_size,scalar,total_length);
	}
}


//Kernel for zeroing the W gradient
//length the special length for W grad
template<typename dType>
__global__ 
void zero_W_gradient(dType *d_W_gradient,int *d_vocab_indicies_m1,int hiddenstate_size,int total_length) {
	for(int j=blockIdx.y; j<total_length; j+=gridDim.y) {
		const int idx = threadIdx.x + blockIdx.x*blockDim.x;
		if(idx < hiddenstate_size) {
			d_W_gradient[IDX2C(idx,d_vocab_indicies_m1[j],hiddenstate_size)] = 0;
		}
	}
}

//Kernel for updating the W gradient
template<typename dType>
__global__ 
void update_W_gradient(dType *d_W, dType *d_W_gradient,int *d_vocab_indicies_m1,dType learning_rate,int hiddenstate_size,int total_length) {
	for(int j = blockIdx.y; j<total_length; j+=gridDim.y) {
		int idx = threadIdx.x + blockIdx.x*blockDim.x;
		if(idx < hiddenstate_size) {
			int index = IDX2C(idx,d_vocab_indicies_m1[j],hiddenstate_size);
			d_W[index] = learning_rate* d_W_gradient[index] + d_W[index];
		}
	}
}

template<typename dType>
__global__
void add_grad_vecs(dType *vec1,dType *vec2,dType learning_rate,int size) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < size) {
		vec1[idx] = learning_rate*vec2[idx] + vec1[idx];
	}

}


//-------------------------------------------------Decoder Stuff----------------------------------------

template<typename dType>
__global__
void ones_mat(dType *mat,int size) {
	for(int i= threadIdx.x; i<size; i+=blockDim.x) {
		mat[i] = 1;
	}
}



//-------------------------------------------------stuff for truncated softmax----------------------------------------

//called when updating parameters
template<typename dType>
__global__
void trunc_D_grad_nonshort(dType *d_subset_D_grad,dType *d_D,int *d_vocab_mappings,int hiddenstate_size,int trunc_size,int output_vocab_size,dType learning_rate,int shortlist_size) {

	for(int j = blockIdx.x+shortlist_size; j < trunc_size; j += gridDim.x) {
		for(int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
			d_D[IDX2C(d_vocab_mappings[j-shortlist_size],i,output_vocab_size)]+= learning_rate*d_subset_D_grad[IDX2C(j,i,trunc_size)];
		}
	}
}


template<typename dType>
__global__
void trunc_D_grad_short(dType *d_subset_D_grad,dType *d_subset_D,int hiddenstate_size,int shortlist_size,dType learning_rate,int trunc_size) {

	for(int j = blockIdx.x; j < shortlist_size; j += gridDim.x) {
		for(int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
			d_subset_D[IDX2C(j,i,trunc_size)]+= learning_rate*d_subset_D_grad[IDX2C(j,i,trunc_size)];
		}
	}
}

//called when beginning minibatch
template<typename dType>
__global__
void trunc_set_D(dType *d_D,dType *d_subset_D,int trunc_size,int output_vocab_size,int shortlist_size,int *d_vocab_mappings,int hiddenstate_size) {
	for(int j = blockIdx.x+shortlist_size; j < trunc_size; j += gridDim.x) {
		for(int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
			d_subset_D[IDX2C(j,i,trunc_size)] = d_D[IDX2C(d_vocab_mappings[j-shortlist_size],i,output_vocab_size)];
		}
	}
}


//for multiply outputdist by the sample rate correction
//shortlist size plus is the size of shortlist plus the unique words in the minibatch
//put an error check for this on the CPU
// template<typename dType>
// __global__
// void scale_truncated_softmax(dType *d_subset_outputdist,dType sample_rate,int shortlist_size_plus,int trunc_size,int minibatch_size) {
// 	for(int j = blockIdx.x+shortlist_size_plus; j < trunc_size; j += gridDim.x) {
// 		for(int i = threadIdx.x; i < minibatch_size; i += blockDim.x) {
// 			d_subset_outputdist[IDX2C(j,i,trunc_size)]= sample_rate*d_subset_outputdist[IDX2C(j,i,trunc_size)];
// 		}
// 	}
// }

//called when finished training before parameters are written to a file
template<typename dType>
__global__
void load_shortlist_D(dType *d_subset_D,dType *d_D,int hiddenstate_size,int trunc_size,int output_vocab_size,int shortlist_size) {
	for(int j = blockIdx.x; j < shortlist_size; j += gridDim.x) {
		for(int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
			d_D[IDX2C(j,i,output_vocab_size)]= d_subset_D[IDX2C(j,i,trunc_size)];
		}
	}
}


//scales before normalization stage
//call in place of overflow kernel
template<typename dType>
__global__
void outputdist_truncated_kernel(dType *output, dType *input, int dim,dType sample_rate,int shortlist_size_plus) {
	__shared__ dType buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	dType *output_k = output + k*dim; //again all threads in block start from same index

	int i_start = threadIdx.x; //start at the thread index
	int i_end = dim; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	//get the max element for each thread's assigned locations and put them in the buffer
	//dim elements are covered in this reduction
	//buffer[threadIdx.x] = -FLT_MAX;
	buffer[threadIdx.x] = -FLT_MAX;
	for(int i=i_start; i<i_end; i+=i_step) {
		dType z = input_k[i];
		if(buffer[threadIdx.x] < z) {
			buffer[threadIdx.x] = z;
		}
	}

	 __syncthreads();

	 // reduce
	 //first thread goes through and finds the max element in the buffer
	 //after this stage the max element for dim items is found
	for(int stride=SOFTMAX_THREADS/2; stride>0/*32*/; stride>>=1) {
		if(tid < stride) {
			buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
		}
		__syncthreads();
	}

	// if(tid<32) {
	// 	warpReduceMax(buffer,tid);
	// }

	__syncthreads();

	// sum
	//Now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
	dType max_k = buffer[0];
	__syncthreads(); //THIS MUST BE HERE
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		dType z;
		if(i>=shortlist_size_plus) {
			z = sample_rate*cuda_exp_wrapper(input_k[i]-max_k);
		}
		else
		{	
			z = cuda_exp_wrapper(input_k[i]-max_k);
		}
		buffer[threadIdx.x] += z; //keep a running sum of these values for the normalization constant
		output_k[i] = z; //set the output as this value, then get ready to divide by the sum again
	}

 	__syncthreads();

 	// reduce
 	//Now sum all the elements in the buffer, for the normalization constant
 	for(int stride=SOFTMAX_THREADS/2; stride>0/*32*/; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	// if(tid<32) {
	// 	warpReduceSum(buffer,tid);
	// }

  	__syncthreads();

  	// normalize the softmax
	dType sum_k = buffer[0];
	for (int i=i_start; i<i_end; i+=i_step) {
		output_k[i] = output_k[i] / sum_k;
	}
}

#endif

