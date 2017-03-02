//Custom Kernels
#ifndef CUSTOM_KERNELS_H
#define CUSTOM_KERNELS_H
#include <thrust/transform_reduce.h>
#include <assert.h>
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


template<typename dType>
__global__
void forward_sigmoid_kernel_small(dType *d_final,dType *temp1,dType *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double temp_val = temp1[index] + d_bias[idx];
		d_final[index] = 1.0/(1.0 + exp(-1.0*temp_val));
	}
}


template<typename dType>
__global__
void forward_sigmoid_kernel_feed(dType *d_final,dType *temp1,dType *temp2,dType *temp3,dType *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		dType temp_val = temp1[index] + temp2[index] + temp3[index] +d_bias[idx];
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


template<typename dType>
__global__
void forward_tanh_kernel_feed(dType *d_final,dType *temp1,dType *temp2,dType *temp3,dType *d_bias,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		dType temp_val = temp1[index] + temp2[index] + temp3[index] + d_bias[idx];
		d_final[index] = tanh_wrapper(temp_val);
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

template<typename dType>
__global__
void forward_c_t_kernel_tree(dType *d_c_t,dType *d_f_t_1,dType *d_c_t_prev_1,dType *d_f_t_2,dType *d_c_t_prev_2,dType *d_i_t,dType *d_c_prime_t_tanh,int hiddenstate_size) {
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index = IDX2C(idx,blockIdx.x,hiddenstate_size);
		d_c_t[index] = d_f_t_1[index] * d_c_t_prev_1[index] + d_f_t_2[index] * d_c_t_prev_2[index] + d_i_t[index] * d_c_prime_t_tanh[index];
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

template<typename dType>
__global__ 
void elementwise_mult_kernel_add(dType *d_mat1,dType *d_mat2,dType *d_final,int hiddenstate_size) 
{
	int idx = threadIdx.x+blockIdx.y*blockDim.x;
  if(idx < hiddenstate_size) {
		d_final[IDX2C(idx,blockIdx.x,hiddenstate_size)] += d_mat1[IDX2C(idx,blockIdx.x,hiddenstate_size)] * d_mat2[IDX2C(idx,blockIdx.x,hiddenstate_size)];
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



__global__
void W_gradient_kernel_dropout(float *d_W_grad,int *d_vocab_indices,float *temp1,float *temp2,float *temp3,
	float *temp4,int hiddenstate_size,float *d_dropout_mask,float rate) 
{
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index_cols = IDX2C(idx,blockIdx.x,hiddenstate_size);
		float sum = (temp1[index_cols] + temp2[index_cols] + temp3[index_cols] + temp4[index_cols])*(rate > d_dropout_mask[index_cols]) * (1/rate);
		atomicAdd(&(d_W_grad[IDX2C(idx,d_vocab_indices[blockIdx.x],hiddenstate_size)]),sum);
	}
}


__global__
void W_gradient_kernel_dropout(double *d_W_grad,int *d_vocab_indices,double *temp1,double *temp2,double *temp3,
	double *temp4,int hiddenstate_size,double *d_dropout_mask,double rate) 
{
	int idx = threadIdx.x + blockIdx.y*blockDim.x;
	if(idx < hiddenstate_size) {
		int index_cols = IDX2C(idx,blockIdx.x,hiddenstate_size);
		double sum = (temp1[index_cols] + temp2[index_cols] + temp3[index_cols] + temp4[index_cols])*(rate > d_dropout_mask[index_cols]) * (1/rate);
		atomicAddDouble(&(d_W_grad[IDX2C(idx,d_vocab_indices[blockIdx.x],hiddenstate_size)]),sum);
	}
}


template<typename dType> 
__global__
void W_small_gradient_kernel(dType *d_small_W_grad,int *d_reverse_unique_indicies,dType *temp1,dType *temp2,dType *temp3,
	dType *temp4,int *d_vocab_indicies,int LSTM_size,int minibatch_size) 
{	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < minibatch_size; k+=gridDim.x) {
		int vocab_index = d_vocab_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			dType sum = temp1[IDX2C(i,k,LSTM_size)] + temp2[IDX2C(i,k,LSTM_size)] + temp3[IDX2C(i,k,LSTM_size)] + temp4[IDX2C(i,k,LSTM_size)];
			atomicAdd(&(d_small_W_grad[IDX2C(i,d_reverse_unique_indicies[vocab_index],LSTM_size)]),sum);
		}
	}
}


template<typename dType>
__global__
void W_small_dropout_gradient_kernel(dType *d_small_W_grad,int *d_reverse_unique_indicies,dType *temp1,dType *temp2,dType *temp3,
	dType *temp4,int *d_vocab_indicies,int LSTM_size,int minibatch_size,dType *d_dropout_mask,dType rate) 
{	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < minibatch_size; k+=gridDim.x) {
		int vocab_index = d_vocab_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			dType sum = temp1[IDX2C(i,k,LSTM_size)] + temp2[IDX2C(i,k,LSTM_size)] + temp3[IDX2C(i,k,LSTM_size)] + temp4[IDX2C(i,k,LSTM_size)];
			sum = sum*(rate > d_dropout_mask[IDX2C(i,k,LSTM_size)])*(1/rate);
			atomicAdd(&(d_small_W_grad[IDX2C(i,d_reverse_unique_indicies[vocab_index],LSTM_size)]),sum);
		}
	}
}



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



template<typename dType,typename dType2>
__global__
void outputdist_perplexity_kernel(dType2 *output, dType *input, int dim,bool print_partition_function,double *d_partition_vals) {
	__shared__ double buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	dType2 *output_k = output + k*dim; //again all threads in block start from same index

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

	if(print_partition_function && threadIdx.x == 0) {
		d_partition_vals[blockIdx.x] = sum_k;
	}
}


template<typename dType,typename dType2>
__global__
void outputdist_perplexity_kernel_NCE(dType2 *output, dType *input, int dim,bool print_partition_function,double *d_partition_vals) {
	__shared__ double buffer[SOFTMAX_THREADS]; //shared memory for the block, this must be the number of threads per block in size
	int k = blockIdx.x; //get the block index
	dType *input_k = input + k*dim; //all threads in block start from same index
	dType2 *output_k = output + k*dim; //again all threads in block start from same index

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
	//double max_k = buffer[0];
	__syncthreads();
	buffer[threadIdx.x] = 0;
	for (int i=i_start; i<i_end; i+=i_step) {
		//dType z = exp(input_k[i]-max_k); //subtract the max from the input, then exp it for the softmax
		double z = cuda_exp_wrapper(input_k[i]);
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

	if(print_partition_function && threadIdx.x == 0) {
		d_partition_vals[blockIdx.x] = sum_k;
	}
}


//for re-scoring
template<typename dType>
__global__
void nce_score_dot(double *d_outputdist_perp,dType *d_h_t,dType *d_D,dType *d_b_d,int *d_vocab_indicies,int LSTM_size,int minibatch_size,int output_vocab_size) {

	__shared__ double buffer[SOFTMAX_THREADS];
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	int tid = threadIdx.x;

	for(int k = blockIdx.x; k < minibatch_size; k+=gridDim.x) {

		int vocab_index = d_vocab_indicies[k];
		buffer[tid] = 0;

		__syncthreads();

		for(int i= i_start; i < i_end; i += i_step) {
			buffer[tid] += d_D[IDX2C(vocab_index,i,output_vocab_size)] * d_h_t[IDX2C(i,k,LSTM_size)];
		}

		__syncthreads();

		for(int stride=SOFTMAX_THREADS/2; stride>0; stride>>=1) {
			if(tid < stride) {
				buffer[tid] += buffer[stride + tid];
			}
			__syncthreads();
		}

		if(threadIdx.x==0) {
			d_outputdist_perp[IDX2C(vocab_index,k,output_vocab_size)] = buffer[0] + d_b_d[vocab_index];
		}
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




template<typename dType>
__global__
void print_norm_function_softmax(dType *d_mat,int size,int index,dType *d_error) {
	__shared__ dType buffer[NORM_THREADS];
	int i_start = threadIdx.x; //start at the thread index
	int i_end = size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	int tid = threadIdx.x;

	buffer[tid] = 0;
	for(int i= i_start; i<i_end; i+=i_step) {
		buffer[tid]+=(d_mat[i]*d_mat[i]);
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
		d_error[index] = buffer[0];
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
	BZ_CUDA::recent_sum = norm;
	norm = std::sqrt(norm);
	if(norm > norm_threshold) {
		//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
		re_scale_norm_functor<dType> unary_op(norm_threshold,norm);
		thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	}
}



//for global clipping
template<typename dType>
void norm_clip_GPU_v2_p1(thrust::device_ptr<dType> &thrust_d_gradient,dType *d_gradient,dType norm_threshold,int size,dType *d_temp_result,dType *d_result) {
	dType norm;
	basic_compute_norm_p1<<<NORM_THREADS,NORM_THREADS>>>(d_gradient,size,d_temp_result);
	basic_compute_norm_p2<<<1,NORM_THREADS>>>(d_temp_result,d_result);
	devSynchAll();
	cudaMemcpy(&norm,d_result,1*sizeof(dType),cudaMemcpyDeviceToHost);
	//norm = std::sqrt(norm);
	BZ_CUDA::global_norm += norm;
	BZ_CUDA::recent_sum = norm;
	// if(norm > norm_threshold) {
	// 	//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
	// 	re_scale_norm_functor<dType> unary_op(norm_threshold,norm);
	// 	thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	// }
}


//for global clipping
template<typename dType>
void norm_clip_GPU_v2_p1_DEBUG(dType *d_gradient,int size,dType *d_temp_result,dType *d_result) {
	dType norm;
	basic_compute_norm_p1<<<NORM_THREADS,NORM_THREADS>>>(d_gradient,size,d_temp_result);
	basic_compute_norm_p2<<<1,NORM_THREADS>>>(d_temp_result,d_result);
	devSynchAll();
	cudaMemcpy(&norm,d_result,1*sizeof(dType),cudaMemcpyDeviceToHost);
	//norm = std::sqrt(norm);
	BZ_CUDA::recent_sum = norm;
	// if(norm > norm_threshold) {
	// 	//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
	// 	re_scale_norm_functor<dType> unary_op(norm_threshold,norm);
	// 	thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	// }
}



template<typename dType>
void norm_clip_GPU_v2_p2(thrust::device_ptr<dType> &thrust_d_gradient,dType *d_gradient,dType norm_threshold,int size,dType *d_temp_result,dType *d_result) {
	// dType norm;
	// basic_compute_norm_p1<<<NORM_THREADS,NORM_THREADS>>>(d_gradient,size,d_temp_result);
	// basic_compute_norm_p2<<<1,NORM_THREADS>>>(d_temp_result,d_result);
	// devSyncAll();
	// cudaMemcpy(&norm,d_result,1*sizeof(dType),cudaMemcpyDeviceToHost);
	// norm = std::sqrt(norm);
	//BZ_CUDA::global_norm += norm;
	if(BZ_CUDA::global_norm > norm_threshold) {
		//std::cout << "ACTUALLY NORM CLIPPING REGULAR PARAM\n";
		re_scale_norm_functor<dType> unary_op(norm_threshold,BZ_CUDA::global_norm);
		thrust::for_each(thrust_d_gradient,thrust_d_gradient+size,unary_op);
	}
}


//additional gradient clipping stuff
template<typename dType>
__global__
void clip_individual(dType *d_mat, int size,dType threshold) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat[i] = d_mat[i] > threshold ? threshold : d_mat[i];
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



template<typename dType>
__global__
void indv_clip_W_gradient(dType *d_W_gradient,int *d_vocab_indicies_m1,int hiddenstate_size,dType threshold,int total_length) {
	for(int j=blockIdx.y; j<total_length; j+=gridDim.y) {
		const int idx = threadIdx.x + blockIdx.x*blockDim.x;
		if(idx < hiddenstate_size) {
			const int index = IDX2C(idx,d_vocab_indicies_m1[j],hiddenstate_size);
			if(d_W_gradient[index] > 0) {
				d_W_gradient[index] = (d_W_gradient[index] > threshold) ? threshold : d_W_gradient[index];
			}
			else {
				d_W_gradient[index] = (d_W_gradient[index] < -threshold) ? -threshold : d_W_gradient[index];
			}
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

//v2 with custom W gradient clipping
template<typename dType>
void norm_clip_W_GPU_v2_p1(dType *d_global_W_sum,dType * d_grad,
	int *d_vocab_indicies_m1 ,dType norm_threshold,int total_length,int hiddenstate_size) 
{
	devSynchAll();
	dType norm;
	norm_W_compute_p1<<<NORM_THREADS,NORM_THREADS>>>(d_grad,d_global_W_sum,d_vocab_indicies_m1,hiddenstate_size,total_length);
	norm_W_compute_p2<<<1,NORM_THREADS>>>(d_global_W_sum);
	devSynchAll();
	cudaMemcpy(&norm,d_global_W_sum,1*sizeof(dType),cudaMemcpyDeviceToHost);
	norm = std::sqrt(norm);
	//std::cout << "NORM OF W: " << norm << "\n";
	BZ_CUDA::global_norm += norm;
	// if(norm > norm_threshold) {
	// 	//std::cout << "----ACTUALLY CLIPPING W NORM----\n";
	// 	int threads_per_block = 256;
	// 	int num_block = (hiddenstate_size + threads_per_block-1)/threads_per_block;
	// 	dim3 kernel(num_block,256,1);
	// 	dType scalar = (norm_threshold/norm);
	// 	scale_W_gradient<<<kernel,threads_per_block>>>(d_grad,d_vocab_indicies_m1,hiddenstate_size,scalar,total_length);
	// }
}

//v2 with custom W gradient clipping
template<typename dType>
void norm_clip_W_GPU_v2_p2(dType *d_global_W_sum,dType * d_grad,
	int *d_vocab_indicies_m1 ,dType norm_threshold,int total_length,int hiddenstate_size) 
{
	// dType norm;
	// norm_W_compute_p1<<<NORM_THREADS,NORM_THREADS>>>(d_grad,d_global_W_sum,d_vocab_indicies_m1,hiddenstate_size,total_length);
	// norm_W_compute_p2<<<1,NORM_THREADS>>>(d_global_W_sum);
	// cudaMemcpy(&norm,d_global_W_sum,1*sizeof(dType),cudaMemcpyDeviceToHost);
	// norm = std::sqrt(norm);
	//std::cout << "NORM OF W: " << norm << "\n";
	devSynchAll();
	if(BZ_CUDA::global_norm > norm_threshold) {
		//std::cout << "----ACTUALLY CLIPPING W NORM----\n";
		int threads_per_block = 256;
		int num_block = (hiddenstate_size + threads_per_block-1)/threads_per_block;
		dim3 kernel(num_block,256,1);
		dType scalar = (norm_threshold/BZ_CUDA::global_norm);
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









//-------------------------------------------------Dropout Stuff----------------------------------------

//for forward and backward pass for error and h_t in LSTM

template<typename dType>
__global__
void dropout_kernel(dType *d_dropout_mask,dType rate,dType *d_final, int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		d_final[i] = (d_dropout_mask[i] < rate) * (1/rate) * d_final[i];
	}
}


//-------------------------------------------------Attention model----------------------------------------


__global__
void tanh_kernel(float *d_in,float *d_out,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		d_out[i] = tanhf(d_in[i]);
	}
}

__global__
void tanh_kernel(double *d_in,double *d_out,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		d_out[i] = tanh(d_in[i]);
	}
}


template<typename dType>
__global__
void sigmoid_kernel(dType *d_in,dType *d_out,int total_length) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		d_out[i] = 1.0/(1.0 + cuda_exp_wrapper(-1.0*d_in[i]));
	}
}


/*
	Batch info is in the form

	[sent lens][offsets]

*/
template<typename dType>
__global__
void alignment_pos_kernel(dType *d_in,dType *d_out,int total_length,int *d_batch_info) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<total_length; i+=gridDim.x*blockDim.x) {
		d_out[i] =  d_batch_info[i]*d_in[i];
		//printf("d_batch_info from kernel %d %f\n",d_batch_info[i],d_out[i]);
	}
}


template<typename dType>
__global__
void lower_upper_kernel(dType *d_p_t,int *d_lower_upper,int D,int *d_batch_info,int minibatch_size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<minibatch_size; i+=gridDim.x*blockDim.x) {
		d_lower_upper[IDX2C(0,i,2)] = ( 0 > (int)(d_p_t[i])-D ) ? 0 : ((int)(d_p_t[i])-D);
		d_lower_upper[IDX2C(1,i,2)] = ( (d_batch_info[i]-1) < (int)(d_p_t[i])+D ) ? (d_batch_info[i]-1) : ((int)(d_p_t[i])+D);
	}
}



/*
	For getting viterbi alignments
*/

template<typename dType>
__global__
void get_viterbi_alignment_kernel(dType *d_alignments,int *d_indicies,int D,int minibatch_size,int *d_final_results) {

	int minibatch_index = threadIdx.x;
	dType max_val = -1;
	int max_index = -1;
	for(int i=0; i<2*D+1; i++) {
		if(max_val < d_alignments[IDX2C(minibatch_index,i,minibatch_size)]) {
			max_val = d_alignments[IDX2C(minibatch_index,i,minibatch_size)];
			max_index = d_indicies[IDX2C(minibatch_index,i,minibatch_size)];
		}
	}
	d_final_results[minibatch_index] = max_index;
}


/*
	each thread will initialize 2*D + 1 indicies
	pads from the back

	layout is as follows:
	
	[minibatch] [minibatch] [...]
	There are 2*D + 1 of these minibatch chunks. This how h_s is loaded in so the format is useful

*/

__global__
void create_indicies_kernel(int *d_indicies,int D,int minibatch_size,int *d_lower_upper,int *d_01_mask) {

	for(int i=threadIdx.x; i < minibatch_size; i += blockDim.x) {
		int curr_index = d_lower_upper[IDX2C(0,i,2)];
		int max_index = d_lower_upper[IDX2C(1,i,2)];
		if(d_01_mask[i]==1) {
			for(int j = 0; j < 2*D+1; j++) {

				if(curr_index > max_index) {
					d_indicies[IDX2C(i,j,minibatch_size)] = -1;
				}
				else {
					d_indicies[IDX2C(i,j,minibatch_size)] = curr_index;
				}
				curr_index++;
			}
		}
		else {
			for(int j = 0; j < 2*D+1; j++) {
				d_indicies[IDX2C(i,j,minibatch_size)] = -1;
			}
		}
	}
}




/*
	d_total_hs_mat is the length of the sourth length, where each pointer points
		to h_s minibatch at that source index

	parallelism works as follows:
	each block copies one h_s vector for each minibatch

	d_indices is the size of (2*D + 1) * minibatch of ints
	-1 index means that the alignment is not pointing to a valid source index, will need to zero this out in the exped scores
	
	change the parallelism to make each block do 2*D + 1 operations??? Benchmark this

*/

template<typename dType>
__global__
void load_in_hs_kernel(dType **d_total_hs_mat, int D,dType *d_hs_mat,int *d_indices,int minibatch_size,int LSTM_size,int *d_batch_info) {

	//each block is responsible for copying one h_s vector into the current h_s
	for(int i=blockIdx.x; i < (2*D+1)*minibatch_size; i+=gridDim.x) {
		int minibatch_index = i % minibatch_size;
		int source_index = d_indices[i];
		//printf("index: %d   new-index: %d   offset: %d     length of sentence:  %d\n",source_index,d_batch_info[minibatch_index] - 1 - source_index,d_batch_info[minibatch_size+minibatch_index],d_batch_info[minibatch_index]);
		if(source_index!=-1) {
			for(int j=threadIdx.x; j < LSTM_size ;j+=blockDim.x) {
				//d_hs_mat[IDX2C(j,i,LSTM_size)] = d_total_hs_mat[source_index+d_batch_info[minibatch_size+minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)]; WWW				
				d_hs_mat[IDX2C(j,i,LSTM_size)] = d_total_hs_mat[d_batch_info[minibatch_index] - 1 - source_index + d_batch_info[minibatch_size+minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)];
			}
		}
		else {
			for(int j=threadIdx.x; j < LSTM_size ;j+=blockDim.x) {
				d_hs_mat[IDX2C(j,i,LSTM_size)] = 0;
			}
		}
	}
}


// template<typename dType>
// __global__
// void hs_DEBUG(dType **d_total_hs_mat,int minibatch_size,int LSTM_size,int longest_sent) {
// 	for(int i=blockIdx.x; i < longest_sent; i+=gridDim.x) {
// 		for(int j=threadIdx.x; j < LSTM_size*minibatch_size ;j+=blockDim.x) {
// 			d_total_hs_mat[i][j]+=1;
// 		}
// 	}
// }


template<typename dType>
__global__
void exp_mask_kernel(int *d_indicies,dType *d_alignments,int minibatch_size,int D) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<minibatch_size*(2*D+1); i+=gridDim.x*blockDim.x) {
		//int minibatch_index = i % minibatch_size;
		int source_index = d_indicies[i];
		if(source_index==-1) {
			d_alignments[i] = 0;
		}
		else {
			d_alignments[i] = exp(d_alignments[i]);
		}
	}
}

/*
	alignment are stored in the following way:
	[minibatch, minibatch, minibatch, ...]

	each thread does a reduction for a minibatch
*/

template<typename dType>
__global__
void alignment_reduction_kernel(dType *d_alignments, int LSTM_size,int minibatch_size,int D,dType sigma_sq,dType *d_p_t,int *d_indicies,dType *d_cached_exp) {

	int minibatch_index = threadIdx.x;
	if(minibatch_index < minibatch_size) {
		dType sum=0;
		dType max_val = 0;
		for(int i=0; i<2*D+1; i++) {
			if(d_indicies[minibatch_index + minibatch_size*i]!=-1) {
				if(d_alignments[minibatch_index + minibatch_size*i] > max_val) {
					max_val = d_alignments[minibatch_index + minibatch_size*i];
				}
			}
		}

		for(int i=0; i<2*D+1; i++) {
			if(d_indicies[minibatch_index + minibatch_size*i]!=-1) {
				d_alignments[minibatch_index + minibatch_size*i] = exp(d_alignments[minibatch_index + minibatch_size*i]-max_val);
				sum+= d_alignments[minibatch_index + minibatch_size*i];
			}
			else {
				d_alignments[minibatch_index + minibatch_size*i] = 0;
			}
		}

		for(int i=0; i<2*D+1; i++) {
			if(d_indicies[minibatch_index + minibatch_size*i]!=-1) {
				dType temp = exp( ( -1*pow_wrapper( ( d_p_t[minibatch_index] - d_indicies[minibatch_index + minibatch_size*i] ) ,2.0) )/(2*sigma_sq) );
				
				if(sum!=0) {
					d_alignments[minibatch_index + minibatch_size*i] = (d_alignments[minibatch_index + minibatch_size*i]/sum) \
						*temp;
				}
					//*exp( ( -1*pow( (d_p_t[minibatch_index]-d_indicies[minibatch_index + minibatch_size*i]) ,2.0) )/(2*sigma_sq) );

				d_cached_exp[IDX2C(i,minibatch_index,2*D+1)] = temp;
			}
			else {
				d_alignments[minibatch_index + minibatch_size*i] = 0;
				d_cached_exp[IDX2C(i,minibatch_index,2*D+1)] = 1; //since you divide by this
			}
		}
	}
}

/*
	Each block is responsible for multiplying one column of a h_t matrix

	alignments is laid out as:
	[minibatch] [minibatch] [minibatch] ...
*/
template<typename dType>
__global__
void create_c_t_kernel(dType *d_alignments,dType *d_hs_mat,dType *d_c_t,int LSTM_size,int minibatch_size,int D) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_c_t[i]=0;
		int minibatch_index = (i/LSTM_size);
		for(int j=0; j<2*D+1; j++) {
			d_c_t[i] +=	d_alignments[minibatch_index + minibatch_size*j] * d_hs_mat[i + LSTM_size*minibatch_size*j];
		}
	}
}


template<typename dType>
__global__
void add_two_mats_kernel(dType *d_mat1,dType *d_mat2,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat1[i] = d_mat1[i] + d_mat2[i];
	}
}

template<typename dType>
__global__
void add_two_mats_into_third_kernel(dType *d_mat1,dType *d_mat2,dType *d_mat3,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat1[i] = d_mat2[i] + d_mat3[i];
	}
}


template<typename dType>
__global__
void tanh_grad_kernel(dType *d_output,dType *d_input_Error,dType *d_tanh_val,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_output[i] = d_input_Error[i] * (1- d_tanh_val[i]*d_tanh_val[i]);
	}
}	


template<typename dType>
__global__
void tanh_att_forward_kernel(dType *d_output,dType *d_in1,dType *d_in2,dType *d_bias,int LSTM_size,int minibatch_size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_output[i] = tanh_wrapper(d_in1[i] + d_in2[i] + d_bias[i%LSTM_size]);
	}
}


#define NUM_ATTENTION_THREADS 128
//used for the part 2 of the score function
template<typename dType>
__global__
void elem_reduce_kernel(dType *d_h_t,dType *d_Wa_hs_temp, dType *d_alignments, int LSTM_size, int minibatch_size) {

	__shared__ dType buffer[NUM_ATTENTION_THREADS];
	int minibatch_index = blockIdx.x;
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;
	buffer[tid] = 0;

	for(int i=i_start; i<i_end; i+=i_step) {
		buffer[tid] += d_h_t[IDX2C(i,minibatch_index,LSTM_size)] * d_Wa_hs_temp[IDX2C(i,minibatch_index,LSTM_size)];
	}

	 __syncthreads();

	 for(int stride=NUM_ATTENTION_THREADS/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

  	__syncthreads();

	dType sum_k = buffer[0];
	if(tid==0) {
		d_alignments[minibatch_index] = sum_k; 
	}
}




//this is an improvement over the above kernel as more is done in one kernel launch
template<typename dType>
__global__
void elem_reduce_kernel_large(dType *d_h_t,dType *d_Wa_hs_temp, dType *d_alignments, int LSTM_size, int minibatch_size,int D) {

	__shared__ dType buffer[NUM_ATTENTION_THREADS];
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	for(int minibatch_index = blockIdx.x; minibatch_index<(2*D+1)*minibatch_size; minibatch_index+=gridDim.x) {
		buffer[tid] = 0;

		for(int i=i_start; i<i_end; i+=i_step) {
			buffer[tid] += d_h_t[IDX2C(i,minibatch_index,LSTM_size)] * d_Wa_hs_temp[IDX2C(i,minibatch_index%minibatch_size,LSTM_size)];
		}

		 __syncthreads();

		 for(int stride=NUM_ATTENTION_THREADS/2; stride>0; stride>>=1) {
			if(tid < stride) {
				buffer[tid] += buffer[stride + tid];
			}
			__syncthreads();
		}

	  	__syncthreads();

	  	
		dType sum_k = buffer[0];
		if(tid==0) {
			d_alignments[minibatch_index] = sum_k; 
		}
		__syncthreads();
	}
}



//used for the part 2 of the score function
template<typename dType>
__global__
void error_alignments_kernel(dType *d_ERRnTOt_ct,dType *d_hs_mat, dType *d_ERRnTOt_as, int LSTM_size, int minibatch_size,int s_index,int D) {

	__shared__ dType buffer[NUM_ATTENTION_THREADS];
	int minibatch_index = blockIdx.x;
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;
	buffer[tid] = 0;

	for(int i=i_start; i<i_end; i+=i_step) {
		buffer[tid] += d_ERRnTOt_ct[IDX2C(i,minibatch_index,LSTM_size)] * d_hs_mat[IDX2C(i,minibatch_index,LSTM_size)];
	}

	 __syncthreads();

	 for(int stride=NUM_ATTENTION_THREADS/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

  	__syncthreads();

  	// normalize the softmax
	dType sum_k = buffer[0];
	if(tid==0) {
		d_ERRnTOt_as[s_index + (2*D+1)*minibatch_index] = sum_k; 
	}
}



//used for the part 2 of the score function
template<typename dType>
__global__
void error_alignments_kernel_large(dType *d_ERRnTOt_ct,dType *d_hs_mat, dType *d_ERRnTOt_as, int LSTM_size, int minibatch_size,int D) {

	__shared__ dType buffer[NUM_ATTENTION_THREADS];
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	for(int minibatch_index = blockIdx.x; minibatch_index<(2*D+1)*minibatch_size; minibatch_index+=gridDim.x) {

		buffer[tid] = 0;
		int s_index = minibatch_index/minibatch_size;
		for(int i=i_start; i<i_end; i+=i_step) {
			buffer[tid] += d_ERRnTOt_ct[IDX2C(i,minibatch_index%minibatch_size,LSTM_size)] * d_hs_mat[IDX2C(i,minibatch_index,LSTM_size)];
		}

		 __syncthreads();

		 for(int stride=NUM_ATTENTION_THREADS/2; stride>0; stride>>=1) {
			if(tid < stride) {
				buffer[tid] += buffer[stride + tid];
			}
			__syncthreads();
		}

	  	__syncthreads();

	  	// normalize the softmax
		dType sum_k = buffer[0];
		if(tid==0) {
			d_ERRnTOt_as[s_index + (2*D+1)*(minibatch_index%minibatch_size)] = sum_k; 
		}
		__syncthreads();
	}
}




template<typename dType>
__global__
void error_pt_kernel(dType *d_ERRnTOt_pt,dType *d_ERRnTOt_as,int D,dType sigma_sq,int *d_indicies,int minibatch_size,dType *d_p_t,dType *d_alignments) {

	__shared__ dType buffer[NUM_ATTENTION_THREADS];
	int minibatch_index = blockIdx.x;
	int i_start = threadIdx.x; //start at the thread index
	int i_end = 2*D+1; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;
	buffer[tid] = 0;

	for(int i=i_start; i<i_end; i+=i_step) {
		buffer[tid] += d_ERRnTOt_as[IDX2C(i,minibatch_index,2*D+1)] * d_alignments[IDX2C(minibatch_index,i,minibatch_size)] * ( (d_indicies[minibatch_index + i*minibatch_size] - d_p_t[minibatch_index])/sigma_sq );
	}

	__syncthreads();

	 for(int stride=NUM_ATTENTION_THREADS/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

  	__syncthreads();

  	// normalize the softmax
	dType sum_k = buffer[0];
	if(tid==0) {
		d_ERRnTOt_pt[minibatch_index] = sum_k; 
	}
}


template<typename dType>
__global__
void att_vp_error(dType *d_sigma,dType *d_tanh,dType *d_temp_grad,dType *d_ERRnTOt_pt,int *d_batch_info,int LSTM_size,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		int minibatch_index = i/LSTM_size;
		d_temp_grad[i] = d_ERRnTOt_pt[minibatch_index] * d_sigma[minibatch_index] * (1-d_sigma[minibatch_index]) * d_batch_info[minibatch_index] * d_tanh[i];
	}
}


template<typename dType>
__global__
void grad_W_p_kernel(dType *d_v_p,dType *d_temp,dType *d_sigma,dType *d_tanh,dType *d_ERRnTOt_pt,int *d_batch_info,int LSTM_size,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		int minibatch_index = i/LSTM_size;
		int LSTM_index = i%LSTM_size;
		d_temp[i] = d_ERRnTOt_pt[minibatch_index] * d_batch_info[minibatch_index] * d_v_p[LSTM_index] * d_sigma[minibatch_index] * (1-d_sigma[minibatch_index]) * (1 - d_tanh[i]*d_tanh[i]);
	}
} 



//these two parts are for a highly inefficient way

// //part 1 of calculation (positive part)
// template<typename dType>
// __global__
// void prep_ht_Wa_grad_p1(dType *d_h_t,dType *d_temp1,dType *d_alignments,dType *d_ERRnTOt_as,int global_index,int LSTM_size,int minibatch_size,int D) {

// 	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
// 		int minibatch_index = i/LSTM_size;
// 		d_temp1[i] = d_ERRnTOt_as[IDX2C(global_index,minibatch_index,2*D+1)] * d_alignments[IDX2C(minibatch_index,global_index,minibatch_size)] * d_h_t[i];
// 	}
// }



// //part 2 of calculation (The negative part)
// //global index is the index outside summation
// //local index is the index inside summation
// template<typename dType>
// __global__
// void prep_ht_Wa_grad_p2(dType *d_h_t,dType *d_temp1,dType *d_alignments,dType *d_cached_exp,dType *d_ERRnTOt_as,int global_index,int local_index,int LSTM_size,int minibatch_size,int D) {

// 	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
// 		int minibatch_index = i/LSTM_size;
// 		d_temp1[i] = -1 * d_ERRnTOt_as[IDX2C(global_index,minibatch_index,2*D+1)] *d_alignments[IDX2C(minibatch_index,global_index,minibatch_size)] * ( d_alignments[IDX2C(minibatch_index,local_index,minibatch_size)]/d_cached_exp[IDX2C(local_index,minibatch_index,2*D+1)] ) * d_h_t[i];
// 	}
// }



//faster W_a gradient
template<typename dType>
__global__
void get_ht_scalings_Wa_grad_kernel(dType *d_scalings,dType *d_ERRnTOt_as,dType *d_alignments,dType *d_cached_exp,int D,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<(2*D+1)*minibatch_size; i+=gridDim.x*blockDim.x) {
		int alignment_index = i%(2*D+1);
		int minibatch_index = i/(2*D+1);
		d_scalings[i] = d_ERRnTOt_as[IDX2C(alignment_index,minibatch_index,2*D+1)] * \
			d_alignments[IDX2C(minibatch_index,alignment_index,minibatch_size)] * ( 1- \
			d_alignments[IDX2C(minibatch_index,alignment_index,minibatch_size)]/ \
			d_cached_exp[IDX2C(alignment_index,minibatch_index,2*D+1)] );
		for(int j=0; j<2*D+1; j++) {
			if(j!=alignment_index) {
				d_scalings[i] += -1*d_ERRnTOt_as[IDX2C(j,minibatch_index,2*D+1)] * d_alignments[IDX2C(minibatch_index,j,minibatch_size)] * \
					d_alignments[IDX2C(minibatch_index,alignment_index,minibatch_size)] / d_cached_exp[IDX2C(alignment_index,minibatch_index,2*D+1)];
			}
		}
	}
}


template<typename dType>
__global__
void scale_ht_kernel(dType *d_scalings,dType *d_temp1,dType *d_h_t,int LSTM_size,int minibatch_size,int alignment_index,int D) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		int minibatch_index = i/LSTM_size;
		d_temp1[i] = d_h_t[i] * d_scalings[IDX2C(alignment_index,minibatch_index,2*D+1)];
	}
}


//more efficent version of the above kernel
template<typename dType>
__global__
void scale_ht_kernel_large(dType *d_hs_sum,dType *d_hs_mat,dType *d_scalings,int LSTM_size,int minibatch_size,int D) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_hs_sum[i] = 0;
		int minibatch_index = i/LSTM_size;
		for(int j=0; j<2*D+1; j++) {
			d_hs_sum[i] += d_hs_mat[i + LSTM_size*minibatch_size*j] * d_scalings[IDX2C(j,minibatch_index,2*D+1)];
		}
	}
}


//each block will copy over one vector to the source side
template<typename dType>
__global__
void copy_errors_source(dType **d_total_hs_error,dType *d_temp_error,int *d_indicies,int LSTM_size,int minibatch_size,int D,int alignment_index,int *d_batch_info) {

	for(int i=blockIdx.x; i < minibatch_size; i+=gridDim.x) {
		int minibatch_index = i;
		int source_index = d_indicies[IDX2C(minibatch_index,alignment_index,minibatch_size)];
		if(source_index!=-1) {
			for(int j=threadIdx.x; j < LSTM_size ;j+=blockDim.x) {
				//d_total_hs_error[source_index + d_batch_info[minibatch_size + minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)] += d_temp_error[IDX2C(j,minibatch_index,LSTM_size)]; WWW
				d_total_hs_error[d_batch_info[minibatch_index] - 1 - source_index + d_batch_info[minibatch_size + minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)] += d_temp_error[IDX2C(j,minibatch_index,LSTM_size)];
			}
		}
	}	
}



//get the error for h_s from c_t
template<typename dType>
__global__
void error_hs_ct_kernel(dType *d_ERRnTOt_ct, dType *d_alignments,int *d_indicies,int *d_batch_info,dType **d_total_hs_error,int LSTM_size,int minibatch_size,int D,int alignment_index) {

	for(int i=blockIdx.x; i < minibatch_size; i+=gridDim.x) {
		int minibatch_index = i;
		int source_index = d_indicies[IDX2C(minibatch_index,alignment_index,minibatch_size)];
		if(source_index!=-1) {
			for(int j= threadIdx.x; j<LSTM_size; j+=blockDim.x) {
				d_total_hs_error[d_batch_info[minibatch_index] - 1 - source_index + d_batch_info[minibatch_size + minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)] += d_ERRnTOt_ct[IDX2C(j,minibatch_index,LSTM_size)]*d_alignments[IDX2C(minibatch_index,alignment_index,minibatch_size)];
			}
		}
	}
}


//more efficent version of kernel above
template<typename dType>
__global__
void error_hs_ct_kernel_large(dType *d_ERRnTOt_ct, dType *d_alignments,int *d_indicies,int *d_batch_info,dType **d_total_hs_error,int LSTM_size,int minibatch_size,int D) {

	for(int i=blockIdx.x; i < minibatch_size*(2*D+1); i+=gridDim.x) {
		int minibatch_index = i%minibatch_size;
		int alignment_index = i/minibatch_size;
		int source_index = d_indicies[IDX2C(minibatch_index,alignment_index,minibatch_size)];
		if(source_index!=-1) {
			for(int j= threadIdx.x; j<LSTM_size; j+=blockDim.x) {
				//d_total_hs_error[source_index + d_batch_info[minibatch_size + minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)] += d_ERRnTOt_ct[IDX2C(j,minibatch_index,LSTM_size)]*d_alignments[IDX2C(minibatch_index,alignment_index,minibatch_size)]; WWW
				d_total_hs_error[d_batch_info[minibatch_index] - 1 - source_index + d_batch_info[minibatch_size + minibatch_index]][IDX2C(j,minibatch_index,LSTM_size)] += d_ERRnTOt_ct[IDX2C(j,minibatch_index,LSTM_size)]*d_alignments[IDX2C(minibatch_index,alignment_index,minibatch_size)];
			}
		}
	}
}



template<typename dType>
__global__
void gradient_update_mats(dType *d_mat,dType *d_mat_grad,dType learning_rate,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_mat[i]+= learning_rate * d_mat_grad[i];
	}
}



template<typename dType>
__global__
void zero_h_t(dType *d_h_t, int *d_01_mask,int LSTM_size,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_h_t[i] *= d_01_mask[i/LSTM_size];
	}
}


template<typename dType>
__global__
void clip_mat_kernel(dType *d_mat,dType threshold,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		if(d_mat[i] > 0) {
			d_mat[i] = (d_mat[i] > threshold) ? threshold : d_mat[i];
		}
		else {
			d_mat[i] = (d_mat[i] < -threshold) ? -threshold : d_mat[i];
		}
	}
}


//-------------------------------------------NCE Stuff ------------------------------------------

#define NUM_NCE_THREADS 128

//copy into temp embeddings
//num samples is the size of the negative samples shared across a minibatch and the positive samples
template<typename dType>
__global__
void load_in_embeddings(dType *d_temp_embeddings,dType *d_D,int *d_samples,int num_samples,int LSTM_size) {

	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < num_samples; k+=gridDim.x) {
		int vocab_index = d_samples[k];
		for(int i= i_start; i < i_end; i += i_step) {
			d_temp_embeddings[IDX2C(i,k,LSTM_size)] = d_D[IDX2C(i,vocab_index,LSTM_size)];
		}
	}
}

template<typename dType>
__global__
void nce_dot_product_SPARSE(dType *d_dot_products,dType *d_D,dType *d_h_t,int *d_samples,int LSTM_size,int minibatch_size,int num_samples,int output_vocab_size) {

	__shared__ dType buffer[NUM_NCE_THREADS];

	//per block doing an embedding
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	for(int k = blockIdx.x; k < num_samples*minibatch_size; k+=gridDim.x) {
		int minibatch_index = k/num_samples;
		int sample_index = k%num_samples;
		int vocab_index = d_samples[IDX2C(sample_index,minibatch_index,num_samples)];
		buffer[tid] = 0;

		for(int i=i_start; i<i_end; i+=i_step) {
			buffer[tid] += d_h_t[IDX2C(i,minibatch_index,LSTM_size)] * d_D[IDX2C(i,vocab_index,LSTM_size)];
		}

		 __syncthreads();

		 for(int stride=NUM_NCE_THREADS/2; stride>0; stride>>=1) {
			if(tid < stride) {
				buffer[tid] += buffer[stride + tid];
			}
			__syncthreads();
		}

	  	__syncthreads();

	  	
		dType sum_k = buffer[0];
		if(tid==0) {
			d_dot_products[IDX2C(sample_index,minibatch_index,num_samples)] = sum_k; 
		}
		__syncthreads();
	}
}


template<typename dType>
__device__
inline dType log_add_exp(dType x,dType y) {

	dType min = cuda_min_wrapper(x,y);
	dType max = cuda_max_wrapper(x,y);
	return max + cuda_log1p_wrapper(cuda_exp_wrapper(min-max));
}

//compute -P(true) for all of the elements
template<typename dType>
__global__
void calc_p_true_kernel(dType *d_p_true,dType *d_dot_products,dType *d_sampling_probs,dType *d_b_d,int *d_samples,int num_samples,int minibatch_size,int *d_vocab_indicies_01) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<num_samples*minibatch_size; i+=gridDim.x*blockDim.x) {
		int minibatch_index = i%minibatch_size;
		int sample_index = i/minibatch_size;
		//printf("%i\n",d_samples[sample_index]);
		if(d_vocab_indicies_01[minibatch_index]==1) {
			d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)] = -1*cuda_exp_wrapper( d_dot_products[IDX2C(sample_index,minibatch_index,num_samples)] + d_b_d[d_samples[sample_index]] - \
				log_add_exp(d_dot_products[IDX2C(sample_index,minibatch_index,num_samples)] + d_b_d[d_samples[sample_index]],d_sampling_probs[sample_index]) ); 

			assert(isinf_wrapper(  log_add_exp(d_dot_products[IDX2C(sample_index,minibatch_index,num_samples)] + d_b_d[d_samples[sample_index]],d_sampling_probs[sample_index])  )==0);
			// if(d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)]==0) {
			// 	printf("zero!!!, minibatch_index: %d , sample_index: %d ,   dot_product: %f  ,  d_sampling_probs: %f  logaddexp val: %f \n",minibatch_index,sample_index,\
			// 		d_dot_products[IDX2C(sample_index,minibatch_index,num_samples)] + d_b_d[d_samples[sample_index]], d_sampling_probs[sample_index] \
			// 		,log_add_exp(d_dot_products[IDX2C(sample_index,minibatch_index,num_samples)] + d_b_d[d_samples[sample_index]],d_sampling_probs[sample_index]));
			// }
		}
		else {
			//printf("Setting to zero\n");
			d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)] = 0;
		}
	}		
}


//compute -P(true) for all of the elements
template<typename dType>
__global__
void calc_p_true_kernel_nonshare(dType *d_p_true,dType *d_dot_products,dType *d_sampling_probs,dType *d_b_d,int *d_samples,int num_neg_samples,int minibatch_size,int *d_vocab_indicies_01) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<(num_neg_samples+1)*minibatch_size; i+=gridDim.x*blockDim.x) {
		int minibatch_index = i/(num_neg_samples+1);
		int sample_index = i%(num_neg_samples+1);
		//printf("%i\n",d_samples[sample_index]);
		if(d_vocab_indicies_01[minibatch_index]==1) {
			d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)] = -1*cuda_exp_wrapper( d_dot_products[IDX2C(sample_index,minibatch_index,num_neg_samples+1)] + d_b_d[d_samples[sample_index]] - \
				log_add_exp(d_dot_products[IDX2C(sample_index,minibatch_index,num_neg_samples+1)] + d_b_d[d_samples[sample_index]],d_sampling_probs[sample_index]) ); 
		}
		else {
			//printf("Setting to zero\n");
			d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)] = 0;
		}
	}		
}


//get the objective value for NCE
template<typename dType>
__global__
void objective_val_p1_NCE_kernel(dType *d_p_true,double *d_OBJ_val_temp,int num_negative_samples,int minibatch_size,int *d_vocab_indicies_01) {

	__shared__ dType buffer[NUM_NCE_THREADS];
	int i_start = threadIdx.x; //start at the thread index
	int i_end = minibatch_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	int tid = threadIdx.x;
	buffer[threadIdx.x] = 0;

	for(int k = blockIdx.x; k < num_negative_samples; k+=gridDim.x) {
		for(int i=i_start; i<i_end; i+= i_step) {
			if(d_vocab_indicies_01[i]==1) {
				// if(isinf_wrapper(cuda_log_wrapper(1+d_p_true[IDX2C(i,k,minibatch_size)]))) {
				// 	printf("Value for inf %f , minibatch_index %d \n",d_p_true[IDX2C(i,k,minibatch_size)],i);
				// }
				assert(isinf_wrapper(cuda_log_wrapper(1+d_p_true[IDX2C(i,k,minibatch_size)]))==0);
				buffer[threadIdx.x] += cuda_log_wrapper(1+d_p_true[IDX2C(i,k,minibatch_size)]);
			}
		}
	}

	__syncthreads();

	for(int stride=NUM_NCE_THREADS/2; stride>0; stride>>=1) {
		if(tid < stride) {
			buffer[tid] += buffer[stride + tid];
		}
		__syncthreads();
	}

	__syncthreads();

	if(tid==0) {
		d_OBJ_val_temp[blockIdx.x]=buffer[0];
	}
}




//get the objective value for NCE
template<typename dType>
__global__
void objective_val_p2_NCE_kernel(dType *d_p_true,double *d_final_NCE_OBJ,double *d_OBJ_val_temp,int num_negative_samples,int minibatch_size,int *d_vocab_indicies_01) {

	for(int i=0; i<NUM_NCE_THREADS; i++) {
		d_final_NCE_OBJ[0] +=d_OBJ_val_temp[i];
	}

	for(int i=0; i<minibatch_size; i++) {
		if(d_vocab_indicies_01[i]==1) {
			// if(isinf_wrapper(cuda_log_wrapper(-d_p_true[IDX2C(i,i+num_negative_samples,minibatch_size)]))) {
			// 	printf("Inf statment, val: %f   minibatch index: %d ",d_p_true[IDX2C(i,i+num_negative_samples,minibatch_size)],i);
			// }
			assert(isinf_wrapper(cuda_log_wrapper(-d_p_true[IDX2C(i,i+num_negative_samples,minibatch_size)]))==0);
			d_final_NCE_OBJ[0]+=cuda_log_wrapper(-d_p_true[IDX2C(i,i+num_negative_samples,minibatch_size)]);
		}
	}
}

//get the objective value for NCE
template<typename dType>
__global__
void objective_val_p2_NCE_kernel_nonshare(dType *d_p_true,double *d_final_NCE_OBJ,double *d_OBJ_val_temp,int num_negative_samples,int minibatch_size,int *d_vocab_indicies_01) {

	for(int i=0; i<NUM_NCE_THREADS; i++) {
		d_final_NCE_OBJ[0] +=d_OBJ_val_temp[i];
	}

	for(int i=0; i<minibatch_size; i++) {
		if(d_vocab_indicies_01[i]==1) {
			d_final_NCE_OBJ[0]+=cuda_log_wrapper(-d_p_true[IDX2C(i,1+num_negative_samples,minibatch_size)]);
		}
	}
}

template<typename dType>
__global__
void zero_err_ht(dType *d_err_ht,int *d_vocab_indicies_01,int LSTM_size,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_err_ht[i] *= d_vocab_indicies_01[i/LSTM_size];
	}
}



//compute d_err_ht with respect to positive embeddings
//temp embeddings pointer being passed in skips the beginning negative sample embeddings
template<typename dType>
__global__
void error_ht_positive_kernel(dType *d_d_ERRt_ht,dType *d_p_true,dType *d_temp_embeddings,int num_negative_samples,int LSTM_size,int minibatch_size) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		int minibatch_index = i/LSTM_size;
		int LSTM_index = i%LSTM_size;
		d_d_ERRt_ht[IDX2C(LSTM_index,minibatch_index,LSTM_size)] += (1 + d_p_true[IDX2C(minibatch_index,num_negative_samples+minibatch_index,minibatch_size)]) * d_temp_embeddings[IDX2C(LSTM_index,minibatch_index,LSTM_size)];
	}		
}


template<typename dType>
__global__
void backprop_ht_SPARSE(dType *d_d_ERRt_ht,dType *d_D,dType *d_p_true,int *d_samples,int LSTM_size,int minibatch_size,int num_neg_samples,dType *d_reduction_space) {

	//per block doing an embedding
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	const int tid = threadIdx.x;

	for(int k = blockIdx.x; k < (num_neg_samples+1)*minibatch_size; k+=gridDim.x) {
		int minibatch_index = k/(num_neg_samples+1);
		int sample_index = k%(num_neg_samples+1);
		int vocab_index = d_samples[k];

		if(sample_index!=num_neg_samples) {
			for(int i=i_start; i<i_end; i+=i_step) {
				d_reduction_space[sample_index + (num_neg_samples+1)*i + (num_neg_samples+1)*LSTM_size*minibatch_index] = d_D[IDX2C(i,vocab_index,LSTM_size)]*d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)];
				//atomicAdd(&(d_d_ERRt_ht[(i,minibatch_index,LSTM_size)]),val);
			}
		}
		else {
			for(int i=i_start; i<i_end; i+=i_step) {
				d_reduction_space[sample_index + (num_neg_samples+1)*i + (num_neg_samples+1)*LSTM_size*minibatch_index] = d_D[IDX2C(i,vocab_index,LSTM_size)]*(1+d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)]);
				//atomicAdd(&(d_d_ERRt_ht[(i,minibatch_index,LSTM_size)]),val);
			}
		}
	}

	//now do the reduction
	__syncthreads();
	__shared__ dType buffer[NUM_NCE_THREADS];

	//per block doing an embedding
	i_start = threadIdx.x; //start at the thread index
	i_end = num_neg_samples+1; //end at dim
	i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < LSTM_size*minibatch_size; k+=gridDim.x) {
		int minibatch_index = k/LSTM_size;
		int lstm_index = k%LSTM_size;
		buffer[tid] = 0;

		for(int i=i_start; i<i_end; i+=i_step) {
			buffer[tid] += d_reduction_space[i + (num_neg_samples+1)*lstm_index + (num_neg_samples+1)*LSTM_size*minibatch_index];
		}

		 __syncthreads();

		 for(int stride=NUM_NCE_THREADS/2; stride>0; stride>>=1) {
			if(tid < stride) {
				buffer[tid] += buffer[stride + tid];
			}
			__syncthreads();
		}

	  	__syncthreads();

	  	
		dType sum_k = buffer[0];
		if(tid==0) {
			d_d_ERRt_ht[IDX2C(lstm_index,minibatch_index,LSTM_size)] = sum_k; 
		}
		__syncthreads();
	}
}


template<typename dType>
__global__
void embedding_gradient_sparse(dType *d_D_grad,dType *d_h_t,dType *d_p_true,int *d_samples,int LSTM_size,int minibatch_size,int num_neg_samples) {

	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step
	//int tid = threadIdx.x;

	for(int k = blockIdx.x; k < (num_neg_samples+1)*minibatch_size; k+=gridDim.x) {
		int minibatch_index = k/(num_neg_samples+1);
		int sample_index = k%(num_neg_samples+1);
		int vocab_index = d_samples[k];

		if(sample_index!=num_neg_samples) {
			for(int i=i_start; i<i_end; i+= i_step) {
				dType val = d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)]*d_h_t[IDX2C(i,minibatch_index,LSTM_size)];
				atomicAdd(&(d_D_grad[IDX2C(i,vocab_index,LSTM_size)]),val);
			}
		}
		else {
			for(int i=i_start; i<i_end; i+= i_step) {
				dType val = (1+d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)])*d_h_t[IDX2C(i,minibatch_index,LSTM_size)];
				atomicAdd(&(d_D_grad[IDX2C(i,vocab_index,LSTM_size)]),val);
			}
		}
	}
}


template<typename dType>
__global__
void bias_gradient_sparse(dType *d_b_d_grad,dType *d_p_true,int *d_samples,int LSTM_size,int minibatch_size,int num_neg_samples) {
	for(int k = blockIdx.x; k < (num_neg_samples+1)*minibatch_size; k+=gridDim.x) {
		int minibatch_index = k/(num_neg_samples+1);
		int sample_index = k%(num_neg_samples+1);
		int vocab_index = d_samples[k];

		if(sample_index!=num_neg_samples) {
			atomicAdd(&(d_b_d_grad[vocab_index]),d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)]);
		}
		else {
			atomicAdd(&(d_b_d_grad[vocab_index]),1+d_p_true[IDX2C(minibatch_index,sample_index,minibatch_size)]);
		}
	}
}



//d_samples being passed in is pointing at the positive samples already
template<typename dType>
__global__
void positive_embedding_NCE(dType *d_h_t,dType *d_small_D_grad,dType *d_p_true,int *d_samples,int LSTM_size,int minibatch_size,int *d_vocab_indicies_01,int *d_reverse_unique_indicies) {
	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < minibatch_size; k+=gridDim.x) {
		int vocab_index = d_samples[k];
		if(d_vocab_indicies_01[k]==1) {
			for(int i= i_start; i < i_end; i += i_step) {
				//atomicAdd(&(d_D_grad[IDX2C(i,vocab_index,LSTM_size)]),d_h_t[IDX2C(i,k,LSTM_size)]*(1+d_p_true[IDX2C(k,k,minibatch_size)]));
				atomicAdd(&(d_small_D_grad[IDX2C(i,d_reverse_unique_indicies[vocab_index],LSTM_size)]),d_h_t[IDX2C(i,k,LSTM_size)]*(1+d_p_true[IDX2C(k,k,minibatch_size)]));
			}
		}
	}
}


template<typename dType>
__global__
void negative_embedding_NCE(dType *d_temp_D_grad,dType *d_small_D_grad,int *d_samples,int num_negative_samples,int LSTM_size,int *d_reverse_unique_indicies) {

	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < num_negative_samples; k+=gridDim.x) {
		int vocab_index = d_samples[k];
		for(int i= i_start; i < i_end; i += i_step) {
			//atomicAdd(&(d_D_grad[IDX2C(i,vocab_index,LSTM_size)]),d_temp_D_grad[IDX2C(i,k,LSTM_size)]);
			atomicAdd(&(d_small_D_grad[IDX2C(i,d_reverse_unique_indicies[vocab_index],LSTM_size)]),d_temp_D_grad[IDX2C(i,k,LSTM_size)]);
		}
	}
}


__global__
void setup_reverse_indicies(int *d_reverse_unique_indicies,int *d_unique_indicies,int curr_num_unique) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<curr_num_unique; i+=gridDim.x*blockDim.x) {
		//int temp = d_unique_indicies[i];
		//printf("%d\n",d_unique_indicies[i]);
		d_reverse_unique_indicies[d_unique_indicies[i]] = i;
	}
}


template<typename dType>
__global__
void update_sparse_grad(dType *d_mat,dType *d_small_grad,int *d_unique_indicies,int curr_num_unique,dType learning_rate,int LSTM_size) {
	
	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < curr_num_unique; k+=gridDim.x) {
		int vocab_index = d_unique_indicies[k];
		for(int i= i_start; i < i_end; i += i_step) {
			//atomicAdd(&(d_D_grad[IDX2C(i,vocab_index,LSTM_size)]),d_temp_D_grad[IDX2C(i,k,LSTM_size)]);
			d_mat[IDX2C(i,vocab_index,LSTM_size)] += learning_rate*d_small_grad[IDX2C(i,k,LSTM_size)];
		}
	}
}


template<typename dType>
__global__
void negative_bias_NCE(dType *d_temp_b_d_grad,dType *d_b_d_grad,int *d_samples,int num_negative_samples) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<num_negative_samples; i+=gridDim.x*blockDim.x) {
		atomicAdd(&(d_b_d_grad[d_samples[i]]),d_temp_b_d_grad[i]);
	}
}

template<typename dType>
__global__
void positive_bias_NCE(dType *d_b_d_grad,dType *d_p_true,int *d_samples,int minibatch_size,int num_negative_samples,int *d_vocab_indicies_01) {

	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<minibatch_size; i+=gridDim.x*blockDim.x) {
		if(d_vocab_indicies_01[i]==1) {
			atomicAdd(&(d_b_d_grad[d_samples[i+num_negative_samples]]),1+d_p_true[IDX2C(i,i+num_negative_samples,minibatch_size)]);
		}
	}
}




//----------------------------------------------- Truncated softmax stuff
template<typename dType>
__global__
void load_in_embeddings_trunc(dType *d_temp_embeddings,dType *d_D,int *d_samples,int num_samples,int LSTM_size) {

	int i_start = threadIdx.x; //start at the thread index
	int i_end = LSTM_size; //end at dim
	int i_step = blockDim.x; //the block dimension (aka the number of threads in the block) is the step

	for(int k = blockIdx.x; k < num_samples; k+=gridDim.x) {
		int vocab_index = d_samples[k];
		for(int i= i_start; i < i_end; i += i_step) {
			d_temp_embeddings[IDX2C(i,k,LSTM_size)] = d_D[IDX2C(i,vocab_index,LSTM_size)];
		}
	}
}








//----------------------------------------------bidirectional encoder-----------------------------------------------
template<typename dType>
__global__
void tanh_bi_forward_kernel(dType *d_mat,dType *d_bias,int LSTM_size,int minibatch_size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size*minibatch_size; i+=gridDim.x*blockDim.x) {
		d_mat[i] = tanh_wrapper(d_mat[i] + d_bias[i%LSTM_size]);
	}
}

//for bidirectional model
//used for ht and ct
template<typename dType>
__global__
void add_to_errors(dType *d_error_ht,dType *d_additional_error_ht,int LSTM_size,int index) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<LSTM_size; i+=gridDim.x*blockDim.x) {
		d_error_ht[IDX2C(i,index,LSTM_size)] += d_additional_error_ht[IDX2C(i,index,LSTM_size)];
	}
}



//-----------------------------------------------          -------------------------------------------------------
template<typename dType>
__global__
void add_two_mats(dType *d_final,dType *d_mat1,dType *d_mat2,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_final[i] = d_mat1[i] + d_mat2[i];
	}
}


template<typename dType>
__global__
void check_elems_kernel(dType *d_mat,int size) {
	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		dType x = d_mat[i];
		d_mat[i] = x;
	}
}


template<typename dType>
__global__ 
void add_four_matrices_kernel_stride(dType *d_final,dType *d_mat1,dType *d_mat2,dType *d_mat3,dType *d_mat4,int size) 
{
  	for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<size; i+=gridDim.x*blockDim.x) {
		d_final[i] = d_mat1[i] + d_mat2[i] + d_mat3[i] + d_mat4[i];
	}
}


// for decoder.h

// each block is responsible for a beam_index;
template<typename dType>
__global__
void add_features(dType *probs, dType * sentence_scores, dType * encourage,
                  int *last_word_index, dType adjacent_weight,
                  int *word_len, dType wordlen_weight,
                  int *vocab_bin, dType alliteration_weight,
                  int beam_size, int vocab_size){
    /*
     probs: beam_size * vocab_size;
     sentence_scores: beam_size;
     encourange: vocab_size;
     last_word_index : beam_size;
     word_len : vocab_size;
     */
    int beam_index = blockIdx.x;
    dType sentence_score = sentence_scores[beam_index];
    int last_word = last_word_index[beam_index];
    for(int index=threadIdx.x; index<vocab_size; index+=blockDim.x){
        int global_index = index + beam_index * vocab_size;
        probs[global_index] = log( probs[global_index] ) + sentence_score + encourage[index] + wordlen_weight * word_len[index] * word_len[index];
        if (last_word >=0){
            probs[global_index] += adjacent_weight * (index == last_word);
            probs[global_index] += alliteration_weight * (vocab_bin[last_word] == vocab_bin[index]);
        }
        
    }
}

template<typename dType>
__global__
void add_feature_repeat(dType *probs,
                  int *sentence_set, dType repeat_weight, int size,
                  int vocab_size){
    /*
     sentence_set : [word_index, beam_index, occr_times , ... ,]  = size * 3
     */
    for(int index=threadIdx.x + blockIdx.x*blockDim.x; index<size; index+=gridDim.x*blockDim.x) {
        int word_index = sentence_set[index*3];
        int beam_index = sentence_set[index*3+1];
        int occr_times = sentence_set[index*3+2];
        probs[word_index + beam_index * vocab_size] += occr_times * repeat_weight;
    }
}





// each block is responsible for a beam_index;
template<typename dType>
__global__
void top_k(dType *probs, dType *results, int* pointers, int* dict, int *beams, int *valid_vocab_sizes, int vocab_size)
{
    int beam_index = blockIdx.x;
    int start = valid_vocab_sizes[beam_index];
    int end = valid_vocab_sizes[beam_index + 1];
    for(int index=threadIdx.x; index<end - start; index+=blockDim.x) {
        int dict_index = index + start;
        int prob_index = beam_index * vocab_size + dict[dict_index];
        results[dict_index] = probs[prob_index];
        beams[dict_index] = beam_index;
        pointers[dict_index] = dict_index;
    }
    
}

template<typename dType>
__global__
void top_k(dType *probs, dType *results, int* dict, int dict_size) {
    for(int index=threadIdx.x + blockIdx.x*blockDim.x; index<dict_size; index+=gridDim.x*blockDim.x) {
        int prob_index = dict[index];
        results[index] = log( probs[prob_index] );
    }
    
}

// for LSH

// each core is responsible for one band
// W blocks ; 256 threads per block : for loop vocab_size / 256 ;
// <<<W, 256>>>
template<typename dType>
__global__
void hash_code_kernel(unsigned int *d_codes, dType *d_vectors, int * d_permutes, int P, int W, int K, int units_per_band, int bits_per_band, int n_vector) {
    // bits_per_band = log2(K) * units_per_band;
    int band_index = blockIdx.x;
    for (int vocab_index = threadIdx.x; vocab_index < n_vector; vocab_index += blockDim.x) {
        unsigned int code = 0;
        for (int u = 0 ; u < units_per_band; u ++ ){
            dType max_val = -1000000000;
            int max_index = -1;
            for (int p = 0 ; p < K; p ++){
                int dim = d_permutes[band_index * units_per_band * K + u * K + p];
                dType val = d_vectors[dim * n_vector + vocab_index];
                if (max_val < val){
                    max_val = val;
                    max_index = p;
                }
            }
            code = (code << bits_per_band) + max_index;
        }
        int code_index = band_index * n_vector + vocab_index;
        d_codes[code_index] = code;
    }
}

// <<<beam_size, 256>>>
// d_h_t_pad [beam_size, LSTM_size + 1];
// d_h_t [LSTM_size, beam_size]
template<typename dType>
__global__
void pad_h_t(dType * d_h_t_pad, dType *d_h_t, int LSTM_size, int beam_size){
    int beam_index = blockIdx.x;
    for (int i = threadIdx.x; i < LSTM_size + 1; i += blockDim.x){
        if (i == LSTM_size){
            d_h_t_pad[i * beam_size + beam_index] = 1.0;
        } else {
            d_h_t_pad[i * beam_size + beam_index] = d_h_t[beam_index *  LSTM_size + i];
        }
    }
    
    
}


// d_results : [m, batch_size]
// d_Db : [vocab_size, LSTM_size + 1]
// d_h_t_pad: [batch_size, LSTM_size + 1]
// d_top_ids: [m, batch_size]
// complexity: m * batch_size * (LSTM_size + 1)
// <<<(m,batch_size), 256>>> 256 is required;
// each block just calculate one single dot product;
template<typename dType>
__global__
void sparse_dot_product(dType *d_outputdist, dType *d_results, dType *d_Db, dType *d_h_t_pad, int * d_top_ids, int m, int LSTM_size, int batch_size, int vocab_size){
   
    const int nthreads = 256;
    __shared__ dType buffer[nthreads];

    int m_index = blockIdx.x;
    int batch_index = blockIdx.y;
    int vocab_index = d_top_ids[batch_index * m + m_index];
    if (vocab_index >= 0){
        buffer[threadIdx.x] = 0.0;
        for (int i = threadIdx.x ; i < LSTM_size + 1; i += blockDim.x){
            buffer[threadIdx.x] += d_Db[IDX2C(vocab_index,i, vocab_size)] * d_h_t_pad[IDX2C(batch_index, i, batch_size)];
        }
        
        __syncthreads();
        
        // reduce
        for (int stride = nthreads /2 ; stride > 0 ; stride = stride >> 1) {
            if (threadIdx.x < stride){
                buffer[threadIdx.x] += buffer[threadIdx.x + stride];
            }
            __syncthreads();
        }
        __syncthreads();
        d_results[IDX2C(m_index, batch_index, m)] = buffer[0];
        d_outputdist[IDX2C(vocab_index, batch_index, vocab_size)] = buffer[0];
    }
}

// d_Db : [vocab_size, LSTM_size + 1]
// d_top_ids: [m, batch_size]
// complexity: vocab_size * batch_size * (LSTM_size + 1)
// <<<(vocab_size, batch_size), 256>>> 256 is required;
// each block just calculate one single dot product;
template<typename dType>
__global__
void sparse_dot_product_2(dType *d_outputdist, dType *d_Db, dType *d_h_t_pad, int LSTM_size, int batch_size, int vocab_size){
    
    
    int vocab_index = blockIdx.x;
    int batch_index = blockIdx.y;
    
    if (d_outputdist[IDX2C(vocab_index, batch_index, vocab_size)] > 0){
        const int nthreads = 256;
        __shared__ dType buffer[nthreads];

        buffer[threadIdx.x] = 0.0;
        for (int i = threadIdx.x ; i < LSTM_size + 1; i += blockDim.x){
            buffer[threadIdx.x] += d_Db[IDX2C(vocab_index,i, vocab_size)] * d_h_t_pad[IDX2C(batch_index, i, batch_size)];
        }
        
        __syncthreads();
        
        // reduce
        for (int stride = nthreads /2 ; stride > 0 ; stride = stride >> 1) {
            if (threadIdx.x < stride){
                buffer[threadIdx.x] += buffer[threadIdx.x + stride];
            }
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0){
            d_outputdist[IDX2C(vocab_index, batch_index, vocab_size)] = buffer[0];
        }
    }
    
}


// each thread compute a single value
// <<<block, 256>>>
template<typename dType>
__global__
void sparse_dot_product_3(dType *d_outputdist, dType *d_Db, dType *d_h_t_pad, int LSTM_size, int batch_size, int vocab_size){

    int batch_index = blockIdx.x;
    
    for (int vocab_index = threadIdx.x; vocab_index < vocab_size; vocab_index += blockDim.x){
        
        if (d_outputdist[IDX2C(vocab_index, batch_index, vocab_size)] > 0){
            dType sum = 0.0;
            for (int i = 0; i<LSTM_size+1; i += 1){
                sum += d_Db[IDX2C(vocab_index,i, vocab_size)] * d_h_t_pad[IDX2C(batch_index, i, batch_size)];
            }
            d_outputdist[IDX2C(vocab_index, batch_index, vocab_size)] = sum;
        }
    }
    
}




__device__
unsigned int hash_func_1_gpu(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__device__
unsigned int hash_func_2_gpu(unsigned int key){
    unsigned int c2=0x27d4eb2d; // a prime or an odd constant
    key = (key ^ 61) ^ (key >> 16);
    key = key + (key << 3);
    key = key ^ (key >> 4);
    key = key * c2;
    key = key ^ (key >> 15);
    return key;
}

// d_codes: [batch_size, W]
// d_outputdist: [vocab_size, batch_size]
// <<<batch_size, 256>>> : each block is responsible for each batch
template<typename dType>
__global__
void cuckoo_lookup(unsigned int *d_codes, dType *d_outputdist,int batch_size, int vocab_size, int W,
                   unsigned int *d_key_1, unsigned int *d_value_1, unsigned int * d_length_1,
                   unsigned int *d_key_2, unsigned int *d_value_2, unsigned int * d_length_2,
                   unsigned int *d_bands_index){
    int batch_index = blockIdx.x;
    for (int w_index = threadIdx.x; w_index < W; w_index += blockDim.x){
        unsigned int code = d_codes[w_index * batch_size + batch_index];
        //cuckoo lookup;
        unsigned int key1 = hash_func_1_gpu(code) % vocab_size + w_index * vocab_size;
        int start = -1;
        int length = 0;
        if (d_key_1[key1] == code){
            start = d_value_1[key1];
            length = d_length_1[key1];
        } else {
            unsigned int key2 = hash_func_2_gpu(code) % vocab_size + w_index * vocab_size;
            if (d_key_2[key2] == code){
                start = d_value_2[key2];
                length = d_length_2[key2];
            }
        }
        for (int i = 0 ; i< length; i ++ ){
            unsigned int word_index = d_bands_index[IDX2C(start + i, w_index, vocab_size)];
            d_outputdist[IDX2C(word_index, batch_index, vocab_size)] = 1.0;
        }
    }
}



// for shrink the target vocab set;
// each block for a vocab id;
// <<<new_vocab_size, 256>>>
template<typename dType>
__global__
void shrink_vocab(dType *d_D_shrink, dType *d_D, dType *d_b_shrink, dType * d_b, int *d_new_vocab_index, int new_vocab_size, int vocab_size, int LSTM_size){
    int index = blockIdx.x;
    int vocab_index = d_new_vocab_index[index];
    d_b_shrink[index] = d_b[vocab_index];
    for (int i = threadIdx.x; i < LSTM_size; i += blockDim.x){
        d_D_shrink[IDX2C(index, i, new_vocab_size)] = d_D[IDX2C(vocab_index, i, vocab_size)];
    }
}







#endif














