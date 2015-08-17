#ifndef GPU_INFO_STRUCT_H
#define GPU_INFO_STRUCT_H

struct input_layer_gpu_info {
	int device_number = 0;//Input layer always gets device 0
	cublasHandle_t handle;
	//streams are shared for forward and back prop
	cudaStream_t s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,
		s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27;

	//forward prop events
	cudaEvent_t sparse_forward_start;
	cudaEvent_t i_t_part1,i_t_full;
	cudaEvent_t f_t_part1,f_t_full;
	cudaEvent_t c_prime_t_tanh_part1,c_prime_t_tanh_full;
	cudaEvent_t o_t_part1,o_t_full;

	//backprop events
	cudaEvent_t backprop_init;
	cudaEvent_t err_ot_done;
	cudaEvent_t err_ft_done;
	cudaEvent_t err_tanhcpt_done;
	cudaEvent_t err_it_done;

	cudaEvent_t htm1_p1_done;
	cudaEvent_t htm1_p2_done;
	cudaEvent_t htm1_p3_done;
	cudaEvent_t htm1_p4_done;

	cudaEvent_t W_grad_p1_done;
	cudaEvent_t W_grad_p2_done;
	cudaEvent_t W_grad_p3_done;
	cudaEvent_t W_grad_p4_done;

	//These are for synchronization for the backprop
	cudaEvent_t htm1_done;
	cudaEvent_t ctm1_done;
	cudaEvent_t W_grad_full_done;
	cudaEvent_t W_hi_grad_done;
	cudaEvent_t W_hf_grad_done;
	cudaEvent_t W_ho_grad_done;
	cudaEvent_t W_hc_grad_done;
	cudaEvent_t M_i_grad_done;
	cudaEvent_t M_f_grad_done;
	cudaEvent_t M_o_grad_done;
	cudaEvent_t M_c_grad_done;
	cudaEvent_t b_i_grad_done;
	cudaEvent_t b_f_grad_done;
	cudaEvent_t b_o_grad_done;
	cudaEvent_t b_c_grad_done;

};

struct softmax_layer_gpu_info {
	int device_number = 0;//this is for single GPU at the moment
	cublasHandle_t handle;
	cudaStream_t s0,s1,s2,s3;

	cudaEvent_t outputdist_done;
	cudaEvent_t d_ERR_ht_done;
	cudaEvent_t d_b_d_grad_done;
	cudaEvent_t d_D_grad_done;
};


#endif