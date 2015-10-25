#ifndef GPU_INFO_STRUCT_H
#define GPU_INFO_STRUCT_H

struct layer_gpu_info {
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


	cudaEvent_t attention_forward; //this is gotten from the attention layer if feed input is true
	cudaEvent_t error_htild_below; //this is created here and shared with the attention layer

	//These are for synchronization for the backprop
	cudaEvent_t htm1_done;
	cudaEvent_t htm1_done_temp;
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

	cudaEvent_t h_t_below_transfer; //transfer h_t to upper layer
	cudaEvent_t dropout_done;

	cudaEvent_t d_ERR_ht_done;

	void init(int device_number) {
		this->device_number = device_number;
		cudaSetDevice(device_number);
		CUBLAS_ERROR_WRAPPER(cublasCreate(&handle),"CUBLAS handler initialization failed\n");

		cudaStreamCreate(&s0);
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);
		cudaStreamCreate(&s4);
		cudaStreamCreate(&s5);
		cudaStreamCreate(&s6);
		cudaStreamCreate(&s7);
		cudaStreamCreate(&s8);
		cudaStreamCreate(&s9);
		cudaStreamCreate(&s10);
		cudaStreamCreate(&s11);
		cudaStreamCreate(&s12);
		cudaStreamCreate(&s13);
		cudaStreamCreate(&s14);
		cudaStreamCreate(&s15);
		cudaStreamCreate(&s16);
		cudaStreamCreate(&s17);
		cudaStreamCreate(&s18);
		cudaStreamCreate(&s19);
		cudaStreamCreate(&s20);
		cudaStreamCreate(&s21);
		cudaStreamCreate(&s22);
		cudaStreamCreate(&s23);
		cudaStreamCreate(&s24);
		cudaStreamCreate(&s25);
		cudaStreamCreate(&s26);
		cudaStreamCreate(&s27);

		cudaEventCreate(&sparse_forward_start);
		cudaEventCreate(&i_t_part1);
		cudaEventCreate(&i_t_full);
		cudaEventCreate(&f_t_part1);
		cudaEventCreate(&f_t_full);
		cudaEventCreate(&c_prime_t_tanh_part1);
		cudaEventCreate(&c_prime_t_tanh_full);
		cudaEventCreate(&o_t_part1);
		cudaEventCreate(&o_t_full);
		cudaEventCreate(&W_grad_full_done);

		cudaEventCreate(&error_htild_below);

		cudaEventCreate(&backprop_init);
		cudaEventCreate(&err_ot_done);
		cudaEventCreate(&err_ft_done);
		cudaEventCreate(&err_tanhcpt_done);
		cudaEventCreate(&err_it_done);
		cudaEventCreate(&htm1_p1_done);
		cudaEventCreate(&htm1_p2_done);
		cudaEventCreate(&htm1_p3_done);
		cudaEventCreate(&htm1_p4_done);

		cudaEventCreate(&W_grad_p1_done);
		cudaEventCreate(&W_grad_p2_done);
		cudaEventCreate(&W_grad_p3_done);
		cudaEventCreate(&W_grad_p4_done);

		cudaEventCreate(&htm1_done);
		cudaEventCreate(&htm1_done_temp);
		cudaEventCreate(&ctm1_done);

		cudaEventCreate(&W_hi_grad_done);
		cudaEventCreate(&W_hf_grad_done);
		cudaEventCreate(&W_ho_grad_done);
		cudaEventCreate(&W_hc_grad_done);
		cudaEventCreate(&M_i_grad_done);
		cudaEventCreate(&M_f_grad_done);
		cudaEventCreate(&M_o_grad_done);
		cudaEventCreate(&M_c_grad_done);
		cudaEventCreate(&b_i_grad_done);
		cudaEventCreate(&b_f_grad_done);
		cudaEventCreate(&b_o_grad_done);
		cudaEventCreate(&b_c_grad_done);

		cudaEventCreate(&h_t_below_transfer);

		cudaEventCreate(&b_c_grad_done);

		cudaEventCreate(&dropout_done);

		cudaEventCreate(&d_ERR_ht_done);

		cudaSetDevice(0);
	}
};



struct softmax_layer_gpu_info {
	int device_number = 0;//this is for single GPU at the moment
	cublasHandle_t handle;
	cudaStream_t s0,s1,s2,s3;

	cudaEvent_t outputdist_done;
	cudaEvent_t d_ERR_ht_done;
	cudaEvent_t d_b_d_grad_done;
	cudaEvent_t d_D_grad_done;

	void init(int device_number) {
		this->device_number = device_number;
		cudaSetDevice(device_number);

		CUBLAS_ERROR_WRAPPER(cublasCreate(&handle),"CUBLAS handler initialization failed\n");
		cudaStreamCreate(&s0);
		cudaStreamCreate(&s1);
		cudaStreamCreate(&s2);
		cudaStreamCreate(&s3);

		cudaEventCreate(&outputdist_done);
		cudaEventCreate(&d_ERR_ht_done);
		cudaEventCreate(&d_D_grad_done);
		cudaEventCreate(&d_b_d_grad_done);

		cudaSetDevice(0);
	}
};



struct attention_layer_gpu_info {
	int device_number = 0;
	cudaStream_t s0;

	// cudaEvent_t ht_mat_done;
	// cudaEvent_t ct_mat_done;
	// cudaEvent_t ct_done;

	cudaEvent_t start_forward;
	cudaEvent_t start_backward;

	cudaEvent_t forward_prop_done;
	cudaEvent_t backward_prop_done;

	cudaEvent_t error_htild_below; //this is created here and shared with the attention layer

	// std::vector<cudaStream_t> alignment_streams; // (2*D+1) streams
	// std::vector<cudaEvent_t> alignment_events; // (2*D+1) streams

	void init(int device_number,int D) {
		this->device_number = device_number;
		cudaSetDevice(device_number);
		cudaStreamCreate(&s0);
		// cudaStreamCreate(&s1);
		// cudaStreamCreate(&s2);

		// cudaEventCreate(&ht_mat_done);
		// cudaEventCreate(&ct_mat_done);

		cudaEventCreate(&start_forward);
		cudaEventCreate(&start_backward);
		cudaEventCreate(&forward_prop_done);
		cudaEventCreate(&backward_prop_done);
		// for(int i=0; i<(2*D+1)*3; i++) {
		// 	cudaStream_t temp;
		// 	alignment_streams.push_back(temp);
		// 	cudaStreamCreate(&alignment_streams[alignment_streams.size()-1]);

		// 	cudaEvent_t temp_ev;
		// 	alignment_events.push_back(temp_ev);
		// 	cudaEventCreate(&alignment_events[alignment_events.size()-1]);
		// }
	}

};


#endif