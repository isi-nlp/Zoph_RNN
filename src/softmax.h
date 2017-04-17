#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <Eigen/Dense>
#include "Eigen_Util.h"
#include "gpu_info_struct.h"
#include "base_layer.h"
#include "softmax_node.h"
#include "transfer_layer.h"
#include "base_loss.h"
#include "LSH_WTA.h"

template<typename dType>
class softmax_layer : public base_loss_layer<dType> {
public:

	//Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRt_ht;



	//-----------------------------------------GPU parameters-------------------------------------------
	
	softmax_layer_gpu_info s_layer_info;

	//host pointers
	dType *h_D;
	dType *h_h_t;
	dType *h_b_d;
	dType *h_d_ERRt_ht;
	dType *h_ones;
	int *h_output_vocab_indices;
	int *h_output_vocab_indices_01;
	dType *h_D_grad;
	dType *h_output_vocab_indices_01_float;
	dType *h_b_d_grad;


	thrust::host_vector<dType> thrust_h_outputdist;
	thrust::host_vector<dType> thrust_h_normalization;

	thrust::device_vector<dType> thrust_d_outputdist;
	thrust::device_vector<dType> thrust_d_normalization;

	//device pointers
	dType *d_D; // declared in base class
	dType *d_h_t;
	dType *d_b_d;
	dType *d_d_ERRt_ht;
	dType *d_ones;
	int *d_output_vocab_indices;
	int *d_output_vocab_indices_01;
	dType *d_D_grad;
	dType *d_output_vocab_indices_01_float;
	dType *d_b_d_grad;
	dType *d_outputdist;
	dType *d_normalization;


	//trunacted softmax info
	int *h_truncated_vocab_mapping;//truncated softmax mapping for sampled indices
	int *d_truncated_vocab_mapping;
	bool truncated_softmax; //if using it, then true
	int shortlist_size;
	int sampled_size;
	int trunc_size;//
	dType sample_correction;
	int shortlist_size_plus;//shortlist plus the unique words sampled in minibatch
	int cutoff; //At what index in the truncated softmax should the correct term be multiplied
	dType *d_subset_D; //stores this for the shortlist + sampled vocabulary
	dType *h_subset_D; //stores this for the shortlist + sampled vocabulary
	dType *d_subset_D_grad;
	dType *h_subset_D_grad;
	dType *d_subset_b_d; //stores this for the shortlist + sampled vocabulary
	dType *h_subset_b_d; //stores this for the shortlist + sampled vocabulary
	dType *d_subset_b_d_grad; 
	dType *h_subset_b_d_grad;
	dType *h_subset_outputdist;
	dType *d_subset_outputdist;

	thrust::device_ptr<dType> thrust_d_subset_D_grad; 
	thrust::device_ptr<dType> thrust_d_subset_b_d_grad;

	//Sample the words for the truncated softmax
	void init_truncated_softmax();
	void prep_trunc(int *h_sampled_indices,int len_unique_words_trunc_softmax);

	double *d_train_perplexity;

	double *d_outputdist_perp;

	thrust::device_ptr<dType> thrust_d_D_grad; 
	thrust::device_ptr<dType> thrust_d_b_d_grad;

	//for norm clipping
	dType *d_result;
	dType *d_temp_result;

	//These are simply pointers to the non-single versions, since the full versions contain the indicies for the whole minibatch
	int *d_output_vocab_indices_single;
	int *d_output_vocab_indices_01_single;
	dType *d_output_vocab_indices_01_float_single;


	boost::random::mt19937 gen; //Random number generator for initializing weights

	bool clip_gradients; //If true then clip gradients
	dType norm_clip; //For gradient clipping
	int minibatch_size;
	int output_vocab_size; //declared in base class;
	int LSTM_size;
	dType learning_rate;
	bool scaled;

	//dropout stuff
	bool dropout;
	dType dropout_rate;

	neuralMT_model<precision> *model;

	bool train_perplexity;

	lower_transfer_layer<dType> lower_layer;

	std::vector<softmax_node<dType>> nodes;

	curandGenerator_t rand_gen;

    // for LSH
    int LSH_type = 0;
    LSH_WTA<dType> *lsh_wta;
    global_params * p_params;
    int nnz = 0;

    
	softmax_layer() {};

	void init_loss_layer(struct neuralMT_model<precision> *model,global_params &params); 

	void init_softmax_layer_GPU(int output_vocab_size,int minibatch_size,
	struct neuralMT_model<precision> *model,dType norm_clip,int LSTM_size, bool clip_gradients,dType learning_rate,int longest_sent);

	void clear_gradients();
	void clear_gradients_GPU();

	void forward_prop(int index);
	void forward_prop_GPU(int index);

	void back_prop1(int index);
	void back_prop1_GPU(int index);

	void back_prop2(int index);
	void back_prop2_GPU(int index);

	void update_weights();
	void update_weights_GPU();

	void calculate_global_norm();
	void update_global_params();

	void dump_weights(std::ofstream &output);
	void dump_weights_GPU(std::ofstream &output);

	void load_weights(std::ifstream &input);
	void load_weights_GPU(std::ifstream &input);

	void check_all_gradients(dType epsilon);
	void check_all_gradients_GPU(dType epsilon);

	void get_perplexity_GPU(dType *d_h_t,int index); 

	int stoic_generation(dType *h_outputdist,dType *d_outputdist,double temperature);

	void get_distribution_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D,dType *d_b_d,dType *d_h_t);

	void get_h_t_gradient_GPU(int output_vocab_size,dType *d_D,dType *d_outputdist,dType *d_d_ERRt_ht,int index);

	void compute_D_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D_grad,dType *d_h_t);

	void compute_b_d_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_b_d_grad);

	// //Non multithreaded, need to parallelize later
	// template<typename Derived,typename Derived2>
	// void compute_gradient(const Eigen::MatrixBase<Derived> &h_t,
	// 	const Eigen::MatrixBase<Derived2> &vocab_indicies,int index);

	void compute_gradient_GPU(int index);

	//void compute_gradient_GPU(int *h_output_vocab_indicies_target,int current_target_length);
	double compute_loss_GPU(int index);

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
	//convert to 0/1's and to indicies where there are no -1's
	void prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length);

	void backprop_prep_GPU(dType *d_h_t,int step);

	void backprop_prep_GPU_mgpu(int step);

	void dump_probs(std::ofstream &LSTM_dump_stream);

	void update_learning_rate(dType learning_rate);

	double get_train_perplexity();

	void get_distribution_GPU_decoder_wrapper();

	softmax_layer_gpu_info gpu_init(int device_number);

	void init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer);

	dType *get_ht_ptr(int index);

	void set_ht_ptr(int index,dType *d_h_t);

	cudaEvent_t get_ERR_ht_event();

	dType *get_dist_ptr();
    
    int get_nnz();
    
    int *get_h_rowIdx();

};

#endif
