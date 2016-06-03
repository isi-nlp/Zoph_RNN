#ifndef NCE_H
#define NCE_H

#include "multinomial.h"
#include <algorithm> 
#include <fstream>
#include <unordered_map>
#include "NCE_node.h"

template<typename dType>
class neuralMT_model;


template<typename dType>
class NCE_layer : public base_loss_layer<dType> {
public:

	softmax_layer_gpu_info s_layer_info;

	neuralMT_model<precision> *model;

	int LSTM_size;
	int minibatch_size;
	int output_vocab_size;
	int num_negative_samples;
	int longest_sent;
	dType learning_rate;
	bool dropout;
	dType dropout_rate;
	bool clip_gradients; //If true then clip gradients
	dType norm_clip; //For gradient clipping

	int curr_num_unique = 0; //for the current minibatch, how many unqiue samples are there

	bool share_samples = true; //share the noise samples across the minibatch

	multinomial<long long int,double> unigram;

	dType *d_D; //For NCE this is embedding size by output vocab size, while in softmax it is the other way around
	dType *d_b_d;
	dType *d_dot_products; //stores dot product along with the embedding
	dType *d_outputdist;
 	dType *d_b_d_grad;
 	dType *d_ones; // 1xminibatch size
 	dType *d_temp_b_d_grad; //1 x ( num negative samples + minibatchsize )
 	dType *d_temp_result;
 	dType *d_result;
 	dType *d_h_t;


 	//use curr_num_unique to get the length of this during training
 	//use d_unique_indicies for mapping
 	dType *d_small_D_grad; //this is (num neg samples + minibatchsize)*LSTM size*longestsent
 	int *d_reverse_unique_indicies; //for each possible vocab index
 	thrust::device_ptr<dType> thrust_d_small_D_grad;


 	thrust::device_ptr<dType> thrust_d_b_d_grad;


 	double *d_OBJ_val_temp;
 	double *d_final_NCE_OBJ;

 	dType *d_temp_D_grad;

 	lower_transfer_layer<dType> lower_layer;


	int *h_vocab_indicies;
	int *d_vocab_indicies; //stored as [negative samples][positive words] [negative samples][positive words] ... for the current larget length
	int *h_unique_indicies;
	int *d_unique_indicies;
	int *h_vocab_indicies_01;
	int *d_vocab_indicies_01;
	int *d_vocab_indicies_nonneg;

	int *d_vocab_indicies_single;
	int *d_vocab_indicies_01_single;
	int *d_vocab_indicies_nonneg_single;

	double *h_partition_vals;
	double *d_partition_vals;

	dType *d_reductuction_space;

	//these are for the inputs
	int *d_output_vocab_indices;

	dType *h_sampling_probs; //stores the log (k*Q(w)) 
	dType *d_sampling_probs; //same format as d_vocab_indicies
	dType *d_sampling_probs_single;

	curandGenerator_t rand_gen;

	std::vector<NCE_Node<dType>> nodes;



	NCE_layer() {}

	void init_loss_layer(struct neuralMT_model<precision> *model,global_params &params);

	//this will compute all of the negative samples for a minibatch
	void get_unigram_counts(std::vector<long long int> &unigram_counts,std::string file_name);

	//prep gpu indicies
	void prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length);
	void prep_GPU_vocab_indices_shared_samples(int *h_output_vocab_indicies_target,int current_target_length);
	void prep_GPU_vocab_indices_nonshared_samples(int *h_output_vocab_indicies_target,int current_target_length);

	void forward_prop(int index);

	void back_prop1(int index);
	void back_prop2(int index);

	void calculate_global_norm();
	void update_global_params();

	double compute_loss_GPU(int index);

	void clear_gradients();

	void update_weights();

	void check_all_gradients(dType epsilon);

	void dump_weights(std::ofstream &output);

	void load_weights(std::ifstream &input);

	void backprop_prep_GPU(dType *d_h_t,int step);

	void backprop_prep_GPU_mgpu(int step);

	void update_learning_rate(dType learning_rate);

	softmax_layer_gpu_info gpu_init(int device_number);

	void init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer);

	dType *get_ht_ptr(int index);

	void set_ht_ptr(int index,dType *d_h_t);

	cudaEvent_t get_ERR_ht_event();

	dType *get_dist_ptr();

	void get_perplexity(dType *d_h_t);

	double get_train_perplexity();

	void get_distribution_GPU_decoder_wrapper();

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);

	void check_gradient_GPU_SPARSE(dType epsilon,dType *d_mat,dType *d_grad,int LSTM_size,int *d_unique_indicies,int curr_num_unique);

};



#endif
