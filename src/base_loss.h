
//base loss class, so MLE,NCE, etc ...
#ifndef BASE_LOSS_H
#define BASE_LOSS_H

template<typename dType>
class Hidden_To_Hidden_Layer;

template<typename dType>
class Input_To_Hidden_Layer;

template<typename dType>
class base_loss_layer {
public:


	virtual softmax_layer_gpu_info gpu_init(int device_number) = 0;

	virtual void init_loss_layer(struct neuralMT_model<precision> *model,global_params &params) = 0;
	virtual void forward_prop(int index) = 0;
	virtual void back_prop1(int index) = 0; //this is done for multi GPU paralleism
	virtual void back_prop2(int index) = 0;

	virtual void backprop_prep_GPU(dType *d_h_t,int step) = 0;

	virtual void backprop_prep_GPU_mgpu(int step) = 0;

	virtual void prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length) = 0;

	virtual void update_weights() = 0;
	virtual void clear_gradients() = 0;

	virtual double compute_loss_GPU(int index) = 0;

	virtual void calculate_global_norm() = 0;
	virtual void update_global_params() = 0;

	virtual void check_all_gradients(dType epsilon) = 0;

	virtual void update_learning_rate(dType learning_rate) = 0;


	virtual void init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer)=0;

	virtual dType *get_ht_ptr(int index)=0;

	virtual void set_ht_ptr(int index,dType *d_h_t)=0;

	virtual cudaEvent_t get_ERR_ht_event() = 0;

	virtual void load_weights(std::ifstream &input) = 0;

	virtual void dump_weights(std::ofstream &output) = 0;

	virtual double get_train_perplexity() = 0;

	virtual void get_distribution_GPU_decoder_wrapper() = 0;

	virtual dType *get_dist_ptr() = 0;
};





#endif