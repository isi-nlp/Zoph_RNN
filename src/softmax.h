#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <Eigen/Dense>
#include "Eigen_Util.h"
#include "gpu_info_struct.h"
#include "base_layer.h"

template<typename dType>
class softmax_layer {
public:
	//Vector that contains the distribution over the entire output vocabulary
	//Size (output vocab size)x(minibatch size)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> outputDist;

	//This is the normalization constant for the softmax
	//Size is 1x(minibatch size)
	Eigen::Matrix<dType, 1, Eigen::Dynamic> normalization;

	//This is the derivative of the error at time t with respect to h_t
	//Has size (minibatch size)x(hidden state size)
	Eigen::Matrix<dType, Eigen::Dynamic, Eigen::Dynamic> d_ERRt_ht;

	//Parameters needed to connect hidden to output layer
	//Dimension (output word vocab)x(hidden state size)
	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> D;

	//bias for the output embedding layer
	//Dimension (output vocab size)x(1)
	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_d;

	//Parameters needed to connect hidden to output layer
	//Dimension (output word vocab)x(hidden state size)
	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic> D_grad;

	Eigen::Matrix<dType, Eigen::Dynamic, 1> b_d_grad;


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
	dType *d_D;
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
	int output_vocab_size;
	int LSTM_size;
	dType learning_rate;
	bool scaled;

	neuralMT_model<precision> *model;

	bool train_perplexity;

	void init_softmax_layer(int output_vocab_size,int minibatch_size,
		struct neuralMT_model<precision> *model,dType norm_clip,int LSTM_size, bool clip_gradients,dType learning_rate,
		int longest_sent,bool scaled,bool train_perplexity,bool truncated_softmax,int shortlist_size,int sampled_size); 

	void init_softmax_layer_CPU(int output_vocab_size,int minibatch_size,
	struct neuralMT_model<precision> *model,dType norm_clip,int LSTM_size, bool clip_gradients,dType learning_rate);

	void init_softmax_layer_GPU(int output_vocab_size,int minibatch_size,
	struct neuralMT_model<precision> *model,dType norm_clip,int LSTM_size, bool clip_gradients,dType learning_rate,int longest_sent);

	void clear_normalization();

	void clear_gradients();
	void clear_gradients_CPU();
	void clear_gradients_GPU();

	void update_weights();
	void update_weights_CPU();
	void update_weights_GPU();

	void dump_weights(std::ofstream &output);
	void dump_weights_CPU(std::ofstream &output);
	void dump_weights_GPU(std::ofstream &output);

	void load_weights(std::ifstream &input);
	void load_weights_CPU(std::ifstream &input);
	void load_weights_GPU(std::ifstream &input);

	void check_all_gradients(dType epsilon);
	void check_all_gradients_CPU(dType epsilon);
	void check_all_gradients_GPU(dType epsilon);

	void get_perplexity_GPU(); 

	int stoic_generation(dType *h_outputdist,dType *d_outputdist,double temperature);

	//For softmax calculation
	template <typename Derived>
	void softmax_calc(const Eigen::MatrixBase<Derived> &h_t);

	//gets the distribution over the current word, given the current hidden vector
	//naive implementation, parallelize later
	template<typename Derived>
	void getDist(const Eigen::MatrixBase<Derived> &h_t);

	void get_distribution_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D,dType *d_b_d);

	void get_h_t_gradient_GPU(int output_vocab_size,dType *d_D,dType *d_outputdist);

	void compute_D_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_D_grad);

	void compute_b_d_gradient_GPU(int output_vocab_size,dType *d_outputdist,dType *d_b_d_grad);

	//Non multithreaded, need to parallelize later
	template<typename Derived,typename Derived2>
	void compute_gradient(const Eigen::MatrixBase<Derived> &h_t,
		const Eigen::MatrixBase<Derived2> &vocab_indicies);

	template<typename Derived,typename Derived2>
	void compute_gradient_CPU(const Eigen::MatrixBase<Derived> &h_t,
		const Eigen::MatrixBase<Derived2> &vocab_indicies);

	void compute_gradient_GPU();

	//void compute_gradient_GPU(int *h_output_vocab_indicies_target,int current_target_length);

	template<typename Derived,typename Derived2>
	double compute_loss(const Eigen::MatrixBase<Derived> &h_t,const Eigen::MatrixBase<Derived2> &vocab_indicies);

	template<typename Derived,typename Derived2>
	double compute_loss_CPU(const Eigen::MatrixBase<Derived> &h_t,const Eigen::MatrixBase<Derived2> &vocab_indicies);

	double compute_loss_GPU();

	template<typename Derived,typename Derived2>
	void compute_D_gradient(const Eigen::MatrixBase<Derived> &h_t_const,const Eigen::MatrixBase<Derived2> &vocab_indicies);

	template<typename Derived,typename Derived2>
	void compute_b_d_gradient(const Eigen::MatrixBase<Derived> &h_t_const,const Eigen::MatrixBase<Derived2> &vocab_indicies);

	template<typename Derived>
	void initMatrix(const Eigen::MatrixBase<Derived> &input_const);

	template<typename Derived,typename Derived3>
	void check_gradient(dType epsilon,const Eigen::MatrixBase<Derived3> &parameter_const,const Eigen::MatrixBase<Derived> &grad);

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
	//convert to 0/1's and to indicies where there are no -1's
	void prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length);

	//Get h_t from the previous layer, which lies on a different GPU
	void get_h_t_DMA(dType *d_h_t); //This transfers from GPU to GPU going from device(0) (input layer) to device 1 (softmax)

	void get_h_t_CPU(dType *d_h_t); //transfers this memory to CPU then to the GPU device(1) where the softmax parameters live

	void get_h_t_NOTHING(dType *d_h_t); //no transfers necessary because all the stuff lies on 1 GPU

	void backprop_prep_GPU(dType *d_h_t,int *d_output_vocab_indices_single,int *d_output_vocab_indices_01_single,
		dType *d_output_vocab_indices_01_float_single);

	void dump_probs(std::ofstream &LSTM_dump_stream);
};

#endif