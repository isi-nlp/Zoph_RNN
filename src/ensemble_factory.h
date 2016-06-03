#ifndef ENSEMBLE_FACTORY_H
#define ENSEMBLE_FACTORY_H

#include "file_helper_decoder.h"
#include "decoder.h"

template<typename dType>
class ensemble_factory {
public:
	std::vector< decoder_model_wrapper<dType> > models;
	decoder_model_wrapper<dType> models_2;
	//file_helper_decoder *fileh; //for file input
	decoder<dType> *model_decoder; //pass the output dists to this

	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> outputdist;
	Eigen::Matrix<dType,Eigen::Dynamic, Eigen::Dynamic> normalization;

	int num_lines_in_file; //how many lines in the decoder file
	int target_vocab_size; //target vocabulary size, must agree on all models
	int longest_sent; //set a max to the longest sentence that could be decoded by the decoder
	dType max_decoding_ratio;


	//these must be fixed
	const int start_symbol = 0;
	const int end_symbol = 1;

	ensemble_factory(std::vector<std::string> weight_file_names,int num_hypotheses,int beam_size, dType min_decoding_ratio,\
		dType penalty, int longest_sent,bool print_score,std::string decoder_output_file,
		std::vector<int> gpu_nums,dType max_decoding_ratio, int target_vocab_size,global_params &params);
	void init_index_swapping(); //pass in the master for swapping around
	void decode_file();
	void ensembles_models();
	void get_target_vocab(std::string file_name);

};



#endif