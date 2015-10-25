#ifndef TRUNC_SOFTMAX_H
#define TRUNC_SOFTMAX_H




template<typename dType>
class trunc_softmax {
public:

	int output_vocab_size;
	int LSTM_size;
	int minibatch_size;

	int shortlist_size; //top most frequent words always being updates
	int sampled_size; //how many words to sample for each minibatch
	int trunc_size; //
	dType sample_correction;
	int shortlist_size_plus;//shortlist plus the unique words sampled in minibatch
	int cutoff; //At what index in the truncated softmax should the correct term be multiplied
	

	int *h_sampled_indices; //these are the unqiue words in minibatch + sampled indicies

	std::unordered_map<int,int> resevoir_mapping; //stores mapping for word in vocab to column in D matrix
	bool *bitmap;


	dType *d_D;
	dType *d_b_d;
	dType *d_D_samll; //temp embeddings loaded once per minibatch
	dType *d_b_d_small; //temp bias loaded once per minibatch
	dType *d_outputdist;
	dType *d_outputdist_small; 



};


void zero_bitmap() {
	memset(bitmap,0,output_vocab_size*sizeof(bool));
}

void prep_GPU_vocab_indices(int *h_output_vocab_indicies_target,int current_target_length) {

	//get the unique samples
	zero_bitmap();
	resevoir_mapping.clear();

	int curr_index = 0;
	for(int i=0; i<minibatch_size*current_target_length; i++) {

		if(bitmap[h_output_vocab_indicies_target[i]]==false && h_output_vocab_indicies_target[i] >= shortlist_size) {
			bitmap[h_output_vocab_indicies_target[i]] = true;
			h_sampled_indices[curr_index] = h_output_vocab_indicies_target[i];
			curr_index++;
		}
	}
	int len_unique_words_trunc_softmax = curr_index;

	if(curr_index > sampled_size) {
		std::cout << "ERROR: the sample size of the truncated softmax is too small\n";
		std::cout << "More unique words in the minibatch that there are sampled slots\n";
		exit (EXIT_FAILURE);
	}


	curr_index = 0;
	int num_to_sample = sampled_size - len_unique_words_trunc_softmax;
	boost::uniform_real<> distribution(0,1);
	for(int i=shortlist_size; i<output_vocab_size; i++) {
		if(bitmap[i]==false) {
			//fill the resevoir initially
			if(curr_index < num_to_sample) {
				h_sampled_indices[len_unique_words_trunc_softmax+curr_index] = i;
				curr_index++;
			}
			else {
				int rand_num = (int)(curr_index*distribution(BZ_CUDA::gen));
				
				if (rand_num <num_to_sample) {
					h_sampled_indices[len_unique_words_trunc_softmax+rand_num] = i;
				}
				curr_index++;
			}
		}
	}

	//get the mappings
	for(int i=0; i<sampled_size; i++) {
		resevoir_mapping[h_sampled_indices[i]] = i;
	}

	//get the sample correction
	sample_correction = ((dType)(output_vocab_size-shortlist_size-len_unique_words_trunc_softmax))/(sampled_size-len_unique_words_trunc_softmax);

	//get how many words are in the shortlist
	shortlist_size_plus = shortlist_size + len_unique_words_trunc_softmax;


	//load in the correct embeddings
	

	//load in the correct bias

}




#endif

