//for statatisics on the neural MT model

namespace STATS {

	const bool record_stats = false;
	const bool clip_hidden_state = false;
	const bool clip_cell_state = false;

	//for clipping individual elements
	const bool clip_individual = false;
	precision individual_clip_threshold = 10;

	int number_hidden_clipped = 0;
	int total_hidden_clipped = 0;
	int number_cell_clipped = 0;
	int total_cell_clipped = 0;
	int number_grad_clipped = 0;
	int total_grad_clipped = 0;


	void print_model_stats() {
		std::cout << "\n--------------------------------------------\n";
		std::cout << "Number of hidden states clipped: " <<  << "  out of: " << total_hidden_clipped << "\n";
		std::cout << "\n--------------------------------------------\n";
	}
}