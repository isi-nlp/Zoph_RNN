

//for decoding multilayer models
template<typename dType>
struct prev_source_state {
	dType *d_h_t_prev;
	dType *d_c_t_prev;

	prev_source_state(int LSTM_size) {
		cudaMalloc((void**)&d_h_t_prev, LSTM_size*1*sizeof(dType));
		cudaMalloc((void**)&d_c_t_prev, LSTM_size*1*sizeof(dType));
	}
};


template<typename dType>
struct prev_target_state {
	dType *d_h_t_prev;
	dType *d_c_t_prev;

	prev_target_state(int LSTM_size,int beam_size) {
		cudaMalloc((void**)&d_h_t_prev, LSTM_size*beam_size*sizeof(dType));
		cudaMalloc((void**)&d_c_t_prev, LSTM_size*beam_size*sizeof(dType));
	}
};

