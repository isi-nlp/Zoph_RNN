template<typename dType>
struct highway_node {

	//each node stores the unnormalized probabilities, plus the h_t
	dType *d_t; //gate value
	dType *d_y; //input
	dType *d_g; //new value from ReLU
	dType *d_z;//output
	dType *d_Err_z;//output error being passed in
	int index;

	highway_node(int state_size,int minibatch_size,int index) {
		
		this->index = index;
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_t, state_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_y, state_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_g, state_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_z, state_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
		CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_Err_z, state_size*minibatch_size*sizeof(dType)),"GPU memory allocation failed\n");
	}
};