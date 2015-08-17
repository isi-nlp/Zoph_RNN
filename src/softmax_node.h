#ifndef SOFTMAX_NODE_H
#define SOFTMAX_NODE_H

//for multigpu training
template<typename dType>
struct softmax_node {

	//each node stores the unnormalized probabilities
	dType h_outputdist;
	dType d_outputdist;
};

#endif