#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "model.h"
#include "Eigen_Util.h"

template<typename dType>
class neuralMT_model;

template<typename dType>
class base_layer {
public:

	dType *h_h_t_below;
	dType *d_h_t_below;
	
};


#endif