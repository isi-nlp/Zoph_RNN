#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "model.h"
#include "Eigen_Util.h"

template<typename dType>
class neuralMT_model;

template<typename dType>
class base_layer {
public:

	boost::random::mt19937 gen; //Random number generator for initializing weights
	neuralMT_model<precision> *model;

	template<typename Derived, typename Derived2>
	void check_gradient(dType epsilon,const Eigen::MatrixBase<Derived2> &parameter_const,const Eigen::MatrixBase<Derived> &grad);

	template<typename Derived>
	void initMatrix(const Eigen::MatrixBase<Derived> &input_const);

	void check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols);
};


#endif