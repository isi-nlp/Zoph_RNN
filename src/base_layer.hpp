template<typename dType>
template<typename Derived>
void base_layer<dType>::initMatrix(const Eigen::MatrixBase<Derived> &input_const) {
	UNCONST(Derived,input_const,input);
	dType lower = -1.0; //Lower bound for uniform dist
	dType upper = 1.0; //Upper bound for uniform dist
	boost::uniform_real<> distribution(lower,upper);
	for(int j=0; j<input.cols(); j++) {
		for(int i=0; i<input.rows(); i++) {
			input(i,j) =  distribution(gen);
		}
	}
}

template<typename dType>
template<typename Derived,typename Derived2>
void base_layer<dType>::check_gradient(dType epsilon,const Eigen::MatrixBase<Derived2> &parameter_const,const Eigen::MatrixBase<Derived> &grad) 
{
	UNCONST(Derived2, parameter_const, parameter);
	for(int i=0; i<grad.rows(); i++) {
		for(int j=0; j<grad.cols(); j++) {
			dType loss = 0;
			parameter(i,j)+= epsilon;
			loss = model->getError(false);
			parameter(i,j)+= -2*epsilon;
			loss-= model->getError(false);
			parameter(i,j)+= epsilon;
			if( (std::abs(grad(i,j) - loss/(2.0*epsilon))) > 1/1000.0 ||  (std::abs(grad(i,j) - loss/(2*epsilon))/(std::abs(grad(i,j)) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << grad(i,j) << "\n";
				std::cout << "Gradient difference: " << std::abs(grad(i,j) - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(grad(i,j) - loss/(2.0*epsilon))/(std::abs(grad(i,j)) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}

template<typename dType>
void base_layer<dType>::check_gradient_GPU(dType epsilon,dType *d_mat,dType *d_grad,int rows,int cols) {
	cudaDeviceSynchronize();
	thrust::device_ptr<dType> d_thrust_mat = thrust::device_pointer_cast(d_mat);
	thrust::device_ptr<dType> d_thrust_grad = thrust::device_pointer_cast(d_grad);
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			dType loss =0;
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			loss = model->getError(true);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= -2*epsilon;
			loss -=model->getError(true);
			cudaDeviceSynchronize();
			d_thrust_mat[IDX2C(i,j,rows)]+= epsilon;
			std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
			if( (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))) > 1/(dType)1000.0 ||  (std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)))) > 1/1000.0  ) {
				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
				std::cout << "My gradient: " << d_thrust_grad[IDX2C(i,j,rows)] << "\n";
				std::cout << "Gradient difference: " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon)) << "\n";
				std::cout << "Gradient difference (Equation 2): " << std::abs(d_thrust_grad[IDX2C(i,j,rows)] - loss/(2*epsilon))/(std::abs(d_thrust_grad[IDX2C(i,j,rows)]) + std::abs(loss/(2*epsilon)) ) << "\n\n";
			}
		}
	}
}