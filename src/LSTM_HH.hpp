
//Constructor
LSTM_HH_Node::LSTM_HH_Node(int LSTM_size,int minibatch_size,int vocab_size,struct Hidden_To_Hidden_Layer *m) {

	i_t.resize(LSTM_size,minibatch_size);
	f_t.resize(LSTM_size,minibatch_size);
	c_t.resize(LSTM_size,minibatch_size);
	c_prime_t_tanh.resize(LSTM_size,minibatch_size); //This is the tanh of c
	o_t.resize(LSTM_size,minibatch_size);
	h_t.resize(LSTM_size,minibatch_size);

	//Needed for forward propagation
	//These have dimension (hidden state size)x(size of minibatch)
	h_t_prev.resize(LSTM_size,minibatch_size); //h_(t-1)
	c_t_prev.resize(LSTM_size,minibatch_size); //c_(t-1)

	h_t_prev_b.resize(LSTM_size,minibatch_size); //h_(t-1)

	//This is the length of the minibatch
	//Each index ranges from 0 to (output vocab size-1), in order to select matrix column from embedding layer
	//This is for softmax layer
	vocab_indices_output.resize(minibatch_size,1);

	//This is the derivative of the error from time n to time t with respect to h_t
	//Has size (minibatch size)x(hidden state size)
	d_ERRnTOt_ht.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_ot.resize(minibatch_size,LSTM_size);

	d_ERRt_ct.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_ft.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_tanhcpt.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_it.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_htM1.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_ctM1.resize(minibatch_size,LSTM_size);

	d_ERRnTOt_htM1_b.resize(minibatch_size,LSTM_size);

	Z_i.resize(minibatch_size,LSTM_size);
	Z_f.resize(minibatch_size,LSTM_size);
	Z_c.resize(minibatch_size,LSTM_size);
	Z_o.resize(minibatch_size,LSTM_size);

	model = m;

}



//Update the hidden state and cell state vectors
template<typename Derived>
void LSTM_HH_Node::update_vectors_forward(const Eigen::MatrixBase<Derived> &h_prev,
	const Eigen::MatrixBase<Derived> &c_prev,const Eigen::MatrixBase<Derived> &h_prev_b) 
{
	h_t_prev = h_prev;
	c_t_prev = c_prev;
	h_t_prev_b = h_prev_b;
}

//Update the output vocab indicies
template<typename Derived>
void LSTM_HH_Node::update_vectors_backward(const Eigen::MatrixBase<Derived> &vocab,int index) {
	vocab_indices_output = vocab.col(index);
}

//Compute the forward values for the LSTM node
//This is after the node has recieved the previous hidden and cell state values
void LSTM_HH_Node::forward_prop() {
	//Input gate

	//input gate
	i_t = ((model->W_hi_b*h_t_prev_b + model->W_hi*h_t_prev).colwise() + model->b_i).array().unaryExpr(sigmoid_functor());

	//Forget gate
	f_t = ((model->W_hf_b*h_t_prev_b + model->W_hf*h_t_prev).colwise() + model->b_f).array().unaryExpr(sigmoid_functor());

	//Cell gate
	c_prime_t_tanh = ((model->W_hc_b*h_t_prev_b + model->W_hc*h_t_prev).colwise() + model->b_c).array().unaryExpr(tanh_functor());

	c_t = ((f_t.array())*(c_t_prev.array())).matrix() + (i_t.array()*(c_prime_t_tanh.array())).matrix();

	//Output gate
	o_t = ((model->W_ho_b*h_t_prev_b + model->W_ho*h_t_prev).colwise() + model->b_o).unaryExpr(sigmoid_functor());

	//Output hidden state
	h_t = o_t.array()*(c_t.array().unaryExpr(tanh_functor()));

	//Now do a check to see if h_t or c_t should be zeroed out
	for(int i=0; i< vocab_indices_output.rows(); i++) {
		if(vocab_indices_output(i)==-1) {
			h_t.col(i).setZero();
			c_t.col(i).setZero();
		}
	}
}

//Computes errors for this LSTM node
template<typename Derived>
void LSTM_HH_Node::back_prop(const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ht,const Eigen::MatrixBase<Derived> &d_ERRnTOtp1_ct,
	const Eigen::MatrixBase<Derived> &d_ERRt_ht) {

	//Now get the derivative of h_t with respect to this error and all after it (t-n)
	d_ERRnTOt_ht = d_ERRnTOtp1_ht + d_ERRt_ht;

	//Derivative of error at time t with respect to c_t
	d_ERRt_ct = d_ERRnTOt_ht.array() * (o_t.array()*(1- (c_t).array().unaryExpr(tanh_sq_functor()))).matrix().transpose().array();

	d_ERRnTOt_ct = d_ERRnTOtp1_ct + d_ERRt_ct;

	//Check to see if we should zero out derivatives
	//Now do a check to see if h_t or c_t should be zeroed out
	for(int i=0; i< vocab_indices_output.rows(); i++) {
		if(vocab_indices_output(i)==-1) {
			d_ERRnTOt_ht.row(i).setZero();
			d_ERRnTOt_ct.row(i).setZero();
		}
	}

	//Derivative of error from time t to n with respect to o_t
	d_ERRnTOt_ot = d_ERRnTOt_ht.array()*( (c_t.array().unaryExpr(tanh_functor())).matrix().transpose().array() );

	//Derivative of Error from t to n with respect to f_t
	d_ERRnTOt_ft = d_ERRnTOt_ct.array()*(c_t_prev.transpose().array());

	//This is the derivative of the error from time n to time t with respect to tanhc'_t
	d_ERRnTOt_tanhcpt = d_ERRnTOt_ct.array()*(i_t.transpose().array());

	//This is the derivative of the error from time n to time t with respect to i_t
	d_ERRnTOt_it = d_ERRnTOt_ct.array()*(c_prime_t_tanh.transpose().array());

	//This is the derivative of the error from time n to t with respect to h_(t-1)
	d_ERRnTOt_htM1 = (model->W_ho.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() )).transpose() \
	+ (model->W_hf.transpose()*((d_ERRnTOt_ft.transpose().array() * f_t.array() *(1-f_t.array())).matrix())).transpose() \
	+ (model->W_hi.transpose()*((d_ERRnTOt_it.transpose().array()*i_t.array()*(1-i_t.array())).matrix())).transpose() \
	+ (model->W_hc.transpose()*((d_ERRnTOt_tanhcpt.transpose().array()*(1-c_prime_t_tanh.array().square())).matrix())).transpose();

	//Gets passed to the LSTM block below it
	d_ERRnTOt_htM1_b = (model->W_ho_b.transpose()*( (d_ERRnTOt_ot.transpose().array() * o_t.array() * (1- o_t.array())).matrix() )).transpose() \
	+ (model->W_hf_b.transpose()*((d_ERRnTOt_ft.transpose().array() * f_t.array() *(1-f_t.array())).matrix())).transpose() \
	+ (model->W_hi_b.transpose()*((d_ERRnTOt_it.transpose().array()*i_t.array()*(1-i_t.array())).matrix())).transpose() \
	+ (model->W_hc_b.transpose()*((d_ERRnTOt_tanhcpt.transpose().array()*(1-c_prime_t_tanh.array().square())).matrix())).transpose();

	//Derivative from error from time t to n with respect to ctM1
	d_ERRnTOt_ctM1 = (d_ERRnTOt_ct.array()*f_t.transpose().array());

	//Update the gradients
	update_gradients();
}


void LSTM_HH_Node::update_gradients() {

	//Hiden state matrices
	model->W_hi_grad.noalias() += (h_t_prev*(d_ERRnTOt_it.array() * i_t.transpose().array()*(1-i_t.transpose().array())).matrix()).transpose();
	model->W_hf_grad.noalias() += (h_t_prev*(d_ERRnTOt_ft.array()*f_t.transpose().array()*(1-f_t.transpose().array())).matrix()).transpose();
	model->W_hc_grad.noalias() += (h_t_prev*(d_ERRnTOt_ct.array()*(i_t.transpose().array())*(1-c_prime_t_tanh.transpose().array().square())).matrix()).transpose();
	model->W_ho_grad.noalias() += (h_t_prev*(d_ERRnTOt_ot.array()*o_t.transpose().array()*(1-o_t.transpose().array())).matrix()).transpose();

	model->W_hi_b_grad.noalias() += (h_t_prev_b*(d_ERRnTOt_it.array() * i_t.transpose().array()*(1-i_t.transpose().array())).matrix()).transpose();
	model->W_hf_b_grad.noalias() += (h_t_prev_b*(d_ERRnTOt_ft.array()*f_t.transpose().array()*(1-f_t.transpose().array())).matrix()).transpose();
	model->W_hc_b_grad.noalias() += (h_t_prev_b*(d_ERRnTOt_ct.array()*(i_t.transpose().array())*(1-c_prime_t_tanh.transpose().array().square())).matrix()).transpose();
	model->W_ho_b_grad.noalias() += (h_t_prev_b*(d_ERRnTOt_ot.array()*o_t.transpose().array()*(1-o_t.transpose().array())).matrix()).transpose();

	//Update the bias gradients
	model->b_i_grad.noalias() += ((d_ERRnTOt_it.array() * (i_t.array() * (1-i_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	model->b_f_grad.noalias() += ((d_ERRnTOt_ft.array() * (f_t.array() * (1-f_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
	model->b_c_grad.noalias() += (d_ERRnTOt_tanhcpt.array() * (1-c_prime_t_tanh.array().square()).matrix().transpose().array()).colwise().sum().matrix().transpose();
	model->b_o_grad.noalias() += ((d_ERRnTOt_ot.array() * (o_t.array() * (1-o_t.array())).matrix().transpose().array()).colwise().sum()).matrix().transpose();
}




