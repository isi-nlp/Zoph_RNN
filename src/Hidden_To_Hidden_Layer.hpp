
// void Hidden_To_Hidden_Layer::init_Hidden_To_Hidden_Layer(int LSTM_size,int minibatch_size,int vocab_size,
//  		int longest_sent,bool debug_temp,double learning_rate,bool clip_gradients,double norm_clip,struct neuralMT_model<precision> *model)
// {

// 	//Initialize all of the weights
// 	W_hi.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_hi);
// 	b_i.resize(LSTM_size,1);
// 	initMatrix(b_i);

// 	W_hf.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_hf);
// 	b_f.resize(LSTM_size,1);
// 	initMatrix(b_f);

// 	W_hc.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_hc);
// 	b_c.resize(LSTM_size,1);
// 	initMatrix(b_c);

// 	W_ho.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_ho);
// 	b_o.resize(LSTM_size,1);
// 	initMatrix(b_o);

// 	W_hi_b.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_hi_b);
// 	W_hf_b.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_hf_b);
// 	W_hc_b.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_hc_b);
// 	W_ho_b.resize(LSTM_size,LSTM_size);
// 	initMatrix(W_ho_b);

	
// 	//Initialize the gradients here
// 	W_hi_grad.setZero(LSTM_size,LSTM_size);
// 	b_i_grad.setZero(LSTM_size,1);

// 	W_hf_grad.setZero(LSTM_size,LSTM_size);
// 	b_f_grad.setZero(LSTM_size,1);

// 	W_hc_grad.setZero(LSTM_size,LSTM_size);
// 	b_c_grad.setZero(LSTM_size,1);

// 	W_ho_grad.setZero(LSTM_size,LSTM_size);
// 	b_o_grad.setZero(LSTM_size,1);

// 	W_hi_b_grad.setZero(LSTM_size,LSTM_size);
// 	W_hf_b_grad.setZero(LSTM_size,LSTM_size);
// 	W_hc_b_grad.setZero(LSTM_size,LSTM_size);
// 	W_ho_b_grad.setZero(LSTM_size,LSTM_size);

// 	//Initalize the initial hidden state and cell state vector
// 	init_hidden_vector.setZero(LSTM_size,minibatch_size);
// 	init_cell_vector.setZero(LSTM_size,minibatch_size);

// 	//Initialize the messages being passed into first cells for backprop
// 	init_d_ERRnTOtp1_ht.setZero(minibatch_size,LSTM_size); //Initial hidden state vector
// 	init_d_ERRnTOtp1_ct.setZero(minibatch_size,LSTM_size); //Initial hidden state vector

// 	//Initialize the vector of LSTM nodes to longest sentence
// 	for(int i=0;i < longest_sent; i++) {
// 		nodes.push_back(LSTM_HH_Node(LSTM_size,minibatch_size,vocab_size,this));
// 	}

// 	//Set the debug mode
// 	debug = debug_temp;
// 	this->minibatch_size = minibatch_size;
// 	this->learning_rate = learning_rate;
// 	this->clip_gradients = clip_gradients;
// 	this->norm_clip = norm_clip;
// 	this->model = model;
// 	gen.seed(111);
// }

// void Hidden_To_Hidden_Layer::clear_gradients() {

// 	W_hi_grad.setZero();
// 	b_i_grad.setZero();

// 	W_hf_grad.setZero();
// 	b_f_grad.setZero();

// 	W_hc_grad.setZero();
// 	b_c_grad.setZero();

// 	W_ho_grad.setZero();
// 	b_o_grad.setZero();

// 	W_hi_b_grad.setZero();
// 	W_hf_b_grad.setZero();
// 	W_hc_b_grad.setZero();
// 	W_ho_b_grad.setZero();
// }

// //Update the model parameters
// void Hidden_To_Hidden_Layer::update_weights() {

// 	W_hi_b_grad = (1.0/minibatch_size)*W_hi_b_grad;
// 	W_hi_grad = (1.0/minibatch_size)*W_hi_grad;
// 	b_i_grad = (1.0/minibatch_size)*b_i_grad;

// 	W_hf_b_grad = (1.0/minibatch_size)*W_hf_b_grad;
// 	W_hf_grad = (1.0/minibatch_size)*W_hf_grad;
// 	b_f_grad = (1.0/minibatch_size)*b_f_grad;

// 	W_hc_b_grad = (1.0/minibatch_size)*W_hc_b_grad;
// 	W_hc_grad = (1.0/minibatch_size)*W_hc_grad;
// 	b_c_grad = (1.0/minibatch_size)*b_c_grad;

// 	W_ho_b_grad = (1.0/minibatch_size)*W_ho_b_grad;
// 	W_ho_grad = (1.0/minibatch_size)*W_ho_grad;
// 	b_o_grad = (1.0/minibatch_size)*b_o_grad;


// 	//For gradient clipping
// 	if(clip_gradients) {
// 		computeNorm(W_hi_grad,norm_clip);
// 		computeNorm(b_i_grad,norm_clip);

// 		computeNorm(W_hf_grad,norm_clip);
// 		computeNorm(b_f_grad,norm_clip);

// 		computeNorm(W_hc_grad,norm_clip);
// 		computeNorm(b_c_grad,norm_clip);

// 		computeNorm(W_ho_grad,norm_clip);
// 		computeNorm(b_o_grad,norm_clip);

// 		computeNorm(W_hi_b_grad,norm_clip);
// 		computeNorm(W_hf_b_grad,norm_clip);
// 		computeNorm(W_hc_b_grad,norm_clip);
// 		computeNorm(W_ho_b_grad,norm_clip);
// 	}

// 	W_hi_b.noalias() += (learning_rate)*W_hi_b_grad;
// 	W_hi.noalias() += (learning_rate)*W_hi_grad;
// 	b_i.noalias() += (learning_rate)*b_i_grad;

// 	W_hf_b.noalias() += (learning_rate)*W_hf_b_grad;
// 	W_hf.noalias() += (learning_rate)*W_hf_grad;
// 	b_f.noalias() += (learning_rate)*b_f_grad;

// 	W_hc_b.noalias() += (learning_rate)*W_hc_b_grad;
// 	W_hc.noalias() += (learning_rate)*W_hc_grad;
// 	b_c.noalias() += (learning_rate)*b_c_grad;

// 	W_ho_b.noalias() += (learning_rate)*W_ho_b_grad;
// 	W_ho.noalias() += (learning_rate)*W_ho_grad;
// 	b_o.noalias() += (learning_rate)*b_o_grad;
// }

// template<typename Derived2>
// void Hidden_To_Hidden_Layer::check_all_gradients(double epsilon,const Eigen::MatrixBase<Derived2> &input_minibatch_const,const Eigen::MatrixBase<Derived2> &output_minibatch_const) {
// 		std::cout << "--------------------GRADIENT CHECKING FOR HIDDEN LAYER-------------------------\n";
// 		std::cout << "GRADIENT CHECKING FOR W_hi\n";
// 		check_gradient(epsilon,W_hi,W_hi_grad,input_minibatch_const,output_minibatch_const);
		
// 		std::cout << "GRADIENT CHECKING FOR W_hf\n";
// 		check_gradient(epsilon,W_hf,W_hf_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR W_ho\n";
// 		check_gradient(epsilon,W_ho,W_ho_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR W_hc\n";
// 		check_gradient(epsilon,W_hc,W_hc_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR b_i\n";
// 		check_gradient(epsilon,b_i,b_i_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR b_f\n";
// 		check_gradient(epsilon,b_f,b_f_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR b_c\n";
// 		check_gradient(epsilon,b_c,b_c_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR b_o\n";
// 		check_gradient(epsilon,b_o,b_o_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR W_hi_b\n";
// 		check_gradient(epsilon,W_hi_b,W_hi_b_grad,input_minibatch_const,output_minibatch_const);
		
// 		std::cout << "GRADIENT CHECKING FOR W_hf_b\n";
// 		check_gradient(epsilon,W_hf_b,W_hf_b_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR W_ho_b\n";
// 		check_gradient(epsilon,W_ho_b,W_ho_b_grad,input_minibatch_const,output_minibatch_const);

// 		std::cout << "GRADIENT CHECKING FOR W_hc_b\n";
// 		check_gradient(epsilon,W_hc_b,W_hc_b_grad,input_minibatch_const,output_minibatch_const);
// }

// template<typename Derived,typename Derived2,typename Derived3>
// void Hidden_To_Hidden_Layer::check_gradient(double epsilon,const Eigen::MatrixBase<Derived3> &parameter_const,const Eigen::MatrixBase<Derived> &grad,const Eigen::MatrixBase<Derived2> &input_minibatch_const,const Eigen::MatrixBase<Derived2> &output_minibatch_const) {
// 	UNCONST(Derived3, parameter_const, parameter);
// 	for(int i=0; i<grad.rows(); i++) {
// 		for(int j=0; j<grad.cols(); j++) {
// 			double loss = 0;
// 			parameter(i,j)+= epsilon;
// 			loss = model->getError(input_minibatch_const,output_minibatch_const);
// 			parameter(i,j)+= -2*epsilon;
// 			loss-= model->getError(input_minibatch_const,output_minibatch_const);
// 			if( (std::abs(grad(i,j) - loss/(2.0*epsilon))) > 1/1000.0 ||  (std::abs(grad(i,j) - loss/(2.0*epsilon))/(std::abs(grad(i,j)) + std::abs(loss/(2.0*epsilon)))) > 1/1000.0  ) {
// 				std::cout << "Gradient for gradient check: " << loss/(2*epsilon) << "\n";
// 				std::cout << "My gradient: " << grad(i,j) << "\n";
// 				std::cout << "Gradient difference: " << std::abs(grad(i,j) - loss/(2.0*epsilon)) << "\n";
// 				std::cout << "Gradient difference (Equation 2): " << std::abs(grad(i,j) - loss/(2.0*epsilon))/(std::abs(grad(i,j)) + std::abs(loss/(2.0*epsilon)) ) << "\n\n";
// 			}
// 		}
// 	}
// }

// void Hidden_To_Hidden_Layer::dump_weights(std::ofstream &output) {

// 	writeMatrix(W_hi,output);
// 	writeMatrix(b_i,output);

// 	writeMatrix(W_hf,output);
// 	writeMatrix(b_f,output);

// 	writeMatrix(W_hc,output);
// 	writeMatrix(b_c,output);

// 	writeMatrix(W_ho,output);
// 	writeMatrix(b_o,output);

// 	writeMatrix(W_hi_b,output);
// 	writeMatrix(W_hf_b,output);
// 	writeMatrix(W_hc_b,output);
// 	writeMatrix(W_ho_b,output);
// }

// void Hidden_To_Hidden_Layer::load_weights(std::ifstream &input) {
// 	readMatrix(W_hi,input);
// 	readMatrix(b_i,input);

// 	readMatrix(W_hf,input);
// 	readMatrix(b_f,input);

// 	readMatrix(W_hc,input);
// 	readMatrix(b_c,input);

// 	readMatrix(W_ho,input);
// 	readMatrix(b_o,input);

// 	readMatrix(W_hi_b,input);
// 	readMatrix(W_hf_b,input);
// 	readMatrix(W_hc_b,input);
// 	readMatrix(W_ho_b,input);
// }

// template<typename Derived>
// void Hidden_To_Hidden_Layer::initMatrix(const Eigen::MatrixBase<Derived> &input_const) {
// 	UNCONST(Derived,input_const,input);
// 	double lower = -1.0; //Lower bound for uniform dist
// 	double upper = 1.0; //Upper bound for uniform dist
// 	boost::uniform_real<> distribution(lower,upper);
// 	for(int j=0; j<input.cols(); j++) {
// 		for(int i=0; i<input.rows(); i++) {
// 			input(i,j) =  distribution(gen);
// 		}
// 	}
// }

