//STL includes
#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <chrono>

//Eigen includes
#include <Eigen/Dense>

#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

typedef double precision;

#include "decoder.h"
#include "file_helper_decoder.h"

boost::random::mt19937 gen; //Random number generator for initializing weights

template<typename Derived>
void initMatrix(const Eigen::MatrixBase<Derived> &input_const) {
	UNCONST(Derived,input_const,input);
	precision lower = 0.0; //Lower bound for uniform dist
	precision upper = 5.0; //Upper bound for uniform dist
	boost::uniform_real<> distribution(lower,upper);
	for(int j=0; j<input.cols(); j++) {
		for(int i=0; i<input.rows(); i++) {
			input(i,j) =  distribution(gen);
		}
	}
}

template<typename Derived,typename Derived2>
void init_output_dist(const Eigen::MatrixBase<Derived> &outputDist_const,const Eigen::MatrixBase<Derived2> &normalization_const) {
	UNCONST(Derived,outputDist_const,outputDist);
	UNCONST(Derived2,normalization_const,normalization);
	initMatrix(outputDist);
	//std::cout << outputDist << "\n\n";
	normalization.setZero();
	for(int i=0; i<outputDist.rows(); i++) {
		normalization += outputDist.row(i);
	}

	for(int i=0; i<outputDist.rows(); i++) {
		outputDist.row(i) = (outputDist.row(i).array()/normalization.array()).matrix();
	}
}


template<typename Derived,typename Derived2>
void swap_states(const Eigen::MatrixBase<Derived> &temp_const,const Eigen::MatrixBase<Derived> &mat_const,
	const Eigen::MatrixBase<Derived2> &indicies)
{
	UNCONST(Derived,mat_const,mat);
	UNCONST(Derived,temp_const,temp);
	for(int i=0; i<indicies.rows(); i++) {
		temp.col(i) = mat.col(indicies(i));
	}
	mat = temp;
}

int main() {

	gen.seed(static_cast<unsigned int>(std::time(0)));

	const int beam_size = 3;
	const int vocab_size = 10;
	const int start_symbol = 0;
	const int end_symbol = 1;
	const int max_decoding_length = 30;
	const int min_decoding_length = 10;
	const int source_length = 10;
	const precision penalty = 3;

	decoder<precision> d(beam_size,vocab_size,start_symbol,end_symbol,
		max_decoding_length,min_decoding_length,penalty,"");

	Eigen::Matrix<precision,Eigen::Dynamic, Eigen::Dynamic> outputDist;
	Eigen::Matrix<precision, 1, Eigen::Dynamic> normalization;
	outputDist.resize(vocab_size,beam_size);
	normalization.resize(1,beam_size);

	init_output_dist(outputDist,normalization);

	d.init_decoder();

	for(int i=0; i<20; i++) {
		init_output_dist(outputDist,normalization);
		d.expand_hypothesis(outputDist);
		d.print_current_hypotheses();
	}
	d.finish_current_hypotheses(outputDist);
	d.print_current_hypotheses();
}




