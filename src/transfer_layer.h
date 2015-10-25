#ifndef TRANSFER_LAYER_H
#define TRANSFER_LAYER_H

#include "base_loss.h"

template<typename dType>
class softmax_layer;

template<typename dType>
class Hidden_To_Hidden_Layer;

template<typename dType>
class Input_To_Hidden_Layer;

template<typename dType>
class upper_transfer_layer {
public:
	bool upper_softmax=true; //this is true if the layer above is a softmax, false if hidden layer
	bool copy_h_t = false; //true if upper layer lies on different GPU, false if not
	bool source_side = true;

	base_loss_layer<dType> *softmax;
	Hidden_To_Hidden_Layer<dType> *hidden_layer;

	upper_transfer_layer() {};

	void init_upper_transfer_layer(bool upper_softmax,bool copy_h_t,bool source_side,base_loss_layer<dType> *softmax,Hidden_To_Hidden_Layer<dType> *hidden_layer) {
		this->upper_softmax = upper_softmax;
		this->copy_h_t = copy_h_t;
		this->softmax = softmax;
		this->source_side = source_side;
		this->hidden_layer = hidden_layer;
	}

};

template<typename dType>
class lower_transfer_layer {
public:
	bool lower_input=true; //this is true if the layer below is a input layer, false if hidden layer
	bool copy_d_Err_ht = false; //true if the lower layer lies on different GPU, false if not

	Input_To_Hidden_Layer<dType> *input_layer;
	Hidden_To_Hidden_Layer<dType> *hidden_layer;

	lower_transfer_layer() {};

	void init_lower_transfer_layer(bool lower_input,bool copy_d_Err_ht,Input_To_Hidden_Layer<dType> *input_layer,Hidden_To_Hidden_Layer<dType> *hidden_layer) {
		this->lower_input = lower_input;
		this->copy_d_Err_ht = copy_d_Err_ht;
		this->input_layer = input_layer;
		this->hidden_layer = hidden_layer;
	}

};

#endif