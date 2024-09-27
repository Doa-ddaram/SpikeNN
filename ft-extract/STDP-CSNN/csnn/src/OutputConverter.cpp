#include "OutputConverter.h"

//
//	OutputConverter
//

Shape OutputConverter::compute_shape(const Shape& shape) {
	return shape;
}

size_t OutputConverter::train_pass_number() const {
	return 1;
}

void OutputConverter::process_train_sample(const std::string&, Tensor<float>& sample, size_t, size_t, size_t) {
	sample = process(sample);
}

void OutputConverter::process_test_sample(const std::string&, Tensor<float>& sample, size_t, size_t) {
	sample = process(sample);
}



//
//	TimeObjectiveOutput
//

static RegisterClassParameter<TimeObjectiveOutput, OutputConverterFactory> _register_1("TimeObjectiveOutput");

TimeObjectiveOutput::TimeObjectiveOutput() : OutputConverter(_register_1), _t_obj(0) {
    add_parameter("t_obj", _t_obj);
}


TimeObjectiveOutput::TimeObjectiveOutput(Time t_obj) : TimeObjectiveOutput() {
    parameter<float>("t_obj").set(t_obj);
}

Tensor<float> TimeObjectiveOutput::process(const Tensor<Time>& in) {
	Tensor<float> out(in.shape());

	size_t size = in.shape().product();
	for(size_t i=0; i<size; i++) {
		Time t = in.at_index(i);
		out.at_index(i) = t == INFINITE_TIME ? 0.0 : std::min<Time>(1.0, std::max<Time>(0.0, 1.0-(t-_t_obj)/(1.0-_t_obj)));
	}

	return out;
}

//
//	DefaultOutput
//

static RegisterClassParameter<DefaultOutput, OutputConverterFactory> _register_2("DefaultOutput");

DefaultOutput::DefaultOutput() : OutputConverter(_register_2), _min(0), _max(0) {
	add_parameter("min", _min);
	add_parameter("max", _max);
}


DefaultOutput::DefaultOutput(Time min, Time max) : DefaultOutput() {
	parameter<float>("min").set(min);
	parameter<float>("max").set(max);
}

Tensor<float> DefaultOutput::process(const Tensor<Time>& in) {
	Tensor<float> out(in.shape());

	size_t size = in.shape().product();
	for(size_t i=0; i<size; i++) {
		Time t = in.at_index(i);
		out.at_index(i) = t == INFINITE_TIME ? 0.0 : std::min<Time>(1.0, std::max<Time>(0.0, (_max-t)/(_max-_min)));
	}

	return out;
}

//
//	NoOutputConversion
//

static RegisterClassParameter<NoOutputConversion, OutputConverterFactory> _register_3("NoOutputConversion");

NoOutputConversion::NoOutputConversion() : OutputConverter(_register_3) {

}

Tensor<float> NoOutputConversion::process(const Tensor<float>& in) {
	return Tensor<float>(in);
}

//
//	SoftMaxOutput
//

static RegisterClassParameter<SoftMaxOutput, OutputConverterFactory> _register_4("SoftMaxOutput");

SoftMaxOutput::SoftMaxOutput() : OutputConverter(_register_4) {
}

Tensor<float> SoftMaxOutput::process(const Tensor<Time>& in) {
	Tensor<float> out(in.shape());

	size_t size = in.shape().product();
	float min_v = *std::min_element(std::begin(in), std::end(in));

	double sum=0;
	for(size_t i=0; i<size; i++) {
		sum += in.at_index(i) == INFINITE_TIME ? 0 : std::exp(min_v-in.at_index(i));
	}

	for(size_t i=0; i<size; i++) {
		out.at_index(i) = in.at_index(i) == INFINITE_TIME ? 0 : std::exp(min_v-in.at_index(i))/sum;
	}

	return out;
}

//
//	WTAOutput
//

static RegisterClassParameter<WTAOutput, OutputConverterFactory> _register_5("WTAOutput");

WTAOutput::WTAOutput() : OutputConverter(_register_5) {
}

Tensor<float> WTAOutput::process(const Tensor<Time>& in) {
	Tensor<float> out(in.shape());

	size_t size = in.shape().product();
	float min_v = *std::min_element(std::begin(in), std::end(in));

	for(size_t i=0; i<size; i++) {
		out.at_index(i) = in.at_index(i) == min_v ? 1 : 0;
	}

	return out;
}

//
//	SpikeTiming
//

static RegisterClassParameter<SpikeTiming, OutputConverterFactory> _register_6("SpikeTiming");

// Similar to NoOutputConversion but removes INFINITE_TIME
SpikeTiming::SpikeTiming() : OutputConverter(_register_6) {

}

Tensor<float> SpikeTiming::process(const Tensor<Time>& in) {
	Tensor<float> out(in.shape());

	size_t size = in.shape().product();
	for(size_t i=0; i<size; i++) {
		Time t = in.at_index(i);
		out.at_index(i) = t == INFINITE_TIME ? 1.0 : t;
	}

	return out;
}