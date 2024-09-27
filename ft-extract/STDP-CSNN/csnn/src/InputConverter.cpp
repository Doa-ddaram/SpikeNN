#include "InputConverter.h"

//
//	InputConverter
//


Shape InputConverter::compute_shape(const Shape& shape) {
	return shape;
}

size_t InputConverter::train_pass_number() const {
	return 1;
}

void InputConverter::process_train_sample(const std::string&, Tensor<float>& sample, size_t, size_t, size_t) {
	Tensor<Time> out(sample.shape());
	out.fill(INFINITE_TIME);
	process(sample, out);
	sample = out;
}

void InputConverter::process_test_sample(const std::string&, Tensor<float>& sample, size_t, size_t) {
	Tensor<Time> out(sample.shape());
	out.fill(INFINITE_TIME);
	process(sample, out);
	sample = out;
}

//
//	LatencyCoding
//

static RegisterClassParameter<LatencyCoding, InputConverterFactory> _register_latency("LatencyCoding");

LatencyCoding::LatencyCoding() : InputConverter(_register_latency) {

}

void LatencyCoding::process(const Tensor<float>&in, Tensor<Time>& out) {
	size_t size = in.shape().product();

	for(size_t i=0; i<size; i++) {
		Time ts = std::max<Time>(0.0f, 1.0f-in.at_index(i));
		out.at_index(i) = ts == 1.0f ? INFINITE_TIME : ts;
	}
}


//
//	RankOrderCoding
//

static RegisterClassParameter<RankOrderCoding, InputConverterFactory> _register_rank_order("RankOrderCoding");

RankOrderCoding::RankOrderCoding() : InputConverter(_register_rank_order) {

}

void RankOrderCoding::process(const Tensor<float>&in, Tensor<Time>& out) {
	size_t size = in.shape().product();

	std::vector<std::pair<size_t, float>> list;

	for(size_t i=0; i<size; i++) {
		list.emplace_back(i, in.at_index(i));
	}

	std::sort(std::begin(list), std::end(list), [](const auto& e1, const auto& e2) {
		return e1.second > e2.second;
	});

	for(size_t i=0; i<list.size(); i++) {
		out.at_index(list[i].first) = static_cast<Time>(i)/static_cast<Time>(list.size()-1);
	}

}
