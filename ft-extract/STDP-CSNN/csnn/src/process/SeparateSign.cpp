#include "process/SeparateSign.h"

using namespace process;

static RegisterClassParameter<SeparateSign, ProcessFactory> _register("SeparateSign");

SeparateSign::SeparateSign() : UniquePassProcess(_register) {

}

Shape SeparateSign::compute_shape(const Shape& shape) {
	return Shape({shape.dim(0), shape.dim(1), shape.dim(2)*2});
}

void SeparateSign::process_train(const std::string&, Tensor<float>& sample) {
	sample = sep_sign(sample);
}

void SeparateSign::process_test(const std::string&, Tensor<float>& sample) {
	sample = sep_sign(sample);
}
