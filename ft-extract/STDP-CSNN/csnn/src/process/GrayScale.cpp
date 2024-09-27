#include "process/GrayScale.h"

using namespace process;

static RegisterClassParameter<GrayScale, ProcessFactory> _register("GrayScale");


GrayScale::GrayScale() : UniquePassProcess(_register),
	_width(0), _height(0), _depth(0) {

}

Shape GrayScale::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = 1;
	return Shape({_width, _height, _depth});
}

void GrayScale::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void GrayScale::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void GrayScale::_process(Tensor<float>& in) const {
	assert(in.shape().dim(2) == 3);

	Tensor<float> out(Shape({_width, _height, _depth}));

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			float v = 0;
            for(size_t z=0; z<3; z++) {
				//NOTE: Assume that the input order is RGB
				v += in.at(x,y,z) * ((z == 0) ? 0.2989 : ((z == 1) ? 0.5870 : 0.1140));
			}
			out.at(x, y, 0) = v;
		}
	}

	in = out;
}
