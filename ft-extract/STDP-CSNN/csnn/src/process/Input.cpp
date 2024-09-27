#include "process/Input.h"

using namespace process;

//
//	GaussianTemporalCoding
//

static RegisterClassParameter<GaussianTemporalCoding, ProcessFactory> _register_1("GaussianTemporalCoding");

GaussianTemporalCoding::GaussianTemporalCoding() : UniquePassProcess(_register_1),
	_center(0), _scale(0), _width(0), _height(0), _depth(0) {
	add_parameter("center", _center);
	add_parameter("scale", _scale);
}

GaussianTemporalCoding::GaussianTemporalCoding(float center, float scale) : GaussianTemporalCoding() {
	parameter<float>("center").set( center);
	parameter<float>("scale").set(scale);
}

Shape GaussianTemporalCoding::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);

	return Shape({_width, _height, _depth*2});
}

void GaussianTemporalCoding::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void GaussianTemporalCoding::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void GaussianTemporalCoding::_process(Tensor<InputType>& t) const {
	Tensor<InputType> out(Shape({_width, _height, _depth*2}));


	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			for(size_t z=0; z<_depth; z++) {
				float v = (t.at(x, y, z)-_center)*_scale;
				out.at(x, y, z*2) = std::max<float>(0, std::min<float>(1, v));
				out.at(x, y, z*2+1) = std::max<float>(0, std::min<float>(1, -v));
			}
		}
	}

	t = out;
}

#ifdef ENABLE_QT
QColor GaussianTemporalCodingColor::get(const std::vector<float>& list) {

	std::vector<float> out_list;

	for(size_t i=0; i<list.size()/2; i++) {
		out_list.push_back(0.5+0.5*list[i*2]-0.5*list[i*2+1]);
	}


	return DefaultColor::get(out_list);
}
#endif

