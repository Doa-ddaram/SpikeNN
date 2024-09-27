#include "process/OnOffFilter.h"
#include "Experiment.h"
#include "Math.h"

using namespace process;

//
//  OnOffFilterHelper
//

Tensor<float> process::_priv::OnOffFilterHelper::generate_filter(size_t filter_size, float center_dev, float surround_dev) {

	Tensor<float> d2(Shape({filter_size, filter_size}));
	for(size_t i=0; i<filter_size; i++) {
		for(size_t j=0; j<filter_size; j++) {
			d2.at(i, j) = std::pow((i+1)-static_cast<float>(filter_size)/2.0f-0.5f, 2.0f)+std::pow((j+1)-static_cast<float>(filter_size)/2.0f-0.5f, 2.0f);
		}
	}


	Tensor<float> filter1(Shape({filter_size, filter_size}));
	Tensor<float> filter2(Shape({filter_size, filter_size}));
	float filter_sum1 = 0;
	float filter_sum2 = 0;

	for(size_t i=0; i<filter_size; i++) {
		for(size_t j=0; j<filter_size; j++) {
			filter1.at(i, j) = 1.0f/std::sqrt(2.0f*static_cast<float>(M_PI))*(1.0f/center_dev*std::exp(-d2.at(i, j)/2.0f/(center_dev*center_dev)));
			filter_sum1 += filter1.at(i, j);

			filter2.at(i, j) = 1.0f/std::sqrt(2.0f*static_cast<float>(M_PI))*(1.0f/surround_dev*std::exp(-d2.at(i, j)/2.0f/(surround_dev*surround_dev)));
			filter_sum2 += filter2.at(i, j);

		}
	}

	Tensor<float> filter(Shape({filter_size, filter_size}));
	for(size_t i=0; i<filter_size; i++) {
		for(size_t j=0; j<filter_size; j++) {
			filter.at(i, j) = filter1.at(i, j)/filter_sum1-filter2.at(i, j)/filter_sum2;
		}
	}

	return filter;
}

//
//	DefaultOnOffFilter
//

static RegisterClassParameter<DefaultOnOffFilter, ProcessFactory> _register_1("DefaultOnOffFilter");


DefaultOnOffFilter::DefaultOnOffFilter() : UniquePassProcess(_register_1),
	_filter_size(0), _center_dev(0), _surround_dev(0), _width(0), _height(0), _depth(0), _filter() {
	add_parameter("filter_size", _filter_size);
	add_parameter("center_dev", _center_dev);
	add_parameter("surround_dev", _surround_dev);
}

DefaultOnOffFilter::DefaultOnOffFilter(size_t filter_size, float center_dev, float surround_dev) : DefaultOnOffFilter() {
	parameter<size_t>("filter_size").set(filter_size);
	parameter<float>("center_dev").set(center_dev);
	parameter<float>("surround_dev").set(surround_dev);
}

Shape  DefaultOnOffFilter::compute_shape(const Shape& shape) {
	parameter<size_t>("filter_size").ensure_initialized(experiment()->random_generator());
	parameter<float>("center_dev").ensure_initialized(experiment()->random_generator());
	parameter<float>("surround_dev").ensure_initialized(experiment()->random_generator());


	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_filter = _priv::OnOffFilterHelper::generate_filter(_filter_size, _center_dev, _surround_dev);
	return Shape({_width, _height, _depth*2});
}

void DefaultOnOffFilter::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void DefaultOnOffFilter::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void DefaultOnOffFilter::_process(Tensor<InputType>& in) const {
	Tensor<InputType> out(Shape({_width, _height, _depth*2}));

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			for(size_t z=0; z<_depth; z++) {
				float v = 0;
				for(size_t fx=0; fx<_filter_size; fx++) {
					for(size_t fy=0; fy<_filter_size; fy++) {
						size_t x_in = x+fx > _filter_size/2 ? std::min(x+fx-_filter_size/2, _width-1) : 0;
						size_t y_in = y+fy > _filter_size/2 ? std::min(y+fy-_filter_size/2, _height-1) : 0;

						v += in.at(x_in, y_in, z)*_filter.at(fx, fy);
					}
				}

				out.at(x, y, z*2) = std::max<float>(0, v);
				out.at(x, y, z*2+1) = std::max<float>(0, -v);
			}
		}
	}

	in = out;
}

//
//	CustomRGBOnOffFilter
//

static RegisterClassParameter<CustomRGBOnOffFilter, ProcessFactory> _register_2("CustomRGBOnOffFilter");


CustomRGBOnOffFilter::CustomRGBOnOffFilter() : UniquePassProcess(_register_2),
	_r(), _g(), _b(), _width(0), _height(0) {
	add_parameter("r", _r);
	add_parameter("g", _g);
	add_parameter("b", _b);
}

CustomRGBOnOffFilter::CustomRGBOnOffFilter(const std::string& filename): CustomRGBOnOffFilter() {
	NumpyArchive filters;
	NumpyReader::load(filename, filters);

	parameter<Tensor<float>>("r").set(filters.at("arr_0.npy").to_tensor<float>());
	parameter<Tensor<float>>("g").set(filters.at("arr_1.npy").to_tensor<float>());
	parameter<Tensor<float>>("b").set(filters.at("arr_2.npy").to_tensor<float>());
}

Shape CustomRGBOnOffFilter::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);

	if(shape.dim(2) != 3) {
		throw std::runtime_error("Only depth a 3 is valid (RGB)");
	}

	return Shape({_width, _height, 6});
}

void CustomRGBOnOffFilter::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void CustomRGBOnOffFilter::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void CustomRGBOnOffFilter::_process(Tensor<InputType>& in) const {

	Tensor<InputType> out(Shape({_width, _height, 6}));

	// Red
	size_t r_width = _r.shape().dim(0);
	size_t r_height = _r.shape().dim(1);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			float v = 0;
			for(size_t fx=0; fx<r_width; fx++) {
				for(size_t fy=0; fy<r_height; fy++) {
					size_t x_in = x+fx > r_width/2 ? std::min(x+fx-r_width/2, _width-1) : 0;
					size_t y_in = y+fy > r_height/2 ? std::min(y+fy-r_height/2, _height-1) : 0;

					for(size_t z=0; z<3; z++) {
						v += in.at(x_in, y_in, z)*_r.at(fx, fy, z);
					}
				}
			}

			out.at(x, y, 0) = std::max<float>(0, v);
			out.at(x, y, 1) = std::max<float>(0, -v);
		}
	}

	// Green
	size_t g_width = _g.shape().dim(0);
	size_t g_height = _g.shape().dim(1);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			float v = 0;
			for(size_t fx=0; fx<g_width; fx++) {
				for(size_t fy=0; fy<g_height; fy++) {
					size_t x_in = x+fx > g_width/2 ? std::min(x+fx-g_width/2, _width-1) : 0;
					size_t y_in = y+fy > g_height/2 ? std::min(y+fy-g_height/2, _height-1) : 0;

					for(size_t z=0; z<3; z++) {
						v += in.at(x_in, y_in, z)*_g.at(fx, fy, z);
					}
				}
			}

			out.at(x, y, 2) = std::max<float>(0, v);
			out.at(x, y, 3) = std::max<float>(0, -v);
		}
	}

	// Blue
	size_t b_width = _b.shape().dim(0);
	size_t b_height = _b.shape().dim(1);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			float v = 0;
			for(size_t fx=0; fx<b_width; fx++) {
				for(size_t fy=0; fy<b_height; fy++) {
					size_t x_in = x+fx > b_width/2 ? std::min(x+fx-b_width/2, _width-1) : 0;
					size_t y_in = y+fy > b_height/2 ? std::min(y+fy-b_height/2, _height-1) : 0;

					for(size_t z=0; z<3; z++) {
						v += in.at(x_in, y_in, z)*_b.at(fx, fy, z);
					}
				}
			}

			out.at(x, y, 4) = std::max<float>(0, v);
			out.at(x, y, 5) = std::max<float>(0, -v);
		}
	}

	in = out;
}

//
//	BiologicalOnOffFilter
//

static RegisterClassParameter<BiologicalOnOffFilter, ProcessFactory> _register_3("BiologicalOnOffFilter");


BiologicalOnOffFilter::BiologicalOnOffFilter() : UniquePassProcess(_register_3),
	_filter_size(0), _center_dev(0), _surround_dev(0), _width(0), _height(0), _depth(0), _filter() {
	add_parameter("filter_size", _filter_size);
	add_parameter("center_dev", _center_dev);
	add_parameter("surround_dev", _surround_dev);
}

BiologicalOnOffFilter::BiologicalOnOffFilter(size_t filter_size, float center_dev, float surround_dev) : BiologicalOnOffFilter() {
	parameter<size_t>("filter_size").set(filter_size);
	parameter<float>("center_dev").set(center_dev);
	parameter<float>("surround_dev").set(surround_dev);
}

Shape  BiologicalOnOffFilter::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);

	if(_depth != 3) {
		throw std::runtime_error("Incompatible depth (require 3 channels)");
	}

	_filter = _priv::OnOffFilterHelper::generate_filter(_filter_size, _center_dev, _surround_dev);
	return Shape({_width, _height, _depth*2});
}

void BiologicalOnOffFilter::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void BiologicalOnOffFilter::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void BiologicalOnOffFilter::_process(Tensor<InputType>& in) const {
	Tensor<InputType> out(Shape({_width, _height, _depth*2}));

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {

			// Black / White
			float v1 = 0;
			for(size_t fx=0; fx<_filter_size; fx++) {
				for(size_t fy=0; fy<_filter_size; fy++) {
					size_t x_in = x+fx > _filter_size/2 ? std::min(x+fx-_filter_size/2, _width-1) : 0;
					size_t y_in = y+fy > _filter_size/2 ? std::min(y+fy-_filter_size/2, _height-1) : 0;

					v1 += (0.2126*in.at(x_in, y_in, 0)+0.7152*in.at(x_in, y_in, 1)+0.0722*in.at(x_in, y_in, 2))*_filter.at(fx, fy);
				}
			}

			out.at(x, y, 0) = std::max<float>(0, v1);
			out.at(x, y, 1) = std::max<float>(0, -v1);

			// Red / Green
			float v2 = 0;
			for(size_t fx=0; fx<_filter_size; fx++) {
				for(size_t fy=0; fy<_filter_size; fy++) {
					size_t x_in = x+fx > _filter_size/2 ? std::min(x+fx-_filter_size/2, _width-1) : 0;
					size_t y_in = y+fy > _filter_size/2 ? std::min(y+fy-_filter_size/2, _height-1) : 0;

					v2 += (0.5+0.5*in.at(x_in, y_in, 0)-0.5*in.at(x_in, y_in, 1))*_filter.at(fx, fy);
				}
			}

			out.at(x, y, 2) = std::max<float>(0, v2);
			out.at(x, y, 3) = std::max<float>(0, -v2);


			// Yellow / Blue
			float v3 = 0;
			for(size_t fx=0; fx<_filter_size; fx++) {
				for(size_t fy=0; fy<_filter_size; fy++) {
					size_t x_in = x+fx > _filter_size/2 ? std::min(x+fx-_filter_size/2, _width-1) : 0;
					size_t y_in = y+fy > _filter_size/2 ? std::min(y+fy-_filter_size/2, _height-1) : 0;

					v3 += (0.5+0.25*in.at(x_in, y_in, 0)+0.25*in.at(x_in, y_in, 1)-0.5*in.at(x_in, y_in, 2))*_filter.at(fx, fy);
				}
			}

			out.at(x, y, 4) = std::max<float>(0, v3);
			out.at(x, y, 5) = std::max<float>(0, -v3);


		}
	}

	in = out;
}
