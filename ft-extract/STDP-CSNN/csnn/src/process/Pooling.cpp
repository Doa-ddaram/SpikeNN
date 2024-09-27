#include "process/Pooling.h"

using namespace process;

static RegisterClassParameter<SumPooling, ProcessFactory> _register("SumPooling");


SumPooling::SumPooling() : UniquePassProcess(_register),
	_target_width(0), _target_height(0), _width(0), _height(0), _depth(0) {
	add_parameter("width", _target_width);
	add_parameter("height", _target_height);
}

SumPooling::SumPooling(size_t target_width, size_t target_height) : SumPooling() {
	parameter<size_t>("width").set(target_width);
	parameter<size_t>("height").set(target_height);
}

Shape SumPooling::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	return Shape({std::min<size_t>(_target_width, _width),
				  std::min<size_t>(_target_height, _height),
				  _depth});
}

void SumPooling::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void SumPooling::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void SumPooling::_process(Tensor<float>& in) const {

	size_t output_width = std::min<size_t>(_target_width, _width);
	size_t output_height = std::min<size_t>(_target_height, _height);

	size_t filter_width = _width/output_width;
	size_t filter_height = _height/output_height;

	Tensor<float> out(Shape({output_width, output_height, _depth}));

	for(size_t x=0; x<output_width; x++) {
		for(size_t y=0; y<output_height; y++) {
			for(size_t z=0; z<_depth; z++) {
				float v = 0;

				for(size_t fx=0; fx<filter_width; fx++) {
					for(size_t fy=0; fy<filter_height; fy++) {
						v += in.at(x*filter_width+fx, y*filter_height+fy, z);
					}
				}

				out.at(x, y, z) = v;
			}
		}
	}

	in = out;
}

static RegisterClassParameter<MeanPooling, ProcessFactory> _register_1("MeanPooling");


MeanPooling::MeanPooling() : UniquePassProcess(_register_1),
	_target_width(0), _target_height(0), _width(0), _height(0), _depth(0) {
	add_parameter("width", _target_width);
	add_parameter("height", _target_height);
}

MeanPooling::MeanPooling(size_t target_width, size_t target_height) : MeanPooling() {
	parameter<size_t>("width").set(target_width);
	parameter<size_t>("height").set(target_height);
}

Shape MeanPooling::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	return Shape({std::min<size_t>(_target_width, _width),
				  std::min<size_t>(_target_height, _height),
				  _depth});
}

void MeanPooling::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void MeanPooling::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void MeanPooling::_process(Tensor<float>& in) const {

	size_t output_width = std::min<size_t>(_target_width, _width);
	size_t output_height = std::min<size_t>(_target_height, _height);

	size_t filter_width = _width/output_width;
	size_t filter_height = _height/output_height;

	Tensor<float> out(Shape({output_width, output_height, _depth}));

	for(size_t x=0; x<output_width; x++) {
		for(size_t y=0; y<output_height; y++) {
			for(size_t z=0; z<_depth; z++) {
				float v = 0;

				for(size_t fx=0; fx<filter_width; fx++) {
					for(size_t fy=0; fy<filter_height; fy++) {
						v += in.at(x*filter_width+fx, y*filter_height+fy, z);
					}
				}

				out.at(x, y, z) = v / (filter_width*filter_height);
			}
		}
	}

	in = out;
}