#include "layer/Pooling.h"

using namespace layer;

static RegisterClassParameter<Pooling, LayerFactory> _register("Pooling");

Pooling::Pooling() : Layer3D(_register), _inh() {

}

Pooling::Pooling(size_t filter_width, size_t filter_height, size_t stride_x, size_t stride_y, size_t padding_x, size_t padding_y) :
	Layer3D(_register, filter_width, filter_height, 0, stride_x, stride_y, padding_x, padding_y),
	_inh(Shape({_width, _height, _depth})) {

}

Shape Pooling::compute_shape(const Shape& previous_shape) {
	parameter<size_t>("filter_number").set(previous_shape.dim(2));

	Layer3D::compute_shape(previous_shape);

	_inh = Tensor<bool>(Shape({_width, _height, _depth}));

	return Shape({_width, _height, _depth});
}


size_t Pooling::train_pass_number() const {
	return 1;
}

void Pooling::process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number) {
	if(current_index == 0) {
		std::cout << "Process train set" << std::endl;
		_current_width = _width;
		_current_height = _height;
	}
	std::vector<Spike> input_spike;
	SpikeConverter::to_spike(sample, input_spike);
	std::vector<Spike> output_spike;
	test(label, input_spike, sample, output_spike);
	sample = Tensor<float>(shape());
	SpikeConverter::from_spike(output_spike, sample);
}

void Pooling::process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number) {
	if(current_index == 0) {
		std::cout << "Process test set" << std::endl;
		_current_width = _width;
		_current_height = _height;
	}
	std::vector<Spike> input_spike;
	SpikeConverter::to_spike(sample, input_spike);
	std::vector<Spike> output_spike;
	test(label, input_spike, sample, output_spike);
	sample = Tensor<float>(shape());
	SpikeConverter::from_spike(output_spike, sample);
}

void Pooling::train(const std::string&, const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike) {
	_exec(input_spike, output_spike);
}

void Pooling::test(const std::string&, const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike) {
	_exec(input_spike, output_spike);
}

Tensor<float> Pooling::reconstruct(const Tensor<float>& t) const {
	size_t output_width = t.shape().dim(0);
	size_t output_height = t.shape().dim(1);
	size_t output_depth = t.shape().dim(2);

	Tensor<float> out(Shape({output_width*_stride_x-_filter_width+1, output_height*_stride_y-_filter_height+1, output_depth}));
	out.fill(0);

	Tensor<float> norm(Shape({output_width*_stride_x-_filter_width+1, output_height*_stride_y-_filter_height+1, output_depth}));
	norm.fill(0);

	for(size_t x=0; x<output_width; x++) {
		for(size_t y=0; y<output_height; y++) {
			for(size_t z=0; z<output_depth; z++) {
				out.at(x*_stride_x, y*_stride_y, z) = t.at(x, y, z);
			}
		}
	}

	return out;
}

void Pooling::_exec(const std::vector<Spike>& input_spike, std::vector<Spike>& output_spike) {
	std::fill(std::begin(_inh), std::end(_inh), false);

	for(const Spike& spike : input_spike) {
		std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>> output_spikes;
		forward(spike.x, spike.y, output_spikes);

		for(const auto& entry : output_spikes) {
			uint16_t x = std::get<0>(entry);
			uint16_t y = std::get<1>(entry);
			uint16_t z = spike.z;

			if(!_inh.at(x, y, z)) {
				output_spike.emplace_back(spike.time, x, y, z);
				_inh.at(x, y, z) = true;
			}



		}
	}
}
