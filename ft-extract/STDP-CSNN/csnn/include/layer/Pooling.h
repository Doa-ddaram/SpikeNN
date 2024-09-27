#ifndef _LAYER_POOLING_H
#define _LAYER_POOLING_H

#include "Layer.h"

namespace layer {

	class Pooling : public Layer3D {

	public:
        Pooling();
		Pooling(size_t filter_width, size_t filter_height, size_t stride_x, size_t stride_y, size_t padding_x = 0, size_t padding_y = 0);

		virtual Shape compute_shape(const Shape& previous_shape);

		virtual size_t train_pass_number() const;
		virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number);
		virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number);

		virtual void train(const std::string& label, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike);
		virtual void test(const std::string& label, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike);
		virtual Tensor<float> reconstruct(const Tensor<float>& t) const;

	private:
		void _exec(const std::vector<Spike>& input_spike, std::vector<Spike>& output_spike);

		Tensor<bool> _inh;
	};

}

#endif
