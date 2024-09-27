#ifndef _PROCESS_WHITEN_H
#define _PROCESS_WHITEN_H

#include <lapacke.h>

#include "Tensor.h"
#include "Process.h"
#include "Math.h"

namespace process {

	class Whitening : public TwoPassProcess {

	public:
		Whitening();
		Whitening(float eps, float pca_compress = 1.0, size_t max_sample = std::numeric_limits<size_t>::max());

		virtual Shape compute_shape(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		float _eps;
		float _pca_compress;
		size_t _max_sample;

		std::vector<Tensor<float>> _list;

		Tensor<float> _w;
		Tensor<float> _mean;

	};

}

#endif
