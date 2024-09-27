#ifndef _PROCESS_GRAYSCALE_H
#define _PROCESS_GRAYSCALE_H

#include "Process.h"
#include <cassert>
#include <cmath>

namespace process {
	class GrayScale : public UniquePassProcess {

	public:
		GrayScale();

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& sample) const;

		size_t _width;
		size_t _height;
		size_t _depth;
	};
}

#endif
