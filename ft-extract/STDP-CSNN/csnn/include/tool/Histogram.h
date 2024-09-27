#ifndef _TOOL_HISTOGRAM_H
#define _TOOL_HISTOGRAM_H

#include <fstream>
#include "Tensor.h"

namespace tool {

	class Histogram {

	public:
		Histogram() = delete;

		static void process(const std::string& output, const Tensor<float>& t, size_t n);

	};
}

#endif
