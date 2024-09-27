#ifndef _SPIKE_CONVERTER_H
#define _SPIKE_CONVERTER_H

#include <vector>
#include "Spike.h"
#include "Tensor.h"

class SpikeConverter {

public:
	SpikeConverter() = delete;

	static void to_spike(const Tensor<Time>& in, std::vector<Spike>& out);
	static void to_spike(const Tensor<Time>& in, std::vector<Spike>& out, size_t x_start, size_t y_start, size_t x_end, size_t y_end);

	static void from_spike(const std::vector<Spike>& in, Tensor<Time>& out);
};

#endif
