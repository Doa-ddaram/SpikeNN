#include "tool/Histogram.h"

using namespace tool;


void Histogram::process(const std::string& output, const Tensor<float>& t, size_t n) {
	std::ofstream file(output, std::ios::out | std::ios::trunc);

	if(!file.is_open()) {
		throw std::runtime_error("Unable to open "+output);
	}

	Tensor<size_t> count(Shape({n}));
	count.fill(0);

	auto minmax = std::minmax_element(std::begin(t), std::end(t));

	float min_v = *minmax.first;
	float max_v = *minmax.second;

	for(float v : t) {
		count.at_index((v-min_v)/(max_v-min_v)*static_cast<float>(n-1))++;
	}

	for(size_t i=0; i<n; i++) {
		float r = (max_v-min_v)/static_cast<float>(n);
		float lower = static_cast<float>(i)*r+min_v;
		float upper = lower+r;

		file << lower << " " << upper << " " << count.at_index(i) << std::endl;
	}


	file.close();
}
