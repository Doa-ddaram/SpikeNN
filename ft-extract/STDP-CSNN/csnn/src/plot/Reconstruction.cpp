#include "plot/Reconstruction.h"
#include "Layer.h"

#ifdef ENABLE_QT
using namespace plot;

Tensor<float> priv::ReconstructionHelper::process(const std::vector<const Layer *>& layers, size_t i) {
	Tensor<float> out(Shape({1, 1, layers.front()->depth()}));
	out.fill(0);
	out.at(0, 0, i) = 1.0;

	for(const Layer* layer : layers) {
		out = layer->reconstruct(out);
	}

	out.range_normalize();

	return out;
}
#endif
