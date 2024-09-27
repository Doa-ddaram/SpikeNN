#ifndef _TOOL_RECONSTRUCTION_H
#define _TOOL_RECONSTRUCTION_H

#include "Experiment.h"
#include "Color.h"

namespace tool {

	class Reconstruction {

	public:
		Reconstruction() = delete;

		static void process(const std::string& output, const AbstractExperiment& experiment, const Layer& layer, size_t oversampling);

	};
}
#endif
