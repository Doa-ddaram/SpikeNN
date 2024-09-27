#ifndef _MONITOR_SAVE_NETWORK_H
#define _MONITOR_SAVE_NETWORK_H

#include <vector>
#include <string>
#include "Tensor.h"
#include "Monitor.h"

namespace monitor {

	class SaveNetwork : public Monitor {

	public:
		SaveNetwork();

		virtual void on_sample(const AbstractExperiment& experiment, size_t layer_index, size_t sample_count);
		virtual void on_epoch(const AbstractExperiment& experiment, size_t layer_index, size_t epoch_count);
	};
}

#endif
