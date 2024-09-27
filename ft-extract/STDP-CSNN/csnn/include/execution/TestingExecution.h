#ifndef _EXECUTION_TESTING_EXECUTION_H
#define _EXECUTION_TESTING_EXECUTION_H

#include "Experiment.h"
#include "SpikeConverter.h"

class TestingExecution {

public:
	typedef Experiment<TestingExecution> ExperimentType;

	TestingExecution(ExperimentType& experiment);

	void process(size_t refresh_interval);

	Tensor<Time> compute_time_at(size_t i) const;
	
private:
	void _prepare(size_t layer_target_index);
	void _process_sample(const std::string& label, const std::vector<Spike>& in, size_t layer_index);
	void _update_data(size_t layer_index, size_t refresh_interval);

	void _load_data();

	void _process_test_data(AbstractProcess& process, std::vector<std::pair<std::string, Tensor<float>>>& data);
	void _process_output(size_t index);

	ExperimentType& _experiment;

	std::vector<std::pair<std::string, Tensor<float>>> _test_set;
};

#endif
