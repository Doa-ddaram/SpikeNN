#include "execution/TestingExecution.h"
#include "Math.h"

TestingExecution::TestingExecution(ExperimentType& experiment) :
	_experiment(experiment), _test_set() {

}

void TestingExecution::process(size_t refresh_interval) {
	_load_data();

	for(size_t i=0; i<_experiment.process_number(); i++) {
		_experiment.print() << "Process " << _experiment.process_at(i).factory_name() << "." << _experiment.process_at(i).class_name();
		if(!_experiment.process_at(i).name().empty()) {
			_experiment.print() << " (" << _experiment.process_at(i).name() << ")";
		}
		_experiment.print() << std::endl;

		// Load trained parameters
		std::string process_load_path = _experiment.model_path() + "/" + _experiment.process_at(i).factory_name() + "." + _experiment.process_at(i).class_name();
		if (!_experiment.process_at(i).name().empty()) {
			process_load_path += "." + _experiment.process_at(i).name();
		}
		process_load_path += "/";
		bool loaded = _experiment.process_at(i).load_params(process_load_path);
		if (loaded) {
			_experiment.log() << "Load trained parameters at " << process_load_path << std::endl;
		}

		_process_test_data(_experiment.process_at(i), _test_set);
		_process_output(i);
	}

	_test_set.clear();
}

Tensor<Time> TestingExecution::compute_time_at(size_t i) const {
	throw std::runtime_error("Unimplemented");
}

void TestingExecution::_load_data() {
	for(Input* input : _experiment.test_data()) {
		size_t count = 0;
		while(input->has_next()) {
			_test_set.push_back(input->next());
			count ++;
		}
		_experiment.log() << "Load " << count << " test samples from " << input->to_string() << std::endl;
		input->close();
	}
}

void TestingExecution::_process_test_data(AbstractProcess& process, std::vector<std::pair<std::string, Tensor<float>>>& data) {
	for(size_t j=0; j<_test_set.size(); j++) {
		process.process_test_sample(data[j].first, data[j].second, j, data.size());
		if(data[j].second.shape() != process.shape()) {
			throw std::runtime_error("Unexpected shape (actual: "+data[j].second.shape().to_string()+", expected: "+process.shape().to_string()+")");
		}
	}
}

void TestingExecution::_process_output(size_t index) {
	for(size_t i=0; i<_experiment.output_count(); i++) {
		if(_experiment.output_at(i).index() == index) {
			Output& output = _experiment.output_at(i);

			std::vector<std::pair<std::string, Tensor<float>>> output_test_set;

			for(std::pair<std::string, Tensor<float>>& entry : _test_set) {
				output_test_set.emplace_back(entry.first, output.converter().process(entry.second));
			}

			for(Process* process : output.postprocessing()) {
				_experiment.print() << "Process " << process->class_name() << std::endl;
				_process_test_data(*process, output_test_set);
			}

			for(Analysis* analysis : output.analysis()) {

				_experiment.log() << output.name() << ", analysis " << analysis->class_name() << ":" << std::endl;

				size_t n = analysis->train_pass_number();

				if(n == 0) {
					analysis->after_test();
				}
				else {
					analysis->before_test();
					for(std::pair<std::string, Tensor<float>>& entry : output_test_set) {
						analysis->process_test_sample(entry.first, entry.second);
					}
					analysis->after_test();
				}

			}
		}
	}
}
