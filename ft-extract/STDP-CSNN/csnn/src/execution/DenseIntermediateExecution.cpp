#include "execution/DenseIntermediateExecution.h"
#include "Math.h"

DenseIntermediateExecution::DenseIntermediateExecution(ExperimentType& experiment) :
	_experiment(experiment), _train_set(), _test_set() {

}

void DenseIntermediateExecution::process(size_t refresh_interval) {
	_load_data();

	std::vector<size_t> train_index;
	for(size_t i=0; i<_train_set.size(); i++) {
		train_index.push_back(i);
	}

	for(size_t i=0; i<_experiment.process_number(); i++) {
		_experiment.print() << "Process " << _experiment.process_at(i).factory_name() << "." << _experiment.process_at(i).class_name();
		if(!_experiment.process_at(i).name().empty()) {
			_experiment.print() << " (" << _experiment.process_at(i).name() << ")";
		}
		_experiment.print() << std::endl;

		_process_train_data(_experiment.process_at(i), _train_set, refresh_interval);
		_process_test_data(_experiment.process_at(i), _test_set);
		_process_output(i);
	}

	_train_set.clear();
	_test_set.clear();
}

Tensor<Time> DenseIntermediateExecution::compute_time_at(size_t i) const {
	throw std::runtime_error("Unimplemented");
}

void DenseIntermediateExecution::_load_data() {
	for(Input* input : _experiment.train_data()) {
		size_t count = 0;
		while(input->has_next()) {
			_train_set.push_back(input->next());
			count ++;
		}
		_experiment.log() << "Load " << count << " train samples from " << input->to_string() << std::endl;
		input->close();
	}

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

void DenseIntermediateExecution::_process_train_data(AbstractProcess& process, std::vector<std::pair<std::string, Tensor<float>>>& data, size_t refresh_interval) {
	size_t n = process.train_pass_number();

	if(n == 0) {
		throw std::runtime_error("train_pass_number() should be > 0");
	}

	for(size_t i=0; i<n; i++) {
		for(size_t j=0; j<data.size(); j++) {
			process.process_train_sample(data[j].first, data[j].second, i, j, data.size());

			if(i == n-1 && data[j].second.shape() != process.shape()) {
				throw std::runtime_error("Unexpected shape (actual: "+data[j].second.shape().to_string()+", expected: "+process.shape().to_string()+")");
			}

			_experiment.tick(process.index(), i*data.size()+j);

			if((i*data.size()+j) % refresh_interval == 0) {
				_experiment.refresh(process.index());
			}

		}
	}
}

void DenseIntermediateExecution::_process_test_data(AbstractProcess& process, std::vector<std::pair<std::string, Tensor<float>>>& data) {
	for(size_t j=0; j<_test_set.size(); j++) {
		process.process_test_sample(data[j].first, data[j].second, j, data.size());
		if(data[j].second.shape() != process.shape()) {
			throw std::runtime_error("Unexpected shape (actual: "+data[j].second.shape().to_string()+", expected: "+process.shape().to_string()+")");
		}
	}
}

void DenseIntermediateExecution::_process_output(size_t index) {
	for(size_t i=0; i<_experiment.output_count(); i++) {
		if(_experiment.output_at(i).index() == index) {
			Output& output = _experiment.output_at(i);

			std::vector<std::pair<std::string, Tensor<float>>> output_train_set;
			std::vector<std::pair<std::string, Tensor<float>>> output_test_set;

			for(std::pair<std::string, Tensor<float>>& entry : _train_set) {
				output_train_set.emplace_back(entry.first, output.converter().process(entry.second));
			}

			for(std::pair<std::string, Tensor<float>>& entry : _test_set) {
				output_test_set.emplace_back(entry.first, output.converter().process(entry.second));
			}


			for(Process* process : output.postprocessing()) {
				_experiment.print() << "Process " << process->class_name() << std::endl;
				_process_train_data(*process, output_train_set, std::numeric_limits<size_t>::max());
				_process_test_data(*process, output_test_set);
			}

			for(Analysis* analysis : output.analysis()) {

				_experiment.log() << output.name() << ", analysis " << analysis->class_name() << ":" << std::endl;

				size_t n = analysis->train_pass_number();

				for(size_t j=0; j<n; j++) {
					analysis->before_train_pass(j);
					for(std::pair<std::string, Tensor<float>>& entry : output_train_set) {
						analysis->process_train_sample(entry.first, entry.second, j);
					}
					analysis->after_train_pass(j);
				}

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
