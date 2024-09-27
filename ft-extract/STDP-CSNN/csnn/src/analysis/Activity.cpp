#include "analysis/Activity.h"
#include "Experiment.h"

using namespace analysis;

static RegisterClassParameter<Activity, AnalysisFactory> _register("Activity");


Activity::Activity() : UniquePassAnalysis(_register),
	_sparsity(0), _activity(0), _quiet(0), _count(0), _size(0) {

}

void Activity::resize(const Shape& shape) {
	_size = shape.product();
}

void Activity::before_train() {
	_reset();
	experiment().log() << "===Activity===" << std::endl;
	experiment().log() << "* train set:" << std::endl;
}

void Activity::process_train(const std::string&, const Tensor<float>& sample) {
	_process(sample);
}

void Activity::after_train() {
	_print();
}

void Activity::before_test() {
	_reset();
	experiment().log() << "* test set:" << std::endl;
}

void Activity::process_test(const std::string&, const Tensor<float>& sample) {
	_process(sample);
}

void Activity::after_test() {
	_print();
}

void Activity::_reset() {
	_sparsity = 0;
	_activity = 0;
	_quiet = 0;
	_count = 0;
}

void Activity::_process(const Tensor<float>& sample) {

	double c_sparsityn = 0;
	double c_sparsityd = 0;
	double c_activity = 0;

	for(size_t i=0; i<_size; i++) {
		c_sparsityn += std::abs(sample.at_index(i));
		c_sparsityd += sample.at_index(i)*sample.at_index(i);
		c_activity += sample.at_index(i) > 0 ? 1 : 0;
	}

	if(c_activity == 0) {
		_quiet++;
	}

	size_t tmp_1 = c_sparsityd == 0 ? 1.0 : std::max(1.0, c_sparsityn/std::sqrt(c_sparsityd));

	_sparsity += (std::sqrt(_size)-tmp_1)/(std::sqrt(_size)-1);
	_activity += c_activity/static_cast<double>(_size);
	_count++;
}

void Activity::_print() {
	experiment().log() << "Sparsity: " << _sparsity/static_cast<double>(_count) << std::endl;
	experiment().log() << "Active unit: " << (_activity/static_cast<double>(_count))*100.0 << "%" << std::endl;
	experiment().log() << "Quiet: " << (static_cast<double>(_quiet)/static_cast<double>(_count))*100.0 << "%" << std::endl;
}
