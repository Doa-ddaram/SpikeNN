#include "Process.h"

AbstractExperiment* AbstractProcess::experiment() {
	return _experiment;
}

const AbstractExperiment* AbstractProcess::experiment() const {
	return _experiment;
}

void AbstractProcess::_set_info(size_t index, AbstractExperiment* experiment) {
	_index = index;
	_experiment = experiment;
}

const Shape& AbstractProcess::shape() const {
	return _output_shape;
}

const Shape& AbstractProcess::resize(const Shape& shape) {
	_output_shape = compute_shape(shape);
	return _output_shape;
}

size_t AbstractProcess::index() const {
	return _index;
}
