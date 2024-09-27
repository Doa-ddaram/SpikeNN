#include "Output.h"
#include "Experiment.h"

//
//	Output
//

Output::Output(AbstractExperiment* experiment, const std::string& name, size_t index, OutputConverter* converter) :
	 _name(name), _index(index), _experiment(experiment), _converter(converter), _postprocessing(), _analysis() {

}

Output::~Output() {
	delete _converter;

	for(Process* p : _postprocessing) {
		delete p;
	}

	for(Analysis* a : _analysis) {
		delete a;
	}
}

const std::string& Output::name() const {
	return _name;
}

size_t Output::index() const {
	return _index;
}

OutputConverter& Output::converter() {
	return *_converter;
}

std::vector<Process*>& Output::postprocessing() {
	return _postprocessing;
}

const std::vector<Process*>& Output::postprocessing() const {
	return _postprocessing;
}

std::vector<Analysis*>& Output::analysis() {
	return _analysis;
}

const std::vector<Analysis*>& Output::analysis() const {
	return _analysis;
}
