#include "analysis/SaveOutput.h"

using namespace analysis;

static RegisterClassParameter<SaveOutput, AnalysisFactory> _register("SaveOutput");

SaveOutput::SaveOutput() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename(), _sparse(false), _writer(nullptr) {
	throw std::runtime_error("Unimplemented");
}

SaveOutput::SaveOutput(const std::string& train_filename, const std::string& test_filename, bool sparse) :
	UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename), _sparse(sparse), _writer(nullptr) {

}

void SaveOutput::resize(const Shape&) {

}


void SaveOutput::before_train() {
	_writer = new TensorWriter(_train_filename, _sparse);
}

void SaveOutput::process_train(const std::string& label, const Tensor<float>& sample) {
	_writer->write(label, sample);
}

void SaveOutput::after_train() {
	_writer->close();
	delete _writer;
}

void SaveOutput::before_test() {
	_writer = new TensorWriter(_test_filename, _sparse);
}


void SaveOutput::process_test(const std::string& label, const Tensor<float>& sample) {
	_writer->write(label, sample);
}

void SaveOutput::after_test() {
	_writer->close();
	delete _writer;
}
