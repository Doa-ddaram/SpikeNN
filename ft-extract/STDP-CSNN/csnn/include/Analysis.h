#ifndef _ANALYSIS_H
#define _ANALYSIS_H

#include <vector>
#include <string>
#include "Tensor.h"
#include "ClassParameter.h"

class AbstractExperiment;

class Analysis : public ClassParameter {

public:
	template<typename T, typename Factory>
	Analysis(const RegisterClassParameter<T, Factory>& registration) :
		ClassParameter(registration), _experiment(nullptr), _layer_index(0) {

	}

	Analysis(const Analysis& that) = delete;

	Analysis& operator=(const Analysis& that) = delete;

	void set_info(const AbstractExperiment* experiment, size_t layer_index) {
		_experiment = experiment;
		_layer_index = layer_index;
	}

	virtual void resize(const Shape& shape) = 0;
	virtual size_t train_pass_number() const = 0;
	virtual void process_train_sample(const std::string& label, const Tensor<float>& sample, size_t current_pass) = 0;
	virtual void process_test_sample(const std::string& label, const Tensor<float>& sample) = 0;

	virtual void before_train_pass(size_t) {

	}

	virtual void after_train_pass(size_t) {

	}

	virtual void before_test() {

	}

	virtual void after_test() {

	}

	const AbstractExperiment& experiment() const {
		return *_experiment;
	}

	size_t layer_index() const {
		return _layer_index;
	}

private:
	const AbstractExperiment* _experiment;
	size_t _layer_index;

};

class NoPassAnalysis : public Analysis {

public:
	template<typename T, typename Factory>
	NoPassAnalysis(const RegisterClassParameter<T, Factory>& registration) : Analysis(registration) {

	}

	virtual size_t train_pass_number() const final {
		return 0;
	}

	virtual void process_train_sample(const std::string&, const Tensor<float>&, size_t) final {

	}

	virtual void process_test_sample(const std::string&, const Tensor<float>&) final {

	}

	virtual void after_test() final {
		process();
	}

	virtual void process() = 0;
};

class UniquePassAnalysis : public Analysis {

public:
	template<typename T, typename Factory>
	UniquePassAnalysis(const RegisterClassParameter<T, Factory>& registration) : Analysis(registration) {

	}

	virtual size_t train_pass_number() const final {
		return 1;
	}

	virtual void process_train_sample(const std::string& label, const Tensor<float>& sample, size_t) final {
		process_train(label, sample);
	}

	virtual void process_test_sample(const std::string& label, const Tensor<float>& sample) final {
		process_test(label, sample);
	}

	virtual void before_train_pass(size_t) final {
		before_train();
	}

	virtual void after_train_pass(size_t) final {
		after_train();
	}

	virtual void process_train(const std::string& label, const Tensor<float>& sample) = 0;
	virtual void process_test(const std::string& label, const Tensor<float>& sample) = 0;

	virtual void before_train() {

	}

	virtual void after_train() {

	}
};

class TwoPassAnalysis : public Analysis {

public:
	template<typename T, typename Factory>
	TwoPassAnalysis(const RegisterClassParameter<T, Factory>& registration) : Analysis(registration) {

	}

	virtual size_t train_pass_number() const final {
		return 2;
	}

	virtual void process_train_sample(const std::string& label, const Tensor<float>& sample, size_t current_pass) final {
		if(current_pass == 0) {
			compute(label, sample);
		}
		else {
			process_train(label, sample);
		}
	}

	virtual void before_train_pass(size_t current_pass) final {
		if(current_pass == 0)
			before_compute();
		else
			before_train();
	}

	virtual void after_train_pass(size_t current_pass) final {
		if(current_pass == 0)
			after_compute();
		else
			after_train();
	}

	virtual void process_test_sample(const std::string& label, const Tensor<float>& sample) final {
		process_test(label, sample);
	}

	virtual void compute(const std::string& label, const Tensor<float>& sample) = 0;
	virtual void process_train(const std::string& label, const Tensor<float>& sample) = 0;
	virtual void process_test(const std::string& label, const Tensor<float>& sample) = 0;

	virtual void before_compute() {

	}

	virtual void after_compute() {

	}

	virtual void before_train() {

	}

	virtual void after_train() {

	}
};



class AnalysisFactory : public ClassParameterFactory<Analysis, AnalysisFactory> {


public:
	AnalysisFactory() : ClassParameterFactory<Analysis, AnalysisFactory>("Analysis") {

	}

};

#endif
