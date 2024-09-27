#ifndef _PROCESS_H
#define _PROCESS_H

#include <iostream>

#include "Input.h"
#include "Color.h"
#include "ClassParameter.h"


class AbstractExperiment;

class AbstractProcess : public ClassParameter {

	friend class AbstractExperiment;

public:
	template<typename T, typename Factory>
	AbstractProcess(const RegisterClassParameter<T, Factory>& registration) : ClassParameter(registration),
		_output_shape(), _index(std::numeric_limits<size_t>::max()), _experiment(nullptr){

	}

	AbstractProcess(const AbstractProcess& that) = delete;
	AbstractProcess& operator=(const AbstractProcess& that) = delete;

	//virtual void initialize(std::default_random_engine& random_engine) = 0;

	virtual size_t train_pass_number() const = 0;
	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number) = 0;
	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number) = 0;
	virtual bool load_params(const std::string&) { return false; } // To load trained parameters of the process (if it has any)
	virtual bool save_params(const std::string&) { return false; } // To save trained parameters of the process (if it has any)

	const Shape& shape() const;
	const Shape& resize(const Shape& shape);

	size_t index() const;

protected:
	virtual Shape compute_shape(const Shape& shape) = 0;

	AbstractExperiment* experiment();
	const AbstractExperiment* experiment() const;

private:
	void _set_info(size_t index, AbstractExperiment* experiment);

	Shape _output_shape;

	size_t _index;
	AbstractExperiment* _experiment;
};

class Process : public AbstractProcess {

public:
	template<typename T, typename Factory>
	Process(const RegisterClassParameter<T, Factory>& registration) : AbstractProcess(registration) {

	}
};

class UniquePassProcess : public Process {

public:
	template<typename T, typename Factory>
	UniquePassProcess(const RegisterClassParameter<T, Factory>& registration) : Process(registration) {

	}

	virtual size_t train_pass_number() const {
		return 1;
	}

	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t, size_t, size_t) {
		process_train(label, sample);
	}

	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t, size_t) {
		process_test(label, sample);
	}

	virtual void process_train(const std::string& label, Tensor<float>& sample) = 0;
	virtual void process_test(const std::string& label, Tensor<float>& sample) = 0;
};

class TwoPassProcess : public Process {

public:
	template<typename T, typename Factory>
	TwoPassProcess(const RegisterClassParameter<T, Factory>& registration) : Process(registration) {

	}

	virtual size_t train_pass_number() const {
		return 2;
	}

	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t, size_t) {
		if(current_pass == 0) {
			compute(label, sample);
		}
		else {
			process_train(label, sample);
		}
	}

	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t, size_t) {
		process_test(label, sample);
	}

	virtual void compute(const std::string& label, const Tensor<float>& sample) = 0;
	virtual void process_train(const std::string& label, Tensor<float>& sample) = 0;
	virtual void process_test(const std::string& label, Tensor<float>& sample) = 0;
};



class ProcessFactory : public ClassParameterFactory<Process, ProcessFactory> {


public:
	ProcessFactory() : ClassParameterFactory<Process, ProcessFactory>("Process") {

	}

};


#endif
