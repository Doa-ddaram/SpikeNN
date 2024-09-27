#ifndef _INPUT_CONVERTER_H
#define _INPUT_CONVERTER_H

#include "ClassParameter.h"
#include "Spike.h"
#include "Process.h"

class InputConverter : public AbstractProcess {

public:
	template<typename T, typename Factory>
	InputConverter(const RegisterClassParameter<T, Factory>& registration) : AbstractProcess(registration) {

	}

	virtual Shape compute_shape(const Shape& shape);
	virtual size_t train_pass_number() const;
	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number);
	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number);


	virtual void process(const Tensor<float> &in, Tensor<Time>& out) = 0;
};

class InputConverterFactory : public ClassParameterFactory<InputConverter, InputConverterFactory> {

public:
	InputConverterFactory() : ClassParameterFactory<InputConverter, InputConverterFactory>("InputConverter") {

	}

};


class LatencyCoding : public InputConverter {

public:
	LatencyCoding();

	virtual void process(const Tensor<float> &in, Tensor<Time>& out);
};

class RankOrderCoding : public InputConverter {

public:
	RankOrderCoding();

	virtual void process(const Tensor<float> &in, Tensor<Time>& out);
};



#endif
