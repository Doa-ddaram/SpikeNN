#ifndef _OUTPUT_CONVERTER_H
#define _OUTPUT_CONVERTER_H

#include "ClassParameter.h"
#include "Spike.h"
#include "Process.h"

class OutputConverter : public AbstractProcess {

public:
	template<typename T, typename Factory>
	OutputConverter(const RegisterClassParameter<T, Factory>& registration) : AbstractProcess(registration) {

	}

	virtual Shape compute_shape(const Shape& shape);
	virtual size_t train_pass_number() const;
	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number);
	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number);

	virtual Tensor<float> process(const Tensor<float>& in) = 0;
};

class OutputConverterFactory : public ClassParameterFactory<OutputConverter, OutputConverterFactory> {

public:
	OutputConverterFactory() : ClassParameterFactory<OutputConverter, OutputConverterFactory>("OutputConverter") {

	}

};

class NoOutputConversion : public OutputConverter {

public:
	NoOutputConversion();

	virtual Tensor<float> process(const Tensor<float>& in);
};

class DefaultOutput : public OutputConverter {

public:
	DefaultOutput();
	DefaultOutput(Time min, Time max);

	virtual Tensor<float> process(const Tensor<float>& in);

private:
	Time _min;
	Time _max;

};


class TimeObjectiveOutput : public OutputConverter {

public:
    TimeObjectiveOutput();
	TimeObjectiveOutput(Time t_obj);

	virtual Tensor<float> process(const Tensor<float>& in);

private:
	Time _t_obj;

};

class SoftMaxOutput : public OutputConverter {

public:
	SoftMaxOutput();

	virtual Tensor<float> process(const Tensor<float>& in);
};

class WTAOutput : public OutputConverter {

public:
	WTAOutput();

	virtual Tensor<float> process(const Tensor<float>& in);
};

class SpikeTiming : public OutputConverter {

public:
	SpikeTiming();

	virtual Tensor<float> process(const Tensor<float>& in);
};



#endif
