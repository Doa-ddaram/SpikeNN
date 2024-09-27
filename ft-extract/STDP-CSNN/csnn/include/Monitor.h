#ifndef _MONITOR_H
#define _MONITOR_H

#include <vector>
#include <string>
#include "Tensor.h"
#include "ClassParameter.h"

class AbstractExperiment;

class Monitor : public ClassParameter {

public:
	template<typename T, typename Factory>
	Monitor(const RegisterClassParameter<T, Factory>& registration) : ClassParameter(registration) {

	}


	virtual void on_sample(const AbstractExperiment& experiment, size_t layer_index, size_t sample_count) = 0;
	virtual void on_epoch(const AbstractExperiment& experiment, size_t layer_index, size_t epoch_count) = 0;
private:
};

class MonitorFactory : public ClassParameterFactory<Monitor, MonitorFactory> {


public:
	MonitorFactory() : ClassParameterFactory<Monitor, MonitorFactory>("Monitor") {

	}

};

#endif
