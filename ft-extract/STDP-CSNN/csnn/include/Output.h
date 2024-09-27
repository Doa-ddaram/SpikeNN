#ifndef _OUTPUT_H
#define _OUTPUT_H

#include <iostream>

#include "Process.h"
#include "Tensor.h"
#include "Spike.h"
#include "TensorWriter.h"
#include "FeatureWriter.h"
#include "ClassParameter.h"
#include "Analysis.h"
#include "OutputConverter.h"

class Output {

	friend class AbstractExperiment;

public:
	Output(AbstractExperiment* experiment, const std::string& name, size_t index, OutputConverter* converter);
	~Output();

	Output(const Output& that) = delete;
	Output& operator=(const Output& that) = delete;

	template<typename T, typename... Args>
	T& add_postprocessing(Args&&... args) {
		T* obj = new T(std::forward<Args>(args)...);
		_postprocessing.push_back(obj);
		return *obj;
	}

	template<typename T, typename... Args>
	T& add_analysis(Args&&... args) {
		T* obj = new T(std::forward<Args>(args)...);
		obj->set_info(_experiment, _index);
		_analysis.push_back(obj);
		return *obj;
	}

	const std::string& name() const;
	size_t index() const;

	OutputConverter& converter();

	std::vector<Process*>& postprocessing();
	const std::vector<Process*>& postprocessing() const;

	std::vector<Analysis*>& analysis();
	const std::vector<Analysis*>& analysis() const;

private:
	std::string _name;
	size_t _index;
	AbstractExperiment* _experiment;

	OutputConverter* _converter;
	std::vector<Process*> _postprocessing;
	std::vector<Analysis*> _analysis;
};
#endif
