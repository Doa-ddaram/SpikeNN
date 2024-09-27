#ifndef _INPUT_LAYER_H
#define _INPUT_LAYER_H

#include "Tensor.h"
#include "Spike.h"
#include "Plot.h"
#include "ClassParameter.h"
#include "InputConverter.h"

class AbstractExperiment;

class InputLayer {

	friend class AbstractExperiment;

public:
	InputLayer(AbstractExperiment* experiment, InputConverter* converter);

	InputLayer(const InputLayer& that) = delete;
	~InputLayer();
	InputLayer& operator=(const InputLayer& that) = delete;

	void resize(const Shape& shape);

	InputConverter& converter();
	size_t width() const;
	size_t height() const;
	size_t depth() const;

	void plot_time(bool only_in_train);

protected:
#ifdef ENABLE_QT
	template<typename PlotType, typename... Args>
	void add_plot(bool only_in_train, Args&&... args) {

		_add_plot(new PlotType("Input", std::forward<Args>(args)...), only_in_train);
	}
#endif

private:
#ifdef ENABLE_QT
	void _add_plot(Plot* plot, bool only_in_train);
#endif

	AbstractExperiment* _experiment;
	InputConverter* _converter;

	size_t _width;
	size_t _height;
	size_t _depth;

};


#endif
