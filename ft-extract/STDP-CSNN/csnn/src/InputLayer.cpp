#include "InputLayer.h"
#include "Experiment.h"

//
//	InputLayer
//

InputLayer::InputLayer(AbstractExperiment* experiment, InputConverter* converter) :
	_experiment(experiment), _converter(converter), _width(0), _height(0), _depth(0) {

}

InputLayer::~InputLayer() {
	delete _converter;
}

void InputLayer::resize(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
}

InputConverter& InputLayer::converter() {
	return *_converter;
}

size_t InputLayer::width() const {
	return _width;
}

size_t InputLayer::height() const {
	return _height;
}

size_t InputLayer::depth() const {
	return _depth;
}

#ifdef ENABLE_QT
void InputLayer::plot_time(bool only_in_train) {
	add_plot<plot::TimeHistogram>(only_in_train, _experiment, 0);
}

void InputLayer::_add_plot(Plot* plot, bool) {
	_experiment->add_plot(plot, -1);
}
#endif
