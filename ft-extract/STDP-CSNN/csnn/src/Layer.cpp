#include "Layer.h"
#include "Experiment.h"

Layer::~Layer() {

}

void Layer::set_size(size_t width, size_t height) {
	if(width < 1 || height < 1) {
		throw std::runtime_error("Min size is 1x1");
	}
	if(width > _width || height > _height) {
		throw std::runtime_error("Size can't exceed layer dimension");
	}

	_current_width = width;
	_current_height = height;
}

size_t Layer::width() const {
	return _width;
}

size_t Layer::height() const {
	return _height;
}

size_t Layer::depth() const {
	return _depth;
}

bool Layer::require_sorted() const {
	return _require_sorted;
}

#ifdef ENABLE_QT
void Layer::plot_time(bool only_in_train, size_t n, float min, float max) {
	add_plot<plot::TimeHistogram>(only_in_train, experiment(), index()+1, n, min, max);
}

void Layer::_add_plot(Plot* plot, bool only_in_train) {
	experiment()->add_plot(plot, only_in_train ? index() : -1);
}
#endif

std::vector<const Layer*> Layer::_previous_layer_list() const {
	std::vector<const Layer*> layers;
	for(int i=index(); i>=0; i--) {
		if(dynamic_cast<const Layer*>(&experiment()->process_at(i))) {
			layers.push_back(dynamic_cast<const Layer*>(&experiment()->process_at(i)));
		}
	}
	return layers;
}

//
//	Layer3D
//

Shape Layer3D::compute_shape(const Shape& previous_shape) {
	parameter<size_t>("filter_width").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_height").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("filter_number").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_x").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("stride_y").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_x").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("padding_y").ensure_initialized(experiment()->random_generator());

	size_t previous_width = previous_shape.dim(0);
	size_t previous_height = previous_shape.dim(1);

	if(previous_width+2*_padding_x < _filter_width || previous_height+2*_padding_y<_filter_height) {
		throw std::runtime_error("Filter dimension need to be smaller than the input");
	}

	_width = (previous_width+2*_padding_x-_filter_width)/_stride_x+1;
	_height = (previous_height+2*_padding_y-_filter_height)/_stride_y+1;
	_depth = _filter_number;

	return Shape({_width, _height, _depth});
}

// Search for output neurons integrating input of position (x_in, y_in)
// For each involved output neuron, we get the position of the input in its window (i.e. the weight modulating the input)
void Layer3D::forward(uint16_t x_in, uint16_t y_in, std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>>& output) {
	size_t s_x = x_in+_padding_x >= _filter_width-_stride_x ? (x_in+_padding_x-(_filter_width-_stride_x))/_stride_x : 0;
	size_t s_y = y_in+_padding_y >= _filter_height-_stride_y ? (y_in+_padding_y-(_filter_height-_stride_y))/_stride_y : 0;


	size_t l_x = (x_in+_padding_x)/_stride_x;
	size_t l_y = (y_in+_padding_y)/_stride_y;

	for(size_t x = s_x; x <= l_x && x < _current_width; x++) {
		for(size_t y = s_y; y <= l_y && y < _current_height; y++) {

			size_t w_x = x_in+_padding_x-x*_stride_x;
			size_t w_y = y_in+_padding_y-y*_stride_y;

			output.emplace_back(x, y, w_x, w_y);

		}
	}
}

std::pair<uint16_t, uint16_t> Layer3D::to_input_coord(uint16_t x, uint16_t y, uint16_t w_x, uint16_t w_y) const {
	if(x+w_x < _padding_x || y+w_y < _padding_y)
		return std::pair<uint16_t, uint16_t>(std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max());
	else
		return std::pair<uint16_t, uint16_t>(x+w_x-_padding_x, y+w_y-_padding_y);
}

bool Layer3D::is_valid_input_coord(const std::pair<uint16_t, uint16_t>& coord) const {
	return coord.first != std::numeric_limits<uint16_t>::max();
}

std::pair<uint16_t, uint16_t> Layer3D::receptive_field_of(const std::pair<uint16_t, uint16_t>& in) const {
	return std::pair<uint16_t, uint16_t>((in.first-1)*_stride_x+_filter_width, (in.second-1)*_stride_y+_filter_height);
}
