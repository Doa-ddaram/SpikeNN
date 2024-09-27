#include "NumpyInput.h"

NumpyInput::NumpyInput(const std::string& filename) : _name(filename), _archive(), _data(nullptr), _label(nullptr), _current(0) {
	NumpyReader::load(filename, _archive);
	_data = &_archive.at("arr_0.npy");
	_label = &_archive.at("arr_1.npy");

	if(_data->dimension_number() != 4) {
		throw std::runtime_error("[NumpyInput] Unknown format (expected 4-dimension data tensor)");
	}

	if(_label->dimension_number() != 1) {
		throw std::runtime_error("[NumpyInput] Unknown format (expected 1-dimension label tensor)");
	}
	if(_data->dimension(0) != _label->dimension(0)) {
		throw std::runtime_error("[NumpyInput] Incompatible data and label tensor");
	}
}

bool NumpyInput::has_next() const {
	return _current < _label->dimension(0);
}

std::pair<std::string, Tensor<InputType>> NumpyInput::next() {
	size_t width = _data->dimension(1);
	size_t height = _data->dimension(2);
	size_t depth = _data->dimension(3);

	std::pair<std::string, Tensor<InputType>> out(std::to_string(_label->at(_current)), Shape({width, height, depth}));

	for(size_t x=0; x<width; x++) {
		for(size_t y=0; y<height; y++) {
			for(size_t z=0; z<depth; z++) {
				out.second.at(x, y, z) = _data->at(_current, x, y, z);
			}
		}
	}

	_current++;
	return out;
}

void NumpyInput::reset() {
	_current = 0;
}

void NumpyInput::close() {
	_archive.clear();
	_data = nullptr;
	_label = nullptr;
}

std::string NumpyInput::to_string() const {
	return "NumpyArchive("+_name+")";
}

const Shape& NumpyInput::shape() const {
	throw std::runtime_error("Unimplemented");
}
