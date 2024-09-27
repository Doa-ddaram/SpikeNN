#include "dataset/MnistCustom.h"

using namespace dataset;


MnistCustom::MnistCustom(const std::string& image_filename, const std::string& label_filename) :
	_image_filename(image_filename), _label_filename(label_filename),
	_image_file(image_filename, std::ios::in | std::ios::binary), _label_file(label_filename, std::ios::in | std::ios::binary),
	_shape({MNISTC_WIDTH, MNISTC_HEIGHT, MNISTC_DEPTH}), _next_label(0) {

	if(!_image_file.is_open()) {
		throw std::runtime_error("Can't open "+image_filename);
	}
	if(!_label_file.is_open()) {
		throw std::runtime_error("Can't open "+label_filename);
	}

	_prepare_next();
}

bool MnistCustom::has_next() const {
	return !_label_file.eof();
}


std::pair<std::string, Tensor<InputType>> MnistCustom::next() {
	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_next_label)), _shape);

	for(size_t x=0; x<MNISTC_WIDTH; x++) {
		for(size_t y=0; y<MNISTC_HEIGHT; y++) {
			for(size_t z=0; z<MNISTC_DEPTH; z++) {
				uint8_t pixel;
				_image_file.read((char*)&pixel, sizeof(uint8_t));

				out.second.at(x, y, z) = static_cast<InputType>(pixel)/static_cast<InputType>(std::numeric_limits<uint8_t>::max());
			}
		}
	}

	_prepare_next();

	return out;
}

void MnistCustom::reset() {
	_label_file.seekg(0, std::ios::beg);
	_image_file.seekg(0, std::ios::beg);
	_prepare_next();
}


void MnistCustom::close() {
	_label_file.close();
	_image_file.close();
}

size_t MnistCustom::size() const {
	return 0;
}

std::string MnistCustom::to_string() const {
	return "MnistCustom("+_image_filename+", "+_label_filename+")";
}

const Shape& MnistCustom::shape() const {
	return _shape;
}

void MnistCustom::_prepare_next() {
	_label_file.read((char*)&_next_label, sizeof(uint8_t));
}
