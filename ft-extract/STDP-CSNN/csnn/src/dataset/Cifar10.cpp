#include "dataset/Cifar10.h"

using namespace dataset;


Cifar10::Cifar10(const std::string& image_filename, const std::string& label_filename) :
	_image_filename(image_filename), _label_filename(label_filename),
	_image_file(image_filename, std::ios::in | std::ios::binary), _label_file(label_filename, std::ios::in | std::ios::binary),
	_shape({CIFAR10_WIDTH, CIFAR10_HEIGHT, CIFAR10_DEPTH}), _next_label(0) {

	if(!_image_file.is_open()) {
		throw std::runtime_error("Can't open "+image_filename);
	}
	if(!_label_file.is_open()) {
		throw std::runtime_error("Can't open "+label_filename);
	}

	_prepare_next();
}

bool Cifar10::has_next() const {
	return !_label_file.eof();
}


std::pair<std::string, Tensor<InputType>> Cifar10::next() {
	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_next_label)), _shape);

	for(size_t x=0; x<CIFAR10_WIDTH; x++) {
		for(size_t y=0; y<CIFAR10_HEIGHT; y++) {
			for(size_t z=0; z<CIFAR10_DEPTH; z++) {
				uint8_t pixel;
				_image_file.read((char*)&pixel, sizeof(uint8_t));

				out.second.at(x, y, z) = static_cast<InputType>(pixel)/static_cast<InputType>(std::numeric_limits<uint8_t>::max());
			}
		}
	}

	_prepare_next();

	return out;
}

void Cifar10::reset() {
	_label_file.seekg(0, std::ios::beg);
	_image_file.seekg(0, std::ios::beg);
	_prepare_next();
}


void Cifar10::close() {
	_label_file.close();
	_image_file.close();
}

size_t Cifar10::size() const {
	return 0;
}

std::string Cifar10::to_string() const {
	return "Cifar10("+_image_filename+", "+_label_filename+")";
}

const Shape& Cifar10::shape() const {
	return _shape;
}

void Cifar10::_prepare_next() {
	_label_file.read((char*)&_next_label, sizeof(uint8_t));
}
