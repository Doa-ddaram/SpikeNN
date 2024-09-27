#include "dataset/Mnist.h"

using namespace dataset;

Mnist::Mnist(const std::string& image_filename, const std::string& label_filename, size_t max_read) :
	_image_filename(image_filename), _label_filename(label_filename),
	_image_file(image_filename, std::ios::binary), _label_file(label_filename, std::ios::binary),
	_size(0), _cursor(0), _shape({MNIST_WIDTH, MNIST_HEIGHT, MNIST_DEPTH}), _max_read(max_read) {

	if(!_image_file.is_open()) {
		throw std::runtime_error("Can't open "+image_filename);
	}
	if(!_label_file.is_open()) {
		throw std::runtime_error("Can't open "+label_filename);
	}

	read_header();
}

bool Mnist::has_next() const {
	return _cursor < size();
}


std::pair<std::string, Tensor<InputType>> Mnist::next() {
	assert(!_label_file.eof());
	assert(!_image_file.eof());


	uint8_t label;
	_label_file.read((char*)&label, sizeof(uint8_t));

	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(label)), _shape);

	for(size_t y = 0; y < MNIST_WIDTH; y++) {
		for(size_t x = 0; x < MNIST_HEIGHT; x++) {
			uint8_t pixel;
			_image_file.read((char*)&pixel, sizeof(uint8_t));
			out.second.at(x, y, 0) = static_cast<InputType>(pixel)/static_cast<InputType>(std::numeric_limits<uint8_t>::max());
		}
	}




	_cursor++;

	return out;
}

void Mnist::reset() {
	_cursor = 0;
	_label_file.seekg(0, std::ios::beg);
	_image_file.seekg(0, std::ios::beg);
	read_header();
}

void Mnist::close() {
	_label_file.close();
	_image_file.close();
}

size_t Mnist::size() const {
	return std::min(_size, _max_read);
}

std::string Mnist::to_string() const {
	return "Mnist("+_image_filename+", "+_label_filename+")["+std::to_string(size())+"]";
}

const Shape& Mnist::shape() const {
	return _shape;
}

void Mnist::read_header() {
	// image file header
	uint32_t image_magic;
	_image_file.read((char*)&image_magic, sizeof(uint32_t));
	image_magic = swap(image_magic);
	_image_file.read((char*)&_size, sizeof(uint32_t));
	_size = swap(_size);
	uint32_t image_width;
	_image_file.read((char*)&image_width, sizeof(uint32_t));
	image_width = swap(image_width);
	uint32_t image_height;
	_image_file.read((char*)&image_height, sizeof(uint32_t));
	image_height = swap(image_height);

	assert(image_width == MNIST_WIDTH && image_height == MNIST_HEIGHT);
	assert(image_magic == 0x00000803);

	// label file header
	uint32_t label_magic;
	_label_file.read((char*)&label_magic, sizeof(uint32_t));
	label_magic = swap(label_magic);
	uint32_t label_size;
	_label_file.read((char*)&label_size, sizeof(uint32_t));
	label_size = swap(label_size);

	assert(label_magic == 0x00000801);
	assert(label_size == _size);
}

uint32_t Mnist::swap(uint32_t v) {
	return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}
