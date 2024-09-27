#include "dataset/Cifar.h"
#include <iostream>

using namespace dataset;


Cifar::Cifar(const std::vector<std::string>& files) :
	_files(files), _reader(files.front(), std::ios::in | std::ios::binary),
	_file_cursor(0), _shape({CIFAR_WIDTH, CIFAR_HEIGHT, CIFAR_DEPTH}), _next_label(0) {

	if(!_reader.is_open()) {
		throw std::runtime_error("Unable to open "+files.front());
	}

	check_next();
}

bool Cifar::has_next() const {
	return _file_cursor < _files.size();
}


std::pair<std::string, Tensor<InputType>> Cifar::next() {
	assert(!_reader.eof());

	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_next_label)), _shape);

	for(size_t z=0; z<CIFAR_DEPTH; z++) {
		for(size_t y=0; y<CIFAR_HEIGHT; y++) {
			for(size_t x=0; x<CIFAR_WIDTH; x++) {
				uint8_t pixel;
				_reader.read((char*)&pixel, sizeof(uint8_t));
				out.second.at(x, y, z) = static_cast<InputType>(pixel)/static_cast<InputType>(std::numeric_limits<uint8_t>::max());
			}
		}
	}

	//_reader.setstate(std::ios_base::eofbit);

	check_next();

	return out;
}

void Cifar::reset() {
	_file_cursor = 0;
	if(_reader.is_open()) {
		_reader.close();
	}
	_reader.open(_files.front(), std::ios::in | std::ios::binary);
}

void Cifar::close() {
	_reader.close();
}

size_t Cifar::size() const {
	return 0;
}

std::string Cifar::to_string() const {
	return "Cifar("+_files.front()+", ...)";
}

const Shape& Cifar::shape() const {
	return _shape;
}

void Cifar::check_next() {
	_reader.read((char*)&_next_label, sizeof(uint8_t));

	if(_reader.eof()) {
		_file_cursor++;
		_reader.close();
		if(_file_cursor < _files.size()) {
			_reader.open(_files[_file_cursor], std::ios::in | std::ios::binary);

			if(!_reader.is_open()) {
				throw std::runtime_error("Unable to open "+_files[_file_cursor]);
			}

			check_next();
		}
	}
}
