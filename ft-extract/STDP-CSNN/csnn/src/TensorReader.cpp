#include "TensorReader.h"

TensorReader::TensorReader() :
	_name(), _file(), _tensor_counter(0), _cursor(0), _sparse(false) {

}

TensorReader::TensorReader(const std::string& filename) :
	_name(filename), _file(), _tensor_counter(0), _cursor(0), _sparse(false) {
	open(filename);
}

TensorReader::TensorReader(TensorReader&& that) noexcept :
	_name(that._name), _file(std::move(that._file)), _tensor_counter(that._tensor_counter), _cursor(that._cursor), _sparse(that._sparse) {

}

TensorReader::~TensorReader() {
	close();
}

void TensorReader::open(const std::string& filename) {
	close();
	_name = filename;
	_file.open(filename, std::ios::in | std::ios::binary);

	if(!_file.is_open()) {
		throw std::runtime_error("Unable to open "+filename);
	}

	_read_header();

}

bool TensorReader::eof() const {
	return _file.eof();
}

bool TensorReader::has_next() const {
	return _cursor <_tensor_counter;
}

std::pair<std::string, Tensor<InputType>> TensorReader::next() {
	uint8_t label_size = 0;
	_file.read(reinterpret_cast<char*>(&label_size), sizeof(uint8_t));
	char* label_buffer = new char[label_size];
	_file.read(label_buffer, label_size);

	std::string label(label_buffer, label_size);
	delete[] label_buffer;

	uint8_t dim_number = 0;
	_file.read(reinterpret_cast<char*>(&dim_number), sizeof(uint8_t));
	std::vector<size_t> dims;
	for(size_t i = 0; i<dim_number; i++) {
		uint16_t dim = 0;
		_file.read(reinterpret_cast<char*>(&dim), sizeof(uint16_t));
		dims.push_back(dim);
	}
	Shape shape(dims);
	std::pair<std::string, Tensor<InputType>> t(label, shape);

	if(_sparse) {
		t.second.fill(0);
		uint32_t index;
		_file.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
		while(index != 0xFFFFFFFF) {
			float f;
			_file.read(reinterpret_cast<char*>(&f), sizeof(float));
			t.second.at_index(index) = f;
			_file.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
		}
	}
	else {
		_file.read(reinterpret_cast<char*>(t.second.begin()), sizeof(float)*shape.product());
	}

	_cursor++;

	return t;
}

void TensorReader::reset() {
	_file.seekg(0, std::ios::beg);
	_cursor = 0;
	_read_header();
}

void TensorReader::close() {
	if(_file.is_open()) {
		_file.close();
	}
}

void TensorReader::_read_header() {
	uint32_t magic = 0;
	_file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

	if(magic == 0x234264FF) { //v1
		uint8_t flag = 0;
		_file.read(reinterpret_cast<char*>(&flag), sizeof(uint8_t));
		_sparse = (flag & 0x1) != 0;
		uint32_t counter = 0;
		_file.read(reinterpret_cast<char*>(&counter), sizeof(uint32_t));
		_tensor_counter = counter;
	}
	else { //v0
		_tensor_counter = magic;
	}
}

std::string TensorReader::to_string() const {
	return "TensorReader("+_name+")";
}

const Shape& TensorReader::shape() const {
	throw std::runtime_error("Unimplemented");
}
