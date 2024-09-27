#include "TensorWriter.h"

TensorWriter::TensorWriter() : _file(), _tensor_counter(0),_sparse(false) {

}

TensorWriter::TensorWriter(const std::string& filename, bool sparse) : TensorWriter() {
	open(filename, sparse);
}

TensorWriter::~TensorWriter() {
	close();
}

void TensorWriter::open(const std::string& filename, bool sparse) {
	if(_file.is_open()) {
		throw std::runtime_error("File already open");
	}

	_file.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);
	_sparse = sparse;

	if(!_file.is_open()) {
		throw std::runtime_error("Unable to open "+filename);
	}

	uint32_t v1_magic = 0x234264FF;
	_file.write(reinterpret_cast<const char*>(&v1_magic), sizeof(uint32_t));

	uint8_t flag = (sparse ? 0x1 : 0x0);
	_file.write(reinterpret_cast<const char*>(&flag), sizeof(uint8_t));

	uint32_t counter = 0;
	_file.write(reinterpret_cast<const char*>(&counter), sizeof(uint32_t));
}

void TensorWriter::write(const std::string& label, const Tensor<float>& t) {
	if(!_file.good()) {
		throw std::runtime_error("No open file");
	}

	uint8_t label_size = label.size();
	_file.write(reinterpret_cast<const char*>(&label_size), sizeof(uint8_t));
	_file.write(label.c_str(), label_size);

	uint8_t dim_number = t.shape().number();
	_file.write(reinterpret_cast<const char*>(&dim_number), sizeof(uint8_t));
	for(size_t i = 0; i<dim_number; i++) {
		uint16_t dim = t.shape().dim(i);
		_file.write(reinterpret_cast<const char*>(&dim), sizeof(uint16_t));
	}

	if(_sparse) {
		size_t size = t.shape().product();
		for(uint32_t i=0; i<size; i++) {
			if(t.at_index(i) != 0.0) {
				 _file.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
				 float f = t.at_index(i);
				 _file.write(reinterpret_cast<const char*>(&f), sizeof(float));
			}
		}
		uint32_t eol = 0xFFFFFFFF;
		_file.write(reinterpret_cast<const char*>(&eol), sizeof(uint32_t));

	}
	else {
		_file.write(reinterpret_cast<const char*>(t.begin()), sizeof(float)*t.shape().product());
	}


	_file.flush();
	_tensor_counter++;
}

void TensorWriter::close() {
	if(_file.is_open()) {
		_file.clear();
		_file.seekp(sizeof(uint32_t)+sizeof(uint8_t), std::ios::beg);

		uint32_t counter = _tensor_counter;
		_file.write(reinterpret_cast<const char*>(&counter), sizeof(uint32_t));
		_file.flush();
		_file.close();
	}
}
