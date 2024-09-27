#include "FeatureWriter.h"

FeatureWriter::FeatureWriter() : _file() {

}

void FeatureWriter::open(const std::string& filename) {
	if(_file.is_open()) {
		throw std::runtime_error("File already open");
	}

	_file.open(filename, std::ios::out | std::ios::trunc );

	if(!_file.is_open()) {
		throw std::runtime_error("Unable to open "+filename);
	}
}

void FeatureWriter::close() {
	_file.flush();
	_file.close();
}
