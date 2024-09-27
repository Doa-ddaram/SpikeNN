#ifndef _FEATURE_WRITER_H
#define _FEATURE_WRITER_H

#include <fstream>

#include "Tensor.h"

class FeatureWriter {

public:
	FeatureWriter();

	void open(const std::string& filename);

	template<typename T>
	void write(const std::string& label, const Tensor<T>& t) {
		if(label.empty()) {
			throw std::runtime_error("Empty label");
		}

		_file << label;

		size_t size = t.shape().product();

		for(size_t i=0; i<size; i++) {
			if(t.at_index(i) != 0.0) {
				_file << " " << (i+1) << ":" << t.at_index(i);
			}
		}
		_file << std::endl;

	}

	void close();

private:
	std::ofstream _file;


};

#endif
