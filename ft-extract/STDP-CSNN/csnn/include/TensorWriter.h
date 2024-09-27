#ifndef _TENSOR_WRITER_H
#define _TENSOR_WRITER_H

#include <fstream>

#include "Tensor.h"

class TensorWriter {

public:
	TensorWriter();
	TensorWriter(const std::string& filename, bool sparse = false);
	~TensorWriter();

	void open(const std::string& filename, bool sparse = false);
	void write(const std::string& label, const Tensor<float>& t);
	void close();

private:
	std::ofstream _file;
	size_t _tensor_counter;
	bool _sparse;

};

#endif
