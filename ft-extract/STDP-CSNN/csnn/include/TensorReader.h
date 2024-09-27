#ifndef _TENSOR_READER_H
#define _TENSOR_READER_H

#include <fstream>
#include <iostream>

#include "Input.h"

class TensorReader : public Input {

public:
	TensorReader();
	TensorReader(const std::string& filename);
	~TensorReader();

	TensorReader(TensorReader&& that) noexcept;

	void open(const std::string& filename);

	bool eof() const;

	virtual bool has_next() const;
	virtual std::pair<std::string, Tensor<InputType>> next();

	void close();
	void reset();

	virtual std::string to_string() const;

	virtual const Shape& shape() const;

private:
	void _read_header();

	std::string _name;
	std::ifstream _file;
	size_t _tensor_counter;
	size_t _cursor;
	bool _sparse;

};

#endif
