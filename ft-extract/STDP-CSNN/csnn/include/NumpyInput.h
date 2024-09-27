#ifndef _NUMPY_INPUT_H
#define _NUMPY_INPUT_H

#include "Input.h"
#include "NumpyReader.h"

class NumpyInput : public Input {

public:
	NumpyInput(const std::string& filename);
	NumpyInput(const NumpyInput& that) = delete;
	NumpyInput& operator=(const NumpyInput& that) = delete;

	virtual bool has_next() const;
	virtual std::pair<std::string, Tensor<InputType>> next();
	virtual void reset();
	virtual void close();

	virtual std::string to_string() const;

	virtual const Shape& shape() const;

private:
	std::string _name;
	NumpyArchive _archive;
	NumpyArray* _data;
	NumpyArray* _label;

	size_t _current;
};

#endif
