#ifndef _INPUT_H
#define _INPUT_H

#include <string>
#include "Tensor.h"

typedef float InputType;

class Input {

public:
	Input() {

	}

	virtual ~Input() {

	}

	virtual const Shape& shape() const = 0;

	virtual bool has_next() const = 0;
	virtual std::pair<std::string, Tensor<InputType>> next() = 0;
	virtual void reset() = 0;
	virtual void close() = 0;

	virtual std::string to_string() const = 0;

};

#endif
