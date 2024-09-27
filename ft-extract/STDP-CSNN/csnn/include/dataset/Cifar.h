#ifndef _DATASET_CIFAR_H
#define _DATASET_CIFAR_H

#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>

#include "Tensor.h"
#include "Input.h"

#define CIFAR_WIDTH 32
#define CIFAR_HEIGHT 32
#define CIFAR_DEPTH 3

namespace dataset {

	class Cifar : public Input {

	public:
		Cifar(const std::vector<std::string>& files);

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape& shape() const;

	private:
		void check_next();

		std::vector<std::string> _files;

		std::ifstream _reader;

		uint32_t _file_cursor;

		Shape _shape;

		uint8_t _next_label;
	};
}

#endif
