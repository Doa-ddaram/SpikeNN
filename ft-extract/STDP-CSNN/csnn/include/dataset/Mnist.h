#ifndef _DATASET_MNIST_H
#define _DATASET_MNIST_H

#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>

#include "Tensor.h"
#include "Input.h"

#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define MNIST_DEPTH 1

namespace dataset {

	class Mnist : public Input {

	public:
		Mnist(const std::string& image_filename, const std::string& label_filename, size_t max_read = std::numeric_limits<size_t>::max());

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape& shape() const;

	private:
		void read_header();
		uint32_t swap(uint32_t v);

		std::string _image_filename;
		std::string _label_filename;

		std::ifstream _image_file;
		std::ifstream _label_file;

		uint32_t _size;
		uint32_t _cursor;

		Shape _shape;

		uint32_t _max_read;
	};

}

#endif
