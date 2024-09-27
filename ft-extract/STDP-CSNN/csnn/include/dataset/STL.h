#ifndef _DATASET_STL_H
#define _DATASET_STL_H

#include <string>
#include <cassert>
#include <fstream>
#include <limits>

#include "Tensor.h"
#include "Input.h"

#define STL_WIDTH 96
#define STL_HEIGHT 96
#define STL_DEPTH 3

namespace dataset {

	class STL : public Input {

	public:
		STL(const std::string& image_filename, const std::string& label_filename);

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape& shape() const;

	private:
		void _prepare_next();

		std::string _image_filename;
		std::string _label_filename;

		std::ifstream _image_file;
		std::ifstream _label_file;


		Shape _shape;

		uint8_t _next_label;
	};

}

#endif
