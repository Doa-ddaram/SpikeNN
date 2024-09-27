#ifndef _DATASET_IMAGEBIN_H
#define _DATASET_IMAGEBIN_H

#include <string>
#include <cassert>
#include <fstream>
#include <limits>

#include "Tensor.h"
#include "Input.h"


namespace dataset {

	// Generic data loader for 3D images
	// Images must be stored in binary files, with uint8 pixels 
	class ImageBin : public Input {

	public:
		ImageBin(const std::string& image_filename, const std::string& label_filename, int width, int height, int depth, const std::string& dataset_name);

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

		std::string _name;
	};

}

#endif
