#include "process/WhitenPatchesLoader.h"

using namespace process;

static RegisterClassParameter<WhitenPatchesLoader, ProcessFactory> _register("WhitenPatchesLoader");

WhitenPatchesLoader::WhitenPatchesLoader() : UniquePassProcess(_register), _eps(0), _pca_compress(0),
	_patch_size(0), _stride(1), _max_sample(), _filter() {
	add_parameter("eps", _eps);
	add_parameter("pca_compress", _pca_compress);
	add_parameter("patch_size", _patch_size);
	add_parameter("stride", _stride);
	add_parameter("max_sample", _max_sample);
}

WhitenPatchesLoader::WhitenPatchesLoader(const std::string& filename) : WhitenPatchesLoader() {

	std::ifstream file(filename, std::ios::in | std::ios::binary);

	if(!file.good()) {
		throw std::runtime_error("Unable to open "+filename);
	}

	uint32_t magic;
	file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

	if(magic != ClassParameter::Magic) {
		throw std::runtime_error("Invalid ClassParameter magic");
	}

	if(Persistence::load_string(file) != "Process") {
		throw std::runtime_error("Invalid identifier");
	}
	if(Persistence::load_string(file) != "WhiteningPatches") {
		throw std::runtime_error("Invalid identifier");
	}
	UniquePassProcess::load(file);

	uint32_t n_filter;
	file.read(reinterpret_cast<char*>(&n_filter), sizeof(uint32_t));
	for(size_t i=0; i<n_filter; i++) {
		_filter.push_back(Persistence::load_tensor<float>(file));
	}

	file.close();
}

Shape WhitenPatchesLoader::compute_shape(const Shape& shape) {
	return shape;
}

void WhitenPatchesLoader::process_train(const std::string&, Tensor<float>& sample) {
	_apply(sample);
}

void WhitenPatchesLoader::process_test(const std::string&, Tensor<float>& sample) {
	_apply(sample);
}

void WhitenPatchesLoader::_apply(Tensor<float>& sample) const {
	if(sample.shape().number() != 3) {
		throw std::runtime_error("Require 3D inputs");
	}
	size_t width = sample.shape().dim(0);
	size_t height = sample.shape().dim(1);
	size_t depth = sample.shape().dim(2);

	Tensor<float> out(Shape({width, height, depth}));

	for(size_t z=0; z<depth; z++) {
		for(size_t x=0; x<width; x++) {
			for(size_t y=0; y<height; y++) {
				float v = 0;
				for(size_t fx=0; fx<_patch_size; fx++) {
					for(size_t fy=0; fy<_patch_size; fy++) {
						size_t x_in = x+fx > _patch_size/2 ? std::min(x+fx-_patch_size/2, width-1) : 0;
						size_t y_in = y+fy > _patch_size/2 ? std::min(y+fy-_patch_size/2, height-1) : 0;

						for(size_t z1=0; z1<depth; z1++) {
							v += sample.at(x_in, y_in, z1)*_filter.at(z).at(fx, fy, z1);
						}
					}
				}
				out.at(x, y, z) = v;
			}
		}
	}
	sample = out;
}
