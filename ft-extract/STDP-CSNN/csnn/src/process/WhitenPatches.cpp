#include "process/WhitenPatches.h"

using namespace process;

static RegisterClassParameter<WhiteningPatches, ProcessFactory> _register("WhiteningPatches");

WhiteningPatches::WhiteningPatches() : TwoPassProcess(_register), _eps(0), _pca_compress(0),
	_patch_size(0), _stride(1), _max_sample(), _list(), _filter() {
	add_parameter("eps", _eps);
	add_parameter("pca_compress", _pca_compress);
	add_parameter("patch_size", _patch_size);
	add_parameter("stride", _stride);
	add_parameter("max_sample", _max_sample);
}

WhiteningPatches::WhiteningPatches(size_t patch_size, float eps, float pca_compress, size_t stride, size_t max_sample) : WhiteningPatches() {
	parameter<float>("eps").set(eps);
	parameter<float>("pca_compress").set(pca_compress);
	parameter<size_t>("patch_size").set(patch_size);
	parameter<size_t>("stride").set(stride);
	parameter<size_t>("max_sample").set(max_sample);
}

Shape WhiteningPatches::compute_shape(const Shape& shape) {
	return shape;
}

void WhiteningPatches::compute(const std::string&, const Tensor<float>& sample) {
	if(sample.shape().number() != 3) {
		throw std::runtime_error("Require 3D inputs");
	}
	size_t width = sample.shape().dim(0);
	size_t height = sample.shape().dim(1);
	size_t depth = sample.shape().dim(2);
	if(_list.size() < _max_sample) {
		Tensor<float> patch(Shape({_patch_size, _patch_size, depth}));
		for(size_t x=0; x<width-_patch_size+1; x+=_stride) {
			for(size_t y=0; y<height-_patch_size+1; y+=_stride) {

				for(size_t fx=0; fx<_patch_size; fx++) {
					for(size_t fy=0; fy<_patch_size; fy++) {
						for(size_t fz=0; fz<depth; fz++) {
							patch.at(fx, fy, fz) = sample.at(x+fx, y+fy, fz);
						}
					}
				}

				_list.emplace_back(patch);
			}
		}


	}
}

void WhiteningPatches::process_train(const std::string&, Tensor<float>& sample) {
	size_t depth = sample.shape().dim(2);

	if(!_list.empty()) {
		size_t rows = _list.front().shape().product();
		size_t cols = _list.size();

		Tensor<float> x(Shape({rows, cols}));

		for(size_t i=0; i<rows; i++) {
			for(size_t j=0; j<cols; j++) {
				x.at(i, j) = _list[j].at_index(i);
			}
		}

		_list.clear();
		_list.shrink_to_fit();

		Tensor<float> x_mean = mean1(x);
		x = x-expand_in(x_mean, Shape({cols}));

		Tensor<float> cov = transpose(dot(x, transpose(x))/create(cols, Shape({rows, rows})));

		Tensor<float> u(Shape({rows, rows}));
		//Tensor<float> v(Shape({rows, rows}));
		Tensor<float> s(Shape({rows}));

		int m = rows;
		int n = rows;
		int lda = rows;
		int ldu = rows;
		int ldvt = rows;
		int lwork = -1;

		int info;
		float tmp_work;
		sgesvd_("A", "N", &m, &n, cov.begin(), &lda, s.begin(), u.begin(), &ldu, nullptr, &ldvt, &tmp_work, &lwork, &info);
		lwork = tmp_work;
		if(info != 0) {
			throw std::runtime_error("Error in sgesvd_ (1):"+std::to_string(info));
		}
		Tensor<float> work(Shape({lwork}));
		sgesvd_("A", "N", &m, &n, cov.begin(), &lda, s.begin(), u.begin(), &ldu, nullptr, &ldvt, work.begin(), &lwork, &info);
		if(info != 0) {
			throw std::runtime_error("Error in sgesvd_ (2):"+std::to_string(info));
		}

		size_t start_compress = static_cast<float>(rows)*_pca_compress;
		for(size_t i=start_compress; i<rows; i++) {
			//std::cout << i << ":" << s.at(i) << "-> 0" << std::endl;
			//s.at(i) = 0;
			for(size_t j=0; j<rows; j++) {
				u.at(i, j) = 0;
			}
		}

		Tensor<float> w = transpose(dot(transpose(u), dot(diag(create(1.0, Shape({rows}))/sqrt(s+create(_eps, Shape({rows})))), u)));

		for(size_t z=0; z<depth; z++) {
			Tensor<float> inpulse(Shape({_patch_size, _patch_size, depth}));
			inpulse.fill(0);
			inpulse.at(_patch_size/2, _patch_size/2, z) = 1.0;

			Tensor<float> response = reshape(dot(flatten(inpulse), w), inpulse.shape());
			_filter.push_back(response-create(mean(response), response.shape()));
		}
	}

	_apply(sample);
}

void WhiteningPatches::process_test(const std::string&, Tensor<float>& sample) {
	_apply(sample);
}

void WhiteningPatches::_apply(Tensor<float>& sample) const {
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

void WhiteningPatches::save(const std::string& filename) const {
	std::ofstream file(filename, std::ios::out | std::ios::trunc | std::ios::binary);

	if(!file.good()) {
		throw std::runtime_error("Unable to open "+filename);
	}

	TwoPassProcess::save(file);

	uint32_t n_filter = _filter.size();
	file.write(reinterpret_cast<const char*>(&n_filter), sizeof(uint32_t));
	for(size_t i=0; i<_filter.size(); i++) {
		Persistence::save_tensor(_filter[i], file);
	}

	file.close();
}
