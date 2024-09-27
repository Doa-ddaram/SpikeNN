#include "process/Whiten.h"

using namespace process;

static RegisterClassParameter<Whitening, ProcessFactory> _register("Whitening");

Whitening::Whitening() : TwoPassProcess(_register), _eps(0), _pca_compress(0), _max_sample(), _list(), _w() {
	add_parameter("eps", _eps);
	add_parameter("pca_compress", _pca_compress);
	add_parameter("max_sample", _max_sample);
}

Whitening::Whitening(float eps, float pca_compress, size_t max_sample) : Whitening() {
	parameter<float>("eps").set(eps);
	parameter<float>("pca_compress").set(pca_compress);
	parameter<size_t>("max_sample").set(max_sample);
}

Shape Whitening::compute_shape(const Shape& shape) {
	return shape;
}

void Whitening::compute(const std::string&, const Tensor<float>& sample) {
	if(_list.size() < _max_sample) {
		_list.emplace_back(sample);
	}
}

void Whitening::process_train(const std::string&, Tensor<float>& sample) {
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
		//debug_tensor(x ,"X");
		_mean = mean1(x);
		//debug_tensor(_mean ,"Mean");
		x = x-expand_in(_mean, Shape({cols}));
		//debug_tensor(x ,"X_center");

		Tensor<float> cov = transpose(dot(x, transpose(x))/create(cols, Shape({rows, rows})));
		//debug_tensor(cov, "Cov");
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
			s.at(i) = 0;
		}

		_w = transpose(dot(transpose(u), dot(diag(create(1.0, Shape({rows}))/sqrt(s+create(_eps, Shape({rows})))), u)));
		//debug_tensor(_w, "W");
	}

	Tensor<float> x_center = flatten(sample)-_mean;
	//debug_tensor(x_center, "X_center");
	sample = reshape(dot(x_center, _w), sample.shape());
	//debug_tensor(sample, "X_whi");
}

void Whitening::process_test(const std::string&, Tensor<float>& sample) {
	Tensor<float> x_center = flatten(sample)-_mean;
	sample = reshape(dot(x_center, _w), sample.shape());
}
