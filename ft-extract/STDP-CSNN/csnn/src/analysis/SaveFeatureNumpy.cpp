#include "analysis/SaveFeatureNumpy.h"

using namespace analysis;

static RegisterClassParameter<SaveFeatureNumpy, AnalysisFactory> _register("SaveFeatureNumpy");

SaveFeatureNumpy::SaveFeatureNumpy() : UniquePassAnalysis(_register),
	_path("./"), _data_train(), _data_test(), _label_train(), _label_test(),
    _train_sample_cnt(0), _test_sample_cnt(0), _width(0), _height(0), _depth(0) {
	throw std::runtime_error("Unimplemented");
}

SaveFeatureNumpy::SaveFeatureNumpy(const std::string& path) : UniquePassAnalysis(_register),
	_path(path), _data_train(), _data_test(), _label_train(), _label_test(),
    _train_sample_cnt(0), _test_sample_cnt(0), _width(0), _height(0), _depth(0) {

}

void SaveFeatureNumpy::resize(const Shape&) {
 
}

void SaveFeatureNumpy::before_train() {
    _data_train.clear();
    _label_train.clear();
    _train_sample_cnt = 0;
}

void SaveFeatureNumpy::process_train(const std::string& label, const Tensor<float>& sample) {
 	_width = sample.shape().dim(0);
	_height = sample.shape().dim(1);
	_depth = sample.shape().dim(2);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			for(size_t z=0; z<_depth; z++) {
                _data_train.emplace_back(sample.at(x, y, z));
            }
        }
    }
    _label_train.emplace_back(std::stoi(label));
    _train_sample_cnt += 1;
}

void SaveFeatureNumpy::after_train() {
    const bool fortran_order{false};
    
    const std::vector<long unsigned> shape_data{_train_sample_cnt, _width, _height, _depth};
    const std::vector<long unsigned> shape_label{_train_sample_cnt};

    npy::SaveArrayAsNumpy(_path + '/' + "X_csnn_train.npy", fortran_order, shape_data.size(), shape_data.data(), _data_train);
    npy::SaveArrayAsNumpy(_path + '/' + "y_train.npy", fortran_order, shape_label.size(), shape_label.data(), _label_train);
}

void SaveFeatureNumpy::before_test() {
    _data_test.clear();
    _label_test.clear();
    _test_sample_cnt = 0;
}

void SaveFeatureNumpy::process_test(const std::string& label, const Tensor<float>& sample) {
 	_width = sample.shape().dim(0);
	_height = sample.shape().dim(1);
	_depth = sample.shape().dim(2);

	for(size_t x=0; x<_width; x++) {
		for(size_t y=0; y<_height; y++) {
			for(size_t z=0; z<_depth; z++) {
                _data_test.emplace_back(sample.at(x, y, z));
            }
        }
    }
    _label_test.emplace_back(std::stoi(label));
    _test_sample_cnt += 1;
}

void SaveFeatureNumpy::after_test() {
    const bool fortran_order{false};
    
    const std::vector<long unsigned> shape_data{_test_sample_cnt, _width, _height, _depth};
    const std::vector<long unsigned> shape_label{_test_sample_cnt};

    npy::SaveArrayAsNumpy(_path + "/" + "X_csnn_test.npy", fortran_order, shape_data.size(), shape_data.data(), _data_test);    
    npy::SaveArrayAsNumpy(_path + "/" + "y_test.npy", fortran_order, shape_label.size(), shape_label.data(), _label_test);
}
