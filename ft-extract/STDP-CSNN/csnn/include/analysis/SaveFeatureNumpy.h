#ifndef _ANALYSIS_SAVE_FEATURE_NUMPY_H
#define _ANALYSIS_SAVE_FEATURE_NUMPY_H

#include <iostream>
#include "Analysis.h"
#include "dep/npy.hpp"

namespace analysis {

	class SaveFeatureNumpy : public UniquePassAnalysis {

	public:
		SaveFeatureNumpy();
		SaveFeatureNumpy(const std::string& path);
		SaveFeatureNumpy(const SaveFeatureNumpy& that) = delete;
		SaveFeatureNumpy& operator=(const SaveFeatureNumpy& that) = delete;

		void resize(const Shape&);

		void before_train();
		void process_train(const std::string& label, const Tensor<float>& sample);
		void after_train();

		void before_test();
		void process_test(const std::string& label, const Tensor<float>& sample);
		void after_test();

	private:

		std::string _path;

		std::vector<float> _data_train;
		std::vector<float> _data_test;
	
		std::vector<int> _label_train;
		std::vector<int> _label_test;

		long unsigned _train_sample_cnt;
		long unsigned _test_sample_cnt;
		long unsigned _width;
		long unsigned _height;
		long unsigned _depth;	
	};

}

#endif
