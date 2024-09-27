#ifndef _ANALYSIS_SVM_H
#define _ANALYSIS_SVM_H

#include "Analysis.h"
#include "dep/libsvm/svm.h"

namespace analysis {
	class Svm : public TwoPassAnalysis {

	public:
		Svm();

		Svm(const Svm& that) = delete;
		Svm& operator=(const Svm& that) = delete;

		virtual void resize(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, const Tensor<float>& sample);
		virtual void process_test(const std::string& label, const Tensor<float>& sample);

		virtual void before_train();
		virtual void after_train();
		virtual void before_test();
		virtual void after_test();

	private:
		float _c;

		std::map<std::string, double> _label_index;
		size_t _size;
		size_t _node_count;
		size_t _sample_count;


		svm_problem _problem;
		svm_model* _model;
		svm_node* _train_nodes;
		svm_node* _test_nodes;

		size_t _correct_sample;
		size_t _total_sample;
	};
}

#endif
