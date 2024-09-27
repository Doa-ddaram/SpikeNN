#ifndef _ANALYSIS_SAVE_OUTPUT_H
#define _ANALYSIS_SAVE_OUTPUT_H

#include "Analysis.h"
#include "TensorWriter.h"

namespace analysis {

	class SaveOutput : public UniquePassAnalysis {

	public:
		SaveOutput();
		SaveOutput(const std::string& train_filename, const std::string& test_filename, bool sparse);
		SaveOutput(const SaveOutput& that) = delete;
		SaveOutput& operator=(const SaveOutput& that) = delete;

		void resize(const Shape&);

		void before_train();
		void process_train(const std::string& label, const Tensor<float>& sample);
		void after_train();

		void before_test();
		void process_test(const std::string& label, const Tensor<float>& sample);
		void after_test();

	private:
		std::string _train_filename;
		std::string _test_filename;
		bool _sparse;

		TensorWriter* _writer;
	};

}

#endif
