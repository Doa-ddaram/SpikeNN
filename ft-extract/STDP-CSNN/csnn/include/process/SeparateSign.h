#ifndef _PROCESS_SEPARATE_SIGN_H
#define _PROCESS_SEPARATE_SIGN_H

#include "Tensor.h"
#include "Process.h"
#include "Math.h"

namespace process {

	class SeparateSign : public UniquePassProcess {

	public:
		SeparateSign();

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	};

}

#endif
