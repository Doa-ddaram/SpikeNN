#ifndef _PROCESS_POOLING_H
#define _PROCESS_POOLING_H


#include "Process.h"

namespace process {
	class SumPooling : public UniquePassProcess {

	public:
		SumPooling();
		SumPooling(size_t target_width, size_t target_height);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& sample) const;

		size_t _target_width;
		size_t _target_height;

		size_t _width;
		size_t _height;
		size_t _depth;
	};

	class MeanPooling : public UniquePassProcess {

	public:
		MeanPooling();
		MeanPooling(size_t target_width, size_t target_height);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& sample) const;

		size_t _target_width;
		size_t _target_height;

		size_t _width;
		size_t _height;
		size_t _depth;
	};
}

#endif
