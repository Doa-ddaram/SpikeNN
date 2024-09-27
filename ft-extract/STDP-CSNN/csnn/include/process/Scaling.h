#ifndef _PROCESS_SCALING_H
#define _PROCESS_SCALING_H

#include "Tensor.h"
#include "Process.h"
#include "dep/npy.hpp"


namespace process {

	class FeatureScaling : public TwoPassProcess {

	public:
		FeatureScaling();

		virtual Shape compute_shape(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);
		virtual bool save_params(const std::string& path);
		virtual bool load_params(const std::string& path);

	private:
		size_t _size;
		Tensor<float> _min;
		Tensor<float> _max;
	};

	class ChannelScaling : public TwoPassProcess {

	public:
		ChannelScaling();

		virtual Shape compute_shape(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		size_t _width;
		size_t _height;
		size_t _depth;
		Tensor<float> _min;
		Tensor<float> _max;
	};


	class SampleScaling : public TwoPassProcess {

	public:
		SampleScaling();

		virtual Shape compute_shape(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		size_t _size;
		float _min;
		float _max;

	};

	class IndependentScaling : public UniquePassProcess {

	public:
		IndependentScaling();

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	};

}

#endif
