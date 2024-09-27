#ifndef _PROCESS_ON_OFF_FILTER_H
#define _PROCESS_ON_OFF_FILTER_H

#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process {

	namespace _priv {
		class OnOffFilterHelper {

		public:
			OnOffFilterHelper() = delete;

			static Tensor<float> generate_filter(size_t filter_size, float center_dev, float surround_dev);

		};
	}

	class DefaultOnOffFilter : public UniquePassProcess {

	public:
		DefaultOnOffFilter();
		DefaultOnOffFilter(size_t filter_size, float center_dev, float surround_dev);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& in) const;

		size_t _filter_size;
		float _center_dev;
		float _surround_dev;

		size_t _width;
		size_t _height;
		size_t _depth;
		Tensor<float> _filter;
	};

	class CustomRGBOnOffFilter : public UniquePassProcess {

	public:
		CustomRGBOnOffFilter();
		CustomRGBOnOffFilter(const std::string& filename);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& in) const;

		Tensor<float> _r;
		Tensor<float> _g;
		Tensor<float> _b;

		size_t _width;
		size_t _height;
	};

	class BiologicalOnOffFilter : public UniquePassProcess {

	public:
		BiologicalOnOffFilter();
		BiologicalOnOffFilter(size_t filter_size, float center_dev, float surround_dev);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& in) const;

		size_t _filter_size;
		float _center_dev;
		float _surround_dev;

		size_t _width;
		size_t _height;
		size_t _depth;
		Tensor<float> _filter;
	};

}
#endif
