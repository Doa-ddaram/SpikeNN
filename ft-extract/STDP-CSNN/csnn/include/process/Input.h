#ifndef _PROCESS_INPUT_H
#define _PROCESS_INPUT_H

#include "Process.h"
#include "Color.h"

namespace process {

	class GaussianTemporalCoding : public UniquePassProcess {

	public:
		GaussianTemporalCoding();
		GaussianTemporalCoding(float center, float scale);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& t) const;

		float _center;
		float _scale;

		size_t _width;
		size_t _height;
		size_t _depth;
	};

#ifdef ENABLE_QT
	class GaussianTemporalCodingColor {

	public:
		static QColor get(const std::vector<float>& list);

	};
#endif

}

#endif
