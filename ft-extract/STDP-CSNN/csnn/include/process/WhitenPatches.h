#ifndef _PROCESS_WHITEN_PATCHES_H
#define _PROCESS_WHITEN_PATCHES_H

#include <lapacke.h>
#include <fstream>

#include "Tensor.h"
#include "Process.h"
#include "Math.h"

namespace process {

	class WhiteningPatches : public TwoPassProcess {

	public:
		WhiteningPatches();
		WhiteningPatches(size_t patch_size, float eps, float pca_compress = 1.0, size_t stride = 1, size_t max_sample = std::numeric_limits<size_t>::max());

		virtual Shape compute_shape(const Shape& shape);
		virtual void compute(const std::string& label, const Tensor<float>& sample);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

		void save(const std::string& filename) const;

	private:
		void _apply(Tensor<float>& sample) const;

		float _eps;
		float _pca_compress;

		size_t _patch_size;
		size_t _stride;
		size_t _max_sample;

		std::vector<Tensor<float>> _list;
		std::vector<Tensor<float>> _filter;
	};

}

#endif
