#ifndef _PROCESS_WHITEN_PATCHES_LOADER_H
#define _PROCESS_WHITEN_PATCHES_LOADER_H

#include <lapacke.h>
#include <fstream>

#include "Tensor.h"
#include "Process.h"
#include "Math.h"

namespace process {

	class WhitenPatchesLoader : public UniquePassProcess {

	public:
		WhitenPatchesLoader();
		WhitenPatchesLoader(const std::string& filename);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _apply(Tensor<float>& sample) const;

		float _eps;
		float _pca_compress;

		size_t _patch_size;
		size_t _stride;
		size_t _max_sample;

		std::vector<Tensor<float>> _filter;
	};

}

#endif
