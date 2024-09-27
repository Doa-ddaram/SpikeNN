#ifndef _CONVOLTUION_H
#define _CONVOLTUION_H

#include "Layer.h"
#include "Stdp.h"
#include "dep/npy.hpp"
#include "plot/Threshold.h"
#include "plot/Evolution.h"

namespace layer {

class Convolution;

namespace _priv {

#ifdef SMID_AVX256
class ConvolutionImpl {

public:
	ConvolutionImpl(Convolution& model);

	void resize();
	void train(const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike);
	void test(const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike);
private:
	Convolution& _model;
	Tensor<float> _a;
	Tensor<float> _inh;
	Tensor<bool> _wta;
};

#else
class ConvolutionImpl {

public:
	ConvolutionImpl(Convolution& model);

	void resize();
	void train(const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike);
	void test(const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike);

private:
	Convolution& _model;
	Tensor<float> _a;
	Tensor<bool> _inh;
	Tensor<bool> _wta;
};
#endif
}

class Convolution : public Layer3D {

	friend class _priv::ConvolutionImpl;

public:
	Convolution();
	Convolution(size_t filter_width, size_t filter_height, size_t filter_number, size_t stride_x = 1, size_t stride_y = 1, size_t padding_x = 0, size_t padding_y = 0);
	Convolution(const Convolution& that) = delete;
	Convolution& operator=(const Convolution& that) = delete;

	virtual Shape compute_shape(const Shape& previous_shape);

	virtual size_t train_pass_number() const;
	virtual void process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number);
	virtual void process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number);
	virtual bool load_params(const std::string& path);
	virtual bool save_params(const std::string& path);

	virtual void train(const std::string& label, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike);
	virtual void test(const std::string& label, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike);
	virtual void on_epoch_end();

	virtual Tensor<float> reconstruct(const Tensor<float>& t) const;

	void plot_threshold(bool only_in_train);
	void plot_evolution(bool only_in_train);

private:

	uint32_t _epoch_number;
	float _annealing;

	float _min_th;
	float _t_obj;
	float _lr_th;

	Tensor<float> _w;
	Tensor<float> _th;
	STDP* _stdp;
	size_t _input_depth;

	bool _wta_infer;

	_priv::ConvolutionImpl _impl;
};
}
#endif
