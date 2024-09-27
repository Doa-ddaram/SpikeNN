#include "layer/Convolution.h"
#include "Experiment.h"

using namespace layer;

static RegisterClassParameter<Convolution, LayerFactory> _register("Convolution");

Convolution::Convolution() : Layer3D(_register),
	_epoch_number(0), _annealing(1.0), _min_th(0),_t_obj(0), _lr_th(0),
	_w(), _th(), _stdp(nullptr), _input_depth(0), _wta_infer(false), _impl(*this) {

	add_parameter("epoch", _epoch_number);
	add_parameter("annealing", _annealing, 1.0f);

	add_parameter("min_th", _min_th);
	add_parameter("t_obj", _t_obj);
	add_parameter("lr_th", _lr_th);

	add_parameter("w", _w);
	add_parameter("th", _th);

	add_parameter("wta_infer", _wta_infer);

	add_parameter("stdp", _stdp);
}

Convolution::Convolution(size_t filter_width, size_t filter_height, size_t filter_number, size_t stride_x, size_t stride_y, size_t padding_x, size_t padding_y) :
	Layer3D(_register, filter_width, filter_height, filter_number, stride_x, stride_y, padding_x, padding_y),
	_annealing(1.0), _min_th(0),_t_obj(0), _lr_th(0),
	_w(), _th(), _stdp(nullptr), _input_depth(0), _wta_infer(false), _impl(*this) {

	add_parameter("epoch", _epoch_number);
	add_parameter("annealing", _annealing, 1.0f);

	add_parameter("min_th", _min_th);
	add_parameter("t_obj", _t_obj);
	add_parameter("lr_th", _lr_th);

	add_parameter("w", _w);
	add_parameter("th", _th);

	add_parameter("wta_infer", _wta_infer);

	add_parameter("stdp", _stdp);

	parameter<Tensor<float>>("w").shape(0);
	parameter<Tensor<float>>("th").shape(0);
}

Shape Convolution::compute_shape(const Shape& previous_shape) {
	Layer3D::compute_shape(previous_shape);

	_input_depth = previous_shape.dim(2);

	parameter<Tensor<float>>("w").shape(_filter_width, _filter_height, _input_depth, _filter_number);
	parameter<Tensor<float>>("th").shape(_filter_number);

	_impl.resize();

	return Shape({_width, _height, _depth});
}

bool Convolution::save_params(const std::string& path) {
	std::vector<float> weights;
	for(size_t i=0; i<_filter_width; i++) {
		for(size_t j=0; j<_filter_height; j++) {
			for(size_t k=0; k<_input_depth; k++) {
				for(size_t l=0; l<_filter_number; l++) {
					weights.emplace_back(_w.at(i, j, k, l));
				}
			}
		}
	}
	std::vector<float> thresholds;
	for(size_t i=0; i<_filter_number; i++) {
		thresholds.emplace_back(_th.at(i));
	}
	const bool fortran_order{false};
	const std::vector<long unsigned> shape_weights{_filter_width, _filter_height, _input_depth, _filter_number};
	npy::SaveArrayAsNumpy(path + "/weights.npy", fortran_order, shape_weights.size(), shape_weights.data(), weights);    
	const std::vector<long unsigned> shape_thresholds{_filter_number};
	npy::SaveArrayAsNumpy(path + "/thresholds.npy", fortran_order, shape_thresholds.size(), shape_thresholds.data(), thresholds);
	return true;
}

bool Convolution::load_params(const std::string& path) {
	bool fortran_order = false;
	// Weights
	std::vector<float> weights;
	std::vector<long unsigned> shape_weights{_filter_width, _filter_height, _input_depth, _filter_number};	
	npy::LoadArrayFromNumpy(path + "/weights.npy", shape_weights, fortran_order, weights);
	int index = 0;
	for(size_t x=0; x<_filter_width; x++) {
		for(size_t y=0; y<_filter_height; y++) {
			for(size_t z=0; z<_input_depth; z++) {
				for(size_t t=0; t<_filter_number; t++) {
					_w.at(x, y, z, t) = weights.at(index);
					index++;
				}
			}
		}
	}
	// Thresholds
	std::vector<float> thresholds;
	std::vector<long unsigned> shape_thresholds{_filter_number};	
	npy::LoadArrayFromNumpy(path + "/thresholds.npy", shape_thresholds, fortran_order, thresholds);
	for(size_t index=0; index<thresholds.size(); index++) {
		_th.at(index) = thresholds.at(index);
	}
	return true;
}

size_t Convolution::train_pass_number() const {
	return _epoch_number+1;
}

void Convolution::process_train_sample(const std::string& label, Tensor<float>& sample, size_t current_pass, size_t current_index, size_t number) {

	if(current_index == 0) {
		if(current_pass < _epoch_number) {
			_current_width = 1;
			_current_height = 1;
			std::cout << "Epoch " << current_pass << "/" << _epoch_number << std::endl;
			on_epoch_start();
		}
		else {
			_current_width = _width;
			_current_height = _height;
			std::cout << "Process train set" << std::endl;
		}
	}


	std::vector<Spike> input_spike;
	std::vector<Spike> output_spike;

	if(current_pass < _epoch_number) {
		size_t x = 0;
		size_t y = 0;

		if(_filter_width < _width) {
			std::uniform_int_distribution<size_t> rand_x(0, _width-_filter_width);
			x = rand_x(experiment()->random_generator());
		}

		if(_filter_height < _height) {
			std::uniform_int_distribution<size_t> rand_y(0, _height-_filter_height);
			y = rand_y(experiment()->random_generator());
		}

		Tensor<Time> input_time(Shape({_filter_width, _filter_height, _input_depth}));
		for(size_t cx=0; cx<_filter_width; cx++) {
			for(size_t cy=0; cy<_filter_height; cy++) {
				for(size_t cz=0; cz<_input_depth; cz++) {
					input_time.at(cx, cy, cz) = sample.at(cx+x, cy+y, cz);
				}
			}
		}

		SpikeConverter::to_spike(input_time, input_spike);
		train(label, input_spike, input_time, output_spike);
	}
	else {
		SpikeConverter::to_spike(sample, input_spike);
		test(label, input_spike, sample, output_spike);
		sample = Tensor<float>(shape());
		SpikeConverter::from_spike(output_spike, sample);
	}

	if(current_index == number-1 && current_pass < _epoch_number) {
		on_epoch_end();
	}
}

void Convolution::process_test_sample(const std::string& label, Tensor<float>& sample, size_t current_index, size_t number) {
	if(current_index == 0) {
		std::cout << "Process test set" << std::endl;
		_current_width = _width;
		_current_height = _height;
	}

	std::vector<Spike> input_spike;
	SpikeConverter::to_spike(sample, input_spike);
	std::vector<Spike> output_spike;
	test(label, input_spike, sample, output_spike);
	sample = Tensor<float>(shape());
	SpikeConverter::from_spike(output_spike, sample);
}

void Convolution::train(const std::string&, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike) {
	_impl.train(input_spike, input_time, output_spike);
}

void Convolution::test(const std::string&, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike) {
	_impl.test(input_spike, input_time, output_spike);
}

void Convolution::on_epoch_end() {
	_lr_th *= _annealing;
	_stdp->adapt_parameters(_annealing);
}

Tensor<float> Convolution::reconstruct(const Tensor<float>& t) const {
	size_t k = 1;

	size_t output_width = t.shape().dim(0);
	size_t output_height = t.shape().dim(1);
	size_t output_depth = t.shape().dim(2);

	Tensor<float> out(Shape({output_width*_stride_x+_filter_width-1, output_height*_stride_y+_filter_height-1, _input_depth}));
	out.fill(0);

	Tensor<float> norm(Shape({output_width*_stride_x+_filter_width-1, output_height*_stride_y+_filter_height-1, _input_depth}));
	norm.fill(0);

	for(size_t x=0; x<output_width; x++) {
		for(size_t y=0; y<output_height; y++) {

			std::vector<size_t> is;
			for(size_t z=0; z<output_depth; z++) {
				is.push_back(z);
			}

			std::sort(std::begin(is), std::end(is), [&t, x, y](size_t i1, size_t i2) {
				return t.at(x, y, i1) > t.at(x, y, i2);
			});

			for(size_t i=0; i<k; i++) {
				if(t.at(x, y, is[i]) >= 0.0) {
					for(size_t xf=0; xf<_filter_width; xf++) {
						for(size_t yf=0; yf<_filter_height; yf++) {
							for(size_t zf=0; zf<_input_depth; zf++) {
								out.at(x*_stride_x+xf, y*_stride_y+yf, zf) += _w.at(xf, yf, zf, is[i])*t.at(x, y, is[i]);
								norm.at(x*_stride_x+xf, y*_stride_y+yf, zf) += t.at(x, y, is[i]);
							}
						}
					}
				}
			}

		}
	}

	size_t s = out.shape().product();
	for(size_t i=0; i<s; i++) {
		if(norm.at_index(i) != 0)
			out.at_index(i) /= norm.at_index(i);
	}


	out.range_normalize();

	return out;
}

#ifdef ENABLE_QT
void Convolution::plot_threshold(bool only_in_train) {
	add_plot<plot::Threshold>(only_in_train, _th);
}

void Convolution::plot_evolution(bool only_in_train) {
	add_plot<plot::Evolution>(only_in_train, _w);
}
#endif

#ifdef SMID_AVX256
#include <immintrin.h>

#define AVX_256_N 8

static __m256i _generate_mask(int n) {

	int f = 0x00000000;
	int t = 0xFFFFFFFF;

	switch(n) {

	case 0:
		return _mm256_setr_epi32(f, f, f, f, f, f, f, f);
	case 1:
		return _mm256_setr_epi32(t, f, f, f, f, f, f, f);
	case 2:
		return _mm256_setr_epi32(t, t, f, f, f, f, f, f);
	case 3:
		return _mm256_setr_epi32(t, t, t, f, f, f, f, f);
	case 4:
		return _mm256_setr_epi32(t, t, t, t, f, f, f, f);
	case 5:
		return _mm256_setr_epi32(t, t, t, t, t, f, f, f);
	case 6:
		return _mm256_setr_epi32(t, t, t, t, t, t, f, f);
	case 7:
		return _mm256_setr_epi32(t, t, t, t, t, t, t, f);
	default:
		return _mm256_setr_epi32(t, t, t, t, t, t, t, t);
	}
}

_priv::ConvolutionImpl::ConvolutionImpl(Convolution& model) : _model(model), _a(), _inh(), _wta() {

}

void _priv::ConvolutionImpl::resize() {
	_a = Tensor<float>(Shape({_model.width(), _model.height(), _model.depth()}));
	_inh = Tensor<float>(Shape({_model.width(), _model.height(), _model.depth()}));
	_wta = Tensor<bool>(Shape({_model.width(), _model.height()}));
}

void _priv::ConvolutionImpl::train(const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>&) {

	size_t depth = _model.depth();

	Tensor<float>& w = _model._w;
	Tensor<float>& th = _model._th;

	size_t n = depth/AVX_256_N;
	size_t r = depth%AVX_256_N;

	__m256 __c1 = _mm256_setzero_ps();
	__m256 __c2 = _mm256_castsi256_ps( _mm256_set1_epi32(0xFFFFFFFF));

	__m256i __mask = _generate_mask(r);

	for(size_t i=0; i<n; i++) {
		_mm256_storeu_ps(_a.ptr_index(i*AVX_256_N), __c1);
	}

	if(r > 0) {
		_mm256_maskstore_ps(_a.ptr_index(n*AVX_256_N), __mask, __c1);
	}

	for(const Spike& spike : input_spike) {


		for(size_t i=0; i<n; i++) {
			__m256 __w = _mm256_loadu_ps(w.ptr(spike.x, spike.y, spike.z, i*AVX_256_N));
			__m256 __a = _mm256_loadu_ps(_a.ptr_index(i*AVX_256_N));
			__m256 __th = _mm256_loadu_ps(th.ptr_index(i*AVX_256_N));
			__a = _mm256_add_ps(__a, __w);
			_mm256_storeu_ps(_a.ptr_index(i*AVX_256_N), __a);
			__m256 __c = _mm256_cmp_ps(__a, __th, _CMP_GE_OQ);


			if(_mm256_testz_ps(__c, __c2) == 0) {

				for(size_t j=0; j<AVX_256_N; j++) {
					if(_a.at_index(i*AVX_256_N+j) > th.at_index(i*AVX_256_N+j)) {
						for(size_t z1=0; z1<depth; z1++) {
							if (_model._t_obj != -1) {
								th.at(z1) -= _model._lr_th*(spike.time-_model._t_obj);
							}
							if(z1 != i*AVX_256_N+j) {
								th.at(z1) -= _model._lr_th/static_cast<float>(depth-1);
							}
							else {
								th.at(z1) += _model._lr_th;
								//if (_model._t_obj != -1) {
								//	th.at(z1) -= _model._lr_th*(spike.time-_model._t_obj);
								//}
							}
							th.at(z1) = std::max<float>(_model._min_th, th.at(z1));
						}


						for(size_t x=0; x<_model._filter_width; x++) {
							for(size_t y=0; y<_model._filter_height; y++) {
								for(size_t z=0; z<_model._input_depth; z++) {
									w.at(x, y, z, i*AVX_256_N+j) = _model._stdp->process(w.at(x, y, z, i*AVX_256_N+j), input_time.at(x, y, z), spike.time);
								}
							}
						}

						return;
					}
				}
			}
		}

		if(r > 0) {
			__m256 __w = _mm256_maskload_ps(w.ptr(spike.x, spike.y, spike.z, n*AVX_256_N), __mask);
			__m256 __a = _mm256_maskload_ps(_a.ptr_index(n*AVX_256_N), __mask);
			__a = _mm256_add_ps(__a, __w);
			_mm256_maskstore_ps(_a.ptr_index(n*AVX_256_N), __mask, __a);


			for(size_t j=0; j<r; j++) {
				if(_a.at_index(n*AVX_256_N+j) > th.at_index(n*AVX_256_N+j)) {
					for(size_t z1=0; z1<depth; z1++) {
						if (_model._t_obj != -1) {
							th.at(z1) -= _model._lr_th*(spike.time-_model._t_obj);
						}
						if(z1 != n*AVX_256_N+j) {
							th.at(z1) -= _model._lr_th/static_cast<float>(depth-1);
						}
						else {
							th.at(z1) += _model._lr_th;
						//	if (_model._t_obj != -1) {
						//		th.at(z1) -= _model._lr_th*(spike.time-_model._t_obj);
						//	}
						}
						th.at(z1) = std::max<float>(_model._min_th, th.at(z1));
					}


					for(size_t x=0; x<_model._filter_width; x++) {
						for(size_t y=0; y<_model._filter_height; y++) {
							for(size_t z=0; z<_model._input_depth; z++) {
								w.at(x, y, z, n*AVX_256_N+j) = _model._stdp->process(w.at(x, y, z, n*AVX_256_N+j), input_time.at(x, y, z), spike.time);
							}
						}
					}
					return;
				}
			}
		}
	}
}

void _priv::ConvolutionImpl::test(const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike) {
	size_t depth = _model.depth();
	Tensor<float>& w = _model._w;
	Tensor<float>& th = _model._th;

	size_t n = depth/AVX_256_N;
	size_t r = depth%AVX_256_N;

	__m256 __c1 = _mm256_setzero_ps();

	uint32_t mask = 0xFFFFFFFF;


	__m256i __mask = _generate_mask(r);

	for(size_t x=0; x<_model._current_width; x++) {
		for(size_t y=0; y<_model._current_height; y++) {
			for(size_t i=0; i<n; i++) {
				_mm256_storeu_ps(_a.ptr(x, y, i*AVX_256_N), __c1);
				_mm256_storeu_ps(_inh.ptr(x, y, i*AVX_256_N), __c1);
			}


			if(r > 0) {
				_mm256_maskstore_ps(_a.ptr(x, y, n*AVX_256_N), __mask, __c1);
				_mm256_maskstore_ps(_inh.ptr(x, y, n*AVX_256_N), __mask, __c1);
			}
		}
	}

	if(_model._wta_infer) {
		_wta.fill(false);
	}

	for(const Spike& spike : input_spike) {

		std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>> output_spikes;
		_model.forward(spike.x, spike.y, output_spikes);

		for(const auto& entry : output_spikes) {
			uint16_t x = std::get<0>(entry);
			uint16_t y = std::get<1>(entry);
			uint16_t w_x = std::get<2>(entry);
			uint16_t w_y = std::get<3>(entry);

			if(_model._wta_infer && _wta.at(x, y)) {
				continue;
			}

			for(size_t i=0; i<n; i++) {
				__m256 __w = _mm256_loadu_ps(w.ptr(w_x, w_y, spike.z, i*AVX_256_N));
				__m256 __a = _mm256_loadu_ps(_a.ptr(x, y, i*AVX_256_N));
				__m256 __th = _mm256_loadu_ps(th.ptr(i*AVX_256_N));
				__a = _mm256_add_ps(__a, __w);
				_mm256_storeu_ps(_a.ptr(x, y, i*AVX_256_N), __a);
				__m256 __c = _mm256_cmp_ps(__a, __th, _CMP_GE_OQ);

				__m256 __m = _mm256_loadu_ps(_inh.ptr(x, y, i*AVX_256_N));

				if(_mm256_testc_ps(__m, __c) == 0) {
					for(size_t j=0; j<AVX_256_N; j++) {
						if(_a.at(x, y, i*AVX_256_N+j) >= th.at_index(i*AVX_256_N+j) && *reinterpret_cast<uint32_t*>(_inh.ptr(x, y, i*AVX_256_N+j)) != mask) {
							_wta.at(x, y) = true;
							output_spike.emplace_back(spike.time, x, y, i*AVX_256_N+j);
						}
					}

					_mm256_storeu_ps(_inh.ptr(x, y, i*AVX_256_N), __c);


				}

			}

			if(r > 0) {
				__m256 __w = _mm256_maskload_ps(w.ptr(w_x, w_y, spike.z, n*AVX_256_N), __mask);
				__m256 __a = _mm256_maskload_ps(_a.ptr(x, y, n*AVX_256_N), __mask);
				__m256 __th = _mm256_maskload_ps(th.ptr(n*AVX_256_N), __mask);
				__a = _mm256_add_ps(__a, __w);
				_mm256_maskstore_ps(_a.ptr(x, y, n*AVX_256_N), __mask, __a);
				__m256 __c = _mm256_cmp_ps(__a, __th, _CMP_GE_OQ);


				for(size_t j=0; j<r; j++) {
					if(_a.at(x, y, n*AVX_256_N+j) >= th.at_index(n*AVX_256_N+j) && *reinterpret_cast<uint32_t*>(_inh.ptr(x, y, n*AVX_256_N+j)) != mask) {
						_wta.at(x, y) = true;
						output_spike.emplace_back(spike.time, x, y, n*AVX_256_N+j);
					}
				}

				_mm256_maskstore_ps(_inh.ptr(x, y, n*AVX_256_N), __mask, __c);

			}
		}
	}
}

#else
_priv::ConvolutionImpl::ConvolutionImpl(Convolution& model) : _model(model), _a(), _inh(), _wta() {

}

void _priv::ConvolutionImpl::resize() {
	_a = Tensor<float>(Shape({_model.width(), _model.height(), _model.depth()}));
	_inh = Tensor<bool>(Shape({_model.width(), _model.height(), _model.depth()}));
	_wta = Tensor<bool>(Shape({_model.width(), _model.height()}));
}

void _priv::ConvolutionImpl::train(const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike) {
	size_t depth = _model.depth();
	Tensor<float>& w = _model._w;
	Tensor<float>& th = _model._th;

	std::fill(std::begin(_a), std::end(_a), 0);

	for(const Spike& spike : input_spike) {

		for(size_t z=0; z<depth; z++) {

			_a.at(0, 0, z) += w.at(spike.x, spike.y, spike.z, z);

			if(_a.at(0, 0, z) >= th.at(z)) {
				for(size_t z1=0; z1<depth; z1++) {
					if (_model._t_obj != -1) {
						th.at(z1) -= _model._lr_th*(spike.time-_model._t_obj);
					}
					if(z1 != z) {
						th.at(z1) -= _model._lr_th/static_cast<float>(depth-1);
					}
					else {
						th.at(z1) += _model._lr_th;
					//	if (_model._t_obj != -1) {
					//		th.at(z1) -= _model._lr_th*(spike.time-_model._t_obj);
					//	}
					}
					th.at(z1) = std::max<float>(_model._min_th, th.at(z1));
				}

				for(size_t x=0; x<_model._filter_width; x++) {
					for(size_t y=0; y<_model._filter_height; y++) {
						for(size_t zi=0; zi<_model._input_depth; zi++) {
							w.at(x, y, zi, z) = _model._stdp->process(w.at(x, y, zi, z), input_time.at(x, y, zi), spike.time);
						}
					}
				}

				return;
			}
		}
	}
}

void _priv::ConvolutionImpl::test(const std::vector<Spike>& input_spike, const Tensor<Time>&, std::vector<Spike>& output_spike) {
	size_t depth = _model.depth();
	Tensor<float>& w = _model._w;
	Tensor<float>& th = _model._th;


	std::fill(std::begin(_a), std::end(_a), 0);
	std::fill(std::begin(_inh), std::end(_inh), false);

	if(_model._wta_infer) {
		_wta.fill(false);
	}

	for(const Spike& spike : input_spike) {

		// Get the spatial position of output neurons integrating inputs coming from the spatial position of the input spike
		// NOTE: output_spikes should be called pos_to_process or something like that
		std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>> output_spikes;
		_model.forward(spike.x, spike.y, output_spikes);

		// Iterate over output neuron spatial positions 
		for(const auto& entry : output_spikes) {
			uint16_t x = std::get<0>(entry);
			uint16_t y = std::get<1>(entry);
			uint16_t w_x = std::get<2>(entry);
			uint16_t w_y = std::get<3>(entry);

			// WTA inhibition : one spike per spatial position
			if(_model._wta_infer && _wta.at(x, y)) {
				continue;
			}

			// Iterate over channels (<=> output neuron at the given spatial position)
			for(size_t z=0; z<depth; z++) {
				
				// Single spike inhibition : one spike per neuron
				if(_inh.at(x, y, z)) {
					continue;
				}

				// WTA test should be here too, to prevent two neurons from different feature maps
				// but sharing the same spatial position to fire at the same time
				// BUT not in the AVX implementation because hard to do with multiprocessing 
				//if(_model._wta_infer && _wta.at(x, y)) {
				//	continue;
				//}

				// Update the membrane potential of the output neuron 
				// with the weight associated to the input neuron 
				_a.at(x, y, z) += w.at(w_x, w_y, spike.z, z);
				// When the membrane potential reaches the threshold of the channel
				if(_a.at(x, y, z) >= th.at(z)) {
					// Add a spike to the output vector
					output_spike.emplace_back(spike.time, x, y, z);
					// Deactivate the neuron
					_inh.at(x, y, z) = true;
					// Add WTA on the spatial position
					_wta.at(x, y) = true;
				}
			}
		}
	}
}

#endif
