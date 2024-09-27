#ifndef _SPARSE_TENSOR_H
#define _SPARSE_TENSOR_H

#include <map>
#include "Tensor.h"

template<typename T>
class SparseTensor {

public:
	SparseTensor(const Shape& shape, T default_value = 0.0) : _shape(shape), _values(), _default_value(default_value) {

	}

	void reset(T default_value = 0.0) {
		_default_value = default_value;
		_values.clear();
	}

	const Shape& shape() const {
		return _shape;
	}

	void clear() {
		_values.clear();
		_values.shrink_to_fit();
	}

	void add_index(uint32_t index, T value) {
		_values.emplace_back(index, value);
	}

	void optimize_space() {
		_values.shrink_to_fit();
	}

	const std::vector<std::pair<uint32_t, T>>& values() const {
		return _values;
	}

	T default_value() const {
		return _default_value;
	}

private:
	Shape _shape;
	std::vector<std::pair<uint32_t, T>> _values;
	T _default_value;

};

template<typename T>
void to_sparse_tensor(const Tensor<T>& from, SparseTensor<T>& to) {
	if(from.shape() != to.shape()) {
		throw std::runtime_error("Incomaptible shape");
	}

	to.clear();

	size_t size = from.shape().product();

	size_t zero_counter = 0;
	size_t max_counter = 0;
	for(size_t i=0; i<size; i++) {
		T value = from.at_index(i);
		if(value == 0) {
			zero_counter++;
		}
		else if(value == std::numeric_limits<T>::max()) {
			max_counter++;
		}
	}

	T default_value = zero_counter >= max_counter ? 0 : std::numeric_limits<T>::max();

	to.reset(default_value);

	for(size_t i=0; i<size; i++) {
		if(from.at_index(i) != default_value) {
			to.add_index(i, from.at_index(i));
		}
	}
	to.optimize_space();
}

template<typename T>
SparseTensor<T> to_sparse_tensor(const Tensor<T>& from) {
	SparseTensor<T> to(from.shape());
	to_sparse_tensor(from, to);
	return to;
}

template<typename T>
void from_sparse_tensor(const SparseTensor<T>& from, Tensor<T>& to) {
	if(from.shape() != to.shape()) {
		throw std::runtime_error("Incomaptible shape");
	}

	to.fill(from.default_value());

	for(const std::pair<uint32_t, T>& value : from.values()) {
		to.at_index(value.first) = value.second;
	}
}

template<typename T>
Tensor<T> from_sparse_tensor(const SparseTensor<T>& from) {
	Tensor<T> to(from.shape());
	from_sparse_tensor(from, to);
	return to;
}

#endif
