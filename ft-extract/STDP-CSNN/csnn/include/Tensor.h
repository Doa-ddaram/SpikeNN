#ifndef _TENSOR_H
#define _TENSOR_H

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <numeric>

#include "Debug.h"

class Shape {

public:
	Shape() : _dims(), _product() {
		_product.push_back(0);
	}

	Shape(const std::vector<size_t>& dims) : _dims(dims), _product() {
		for(size_t i=0; i<_dims.size(); i++) {
			_product.push_back(std::accumulate(std::begin(_dims)+i, std::end(_dims), 1, std::multiplies<size_t>()));
		}
		_product.push_back(1);
	}

	Shape(const Shape& that) noexcept : _dims(that._dims), _product(that._product) {

	}

	Shape(Shape&& that) noexcept : _dims(std::move(that._dims)), _product(std::move(that._product)) {

	}

	~Shape() {

	}

	Shape& operator=(const Shape& that) noexcept {
		_dims = that._dims;
		_product = that._product;
		return *this;
	}

	Shape& operator=(Shape&& that) noexcept {
		_dims = std::move(that._dims);
		_product = std::move(that._product);
		return *this;
	}

	size_t number() const {
		return _dims.size();
	}

	size_t dim(size_t i) const {
		return _dims.at(i);
	}

	size_t product() const {
		return _product.front();
	}

	template<typename... Index>
	size_t to_index(Index&&... index) const {
		ASSERT_DEBUG(sizeof...(Index) == _dims.size());
		return _to_index<0, Index...>(std::forward<Index>(index)...);
	}

	bool operator==(const Shape& that) const {
		return _dims == that._dims;
	}

	bool operator!=(const Shape& that) const {
		return _dims != that._dims;
	}

	void print(std::ostream& stream) const {
		stream << "[";

		for(size_t i=0; i<_dims.size(); i++) {
			if(i != 0)
				stream << ", ";
			stream << _dims[i];
		}
		stream << "]";
	}

	std::string to_string() const {
		std::stringstream ss;
		print(ss);
		return ss.str();
	}

private:
	template<size_t Index, typename Head, typename... Tail>
	size_t _to_index(Head head, Tail&&... tail) const {
		ASSERT_DEBUG(static_cast<int64_t>(head) >= 0 && static_cast<int64_t>(head) < static_cast<int64_t>(_dims.at(Index)));
		return _product[Index+1]*head+_to_index<Index+1, Tail...>(std::forward<Tail>(tail)...);
	}

	template<size_t Index>
	size_t _to_index() const {
		return 0;
	}

	std::vector<size_t> _dims;
	std::vector<size_t> _product;

};

template<typename T>
class Tensor {

public:
    typedef T Type;

	Tensor() : _shape(), _data(nullptr) {

	}

	Tensor(const Shape& shape) : _shape(shape), _data(new T[_shape.product()]) {

	}

	Tensor(const Tensor& that) noexcept : _shape(that._shape), _data(new T[_shape.product()]) {
		std::copy(that._data, that._data+_shape.product(), _data);
	}

	Tensor(Tensor&& that) noexcept : _shape(std::move(that._shape)), _data(that._data) {
		that._data = nullptr;
	}

	~Tensor() {
		delete[] _data;
	}

	Tensor& operator=(const Tensor& that) noexcept {
		if(_shape.product() != that.shape().product()) {
			delete[] _data;
			_data = new T[that._shape.product()];
		}
		_shape = that._shape;
		std::copy(that._data, that._data+_shape.product(), _data);
		return *this;
	}

	Tensor& operator=(Tensor&& that) noexcept {
		delete[] _data;
		_shape = std::move(that._shape);
		_data = that._data;
		that._data = nullptr;
		return *this;
	}

	template<typename... Index>
	T& at(Index&&... index) {
		return _data[_shape.to_index(std::forward<Index>(index)...)];
	}

	template<typename... Index>
	T at(Index&&... index) const {
		return _data[_shape.to_index(std::forward<Index>(index)...)];
	}

	template<typename... Index>
	T* ptr(Index&&... index) {
		return _data+_shape.to_index(std::forward<Index>(index)...);
	}

	template<typename... Index>
	const T* ptr(Index&&... index) const {
		return _data+_shape.to_index(std::forward<Index>(index)...);
	}

	T& at_index(size_t index) {
		ASSERT_DEBUG(index < _shape.product());
		return _data[index];
	}

	T at_index(size_t index) const {
		ASSERT_DEBUG(index < _shape.product());
		return _data[index];
	}

	T* ptr_index(size_t index) {
		ASSERT_DEBUG(index < _shape.product());
		return _data+index;
	}

	const T* ptr_index(size_t index) const {
		ASSERT_DEBUG(index < _shape.product());
		return _data+index;
	}

	const Shape& shape() const {
		return _shape;
	}

	void reshape(const Shape& shape) {
		if(shape.product() != _shape.product()) {
			throw std::runtime_error("reshape: Shape must be of same length");
		}
		_shape = shape;
	}

	T* begin() {
		return _data;
	}

	const T* begin() const {
		return _data;
	}

	T* end() {
		return _data+_shape.product();
	}

	const T* end() const {
		return _data+_shape.product();
	}

	void fill(T value) {
		std::fill(begin(), end(), value);
	}

	void range_normalize(T min = 0.0, T max = 1.0) {
		auto it = std::minmax_element(begin(), end());
		T cmin = *it.first;
		T cmax = *it.second;
		size_t size = _shape.product();

		if(cmin == cmax) {
			for(size_t i=0; i<size; i++) {
				_data[i] = min;
			}
		}
		else {
			for(size_t i=0; i<size; i++) {
				_data[i] = ((_data[i]-cmin)/(cmax-cmin))*(max-min)+min;
			}
		}

	}

	std::pair<T, T> min_max_exclude(T v = std::numeric_limits<T>::max()) const {
		size_t size = _shape.product();
		T min = std::numeric_limits<T>::max();
		T max = std::numeric_limits<T>::lowest();

		for(size_t i=0; i<size; i++) {
			if(_data[i] != v) {
				min = std::min(min, _data[i]);
				max = std::max(max, _data[i]);
			}
		}

		return std::make_pair(min, max);
	}

	void range_normalize_exclude(T v = std::numeric_limits<T>::max(), T min = 0.0, T max = 1.0) {
		auto minmax = min_max_exclude(v);
		size_t size = _shape.product();
		for(size_t i=0; i<size; i++) {
			_data[i] = ((_data[i]-minmax.first)/(minmax.second-minmax.first))*(max-min)+min;
		}
	}

	void save(std::ostream& stream) const {
		uint8_t dim_number = _shape.number();
		stream.write(reinterpret_cast<const char*>(&dim_number), sizeof(uint8_t));
		for(size_t i = 0; i<dim_number; i++) {
			uint16_t dim = _shape.dim(i);
			stream.write(reinterpret_cast<const char*>(&dim), sizeof(uint16_t));
		}
		size_t size = _shape.product();
		size_t null_value = 0;
		for(uint32_t i=0; i<size; i++) {
			if(at_index(i) == 0.0)
				null_value++;
		}

		if(null_value > size/2-1) { // sparse
			uint8_t flag = 0;
			stream.write(reinterpret_cast<const char*>(&flag), sizeof(uint8_t));
			for(uint32_t i=0; i<size; i++) {
				if(at_index(i) != 0.0) {
					 stream.write(reinterpret_cast<const char*>(&i), sizeof(uint32_t));
					 float f = at_index(i);
					 stream.write(reinterpret_cast<const char*>(&f), sizeof(float));
				}
			}
			uint32_t eol = 0xFFFFFFFF;
			stream.write(reinterpret_cast<const char*>(&eol), sizeof(uint32_t));
		}
		else { // dense
			uint8_t flag = 1;
			stream.write(reinterpret_cast<const char*>(&flag), sizeof(uint8_t));
			stream.write(reinterpret_cast<const char*>(begin()), sizeof(float)*size);
		}

	}

	void load(std::istream& stream) {
		uint8_t dim_number;
		stream.read(reinterpret_cast<char*>(&dim_number), sizeof(uint8_t));
		std::vector<size_t> dims;
		dims.reserve(dim_number);
		for(size_t i = 0; i<dim_number; i++) {
			uint16_t dim;
			stream.read(reinterpret_cast<char*>(&dim), sizeof(uint16_t));
			dims.push_back(dim);
		}

		Shape new_shape(dims);

		if(new_shape.product() != _shape.product()) {
			delete[] _data;
			_data = new T[new_shape.product()];
		}

		_shape = std::move(new_shape);

		size_t size = _shape.product();

		uint8_t flag = 0;
		stream.read(reinterpret_cast<char*>(&flag), sizeof(uint8_t));

		if(flag == 0) { //sparse
			uint32_t index;
			stream.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
			float value;
			while(index != 0xFFFFFFFF) {
				stream.read(reinterpret_cast<char*>(&value), sizeof(float));
				at_index(index) = value;
				stream.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
			}
		}
		else if(flag == 1){
			stream.read(reinterpret_cast<char*>(begin()), sizeof(float)*size);
		}
		else {
			throw std::runtime_error("Tensor load: unkown flag");
		}
	}

private:
	Shape _shape;
	T* _data;

};


#endif
