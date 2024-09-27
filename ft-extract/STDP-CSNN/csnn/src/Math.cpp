#include "Math.h"

Tensor<float> _priv::MathHelper::unary_arithmetic_operator(const Tensor<float>& in) {
	return Tensor<float>(in.shape());
}

Tensor<float> _priv::MathHelper::binary_arithmetic_operator(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	if(lhs.shape() != rhs.shape()) {
		throw std::runtime_error("Incompatible shape ("+lhs.shape().to_string()+" "+rhs.shape().to_string()+")");
	}
	return Tensor<float>(lhs.shape());
}

Tensor<float> add(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	Tensor<float> out = _priv::MathHelper::binary_arithmetic_operator(lhs, rhs);

	for(size_t i=0; i<out.shape().product(); i++) {
		out.at_index(i) = lhs.at_index(i)+rhs.at_index(i);
	}

	return out;
}

Tensor<float> operator+(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	return add(lhs, rhs);
}

Tensor<float> sub(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	Tensor<float> out = _priv::MathHelper::binary_arithmetic_operator(lhs, rhs);

	for(size_t i=0; i<out.shape().product(); i++) {
		out.at_index(i) = lhs.at_index(i)-rhs.at_index(i);
	}

	return out;
}

Tensor<float> operator-(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	return sub(lhs, rhs);
}

Tensor<float> mul(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	Tensor<float> out = _priv::MathHelper::binary_arithmetic_operator(lhs, rhs);

	for(size_t i=0; i<out.shape().product(); i++) {
		out.at_index(i) = lhs.at_index(i)*rhs.at_index(i);
	}

	return out;
}

Tensor<float> operator*(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	return mul(lhs, rhs);
}

Tensor<float> div(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	Tensor<float> out = _priv::MathHelper::binary_arithmetic_operator(lhs, rhs);

	for(size_t i=0; i<out.shape().product(); i++) {
		out.at_index(i) = lhs.at_index(i)/rhs.at_index(i);
	}

	return out;
}

Tensor<float> operator/(const Tensor<float>& lhs, const Tensor<float>& rhs) {
	return div(lhs, rhs);
}

Tensor<float> sqrt(const Tensor<float>& in) {
	Tensor<float> out = _priv::MathHelper::unary_arithmetic_operator(in);

	for(size_t i=0; i<out.shape().product(); i++) {
		out.at_index(i) = std::sqrt(in.at_index(i));
	}

	return out;
}

Tensor<float> diag(const Tensor<float>& in) {
	if(in.shape().number() != 1) {
		throw std::runtime_error("Expected 1D tensor");
	}

	size_t n = in.shape().dim(0);

	Tensor<float> out(Shape({n, n}));
	out.fill(0);

	for(size_t i=0; i<n; i++) {
		out.at(i, i) = in.at(i);
	}

	return out;
}

Tensor<float> dot(const Tensor<float>& lhs, const Tensor<float>& rhs, size_t n) {
	std::vector<size_t> out_dims;


	if(n > lhs.shape().number()) {
		throw std::runtime_error("Expect lhs > n");
	}

	size_t l_size = 1;
	for(size_t i=0; i<lhs.shape().number()-n; i++) {
		l_size *= lhs.shape().dim(i);
		out_dims.push_back(lhs.shape().dim(i));
	}

	if(n > rhs.shape().number()) {
		throw std::runtime_error("Expect rhs > n");
	}


	size_t r_size = 1;
	for(size_t i=n; i<rhs.shape().number(); i++) {
		r_size *= rhs.shape().dim(i);
		out_dims.push_back(rhs.shape().dim(i));
	}

	size_t in_size = 1;
	for(size_t i=0; i<n; i++) {
		if(lhs.shape().dim(lhs.shape().number()-n+i) != rhs.shape().dim(i)) {
			throw std::runtime_error("Incompatible shape "+lhs.shape().to_string()+" "+rhs.shape().to_string());
		}

		in_size *= rhs.shape().dim(i);
	}

	Shape out_shape(out_dims);
	Tensor<float> out(out_shape);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l_size, r_size, in_size, 1.0, lhs.begin(), in_size, rhs.begin(), r_size, 0.0, out.begin(), r_size);

	return out;
}

Tensor<float> sep_sign(const Tensor<float>& in) {

	std::vector<size_t> dims;
	for(size_t i=0; i<in.shape().number(); i++) {
		dims.push_back(in.shape().dim(i));
	}
	if(dims.empty()) {
		throw std::runtime_error("Expected array");
	}
	dims.back() *= 2;

	Shape out_shape(dims);
	Tensor<float> out(out_shape);

	for(size_t i=0; i<in.shape().product(); i++) {
		out.at_index(i*2) = std::max<float>(0, in.at_index(i));
		out.at_index(i*2+1) = std::max<float>(0, -in.at_index(i));
	}
	return out;
}

Tensor<float> reshape(const Tensor<float>& in, const Shape& new_shape) {
	if(new_shape.product() != in.shape().product()) {
		throw std::runtime_error("Incompatible shape");
	}

	Tensor<float> out(new_shape);

	for(size_t i=0; i<new_shape.product(); i++) {
		out.at_index(i) = in.at_index(i);
	}

	return out;
}

Tensor<float> flatten(const Tensor<float>& in) {
	return reshape(in, Shape({in.shape().product()}));
}

Tensor<float> create(float value, const Shape& shape) {
	Tensor<float> out(shape);
	out.fill(value);
	return out;
}

float mean(const Tensor<float>& in) {
	float value = 0;
	for(size_t i=0; i<in.shape().product(); i++) {
		value += in.at_index(i);
	}
	return value/static_cast<float>(in.shape().product());
}

Tensor<float> mean1(const Tensor<float>& in) {
	size_t size = in.shape().dim(0);
	size_t size1 = in.shape().product()/size;
	Tensor<float> out(Shape({size}));

	for(size_t i=0; i<size; i++) {
		float v = 0;
		for(size_t j=0; j<size1; j++) {
			v += in.at_index(i*size1+j);
		}
		out.at(i) = v/size1;
	}

	return out;
}


Tensor<float> expand_in(const Tensor<float>& in, const Shape& new_dims) {
	std::vector<size_t> out_dims;

	for(size_t i=0; i<in.shape().number(); i++) {
		out_dims.push_back(in.shape().dim(i));
	}
	for(size_t i=0; i<new_dims.number(); i++) {
		out_dims.push_back(new_dims.dim(i));
	}

	Shape shape(out_dims);
	Tensor<float> out(shape);

	size_t size1 = in.shape().product();
	size_t size2 = new_dims.product();

	for(size_t i=0; i<size1; i++) {
		for(size_t j=0; j<size2; j++) {
			out.at_index(i*size2+j) = in.at_index(i);
		}
	}
	return out;
}

Tensor<float> transpose(const Tensor<float>& in) {
	if(in.shape().number() != 2) {
		throw std::runtime_error("Unsupported shape");
	}
	size_t n = in.shape().dim(0);
	size_t m = in.shape().dim(1);
	Tensor<float> out(Shape({m, n}));
	for(size_t i=0; i<n; i++) {
		for(size_t j=0; j<m; j++) {
			out.at(j, i) = in.at(i, j);
		}
	}
	return out;
}

Tensor<float> scale(const Tensor<float>& in, float min, float max) {
	Tensor<float> out(in.shape());
	auto limits = std::minmax_element(in.begin(), in.end());
	float min_value = *limits.first;
	float max_value = *limits.second;
	for(size_t i=0; i<in.shape().product(); i++) {
		float norm_value = (in.at_index(i)-min_value)/(max_value-min_value);
		out.at_index(i) = norm_value*(max-min)+min;
	}
	return out;
}

void debug_tensor(const Tensor<float>& in, const std::string& name) {
	std::cout << name << ": " << in.shape().to_string() << std::endl;
	auto limits = std::minmax_element(in.begin(), in.end());
	float sum = 0;
	for(size_t i=0; i<in.shape().product(); i++) {
		sum += in.at_index(i);
	}

	std::cout << "\tRange: [" << *limits.first << ", " << *limits.second << "] at [" <<
				 std::distance(in.begin(), limits.first) << ";" << std::distance(in.begin(), limits.second) << "]" << std::endl;
	std::cout << "\tMean: " << (sum/in.shape().product()) << std::endl;
	std::cout << "\tValues: ";
	for(size_t i=0; i<std::min<size_t>(in.shape().product(), 10); i++) {
		std::cout << in.at_index(i) << " ";
	}
	if(in.shape().product() > 10) {
		std::cout << "...";
	}
	std::cout << std::endl;
}
