#ifndef _M_MATH_H
#define _M_MATH_H

#include <iostream>
#include <cmath>
#include <cblas.h>

#include "Tensor.h"

namespace _priv {

	class MathHelper {

	public:
		MathHelper() = delete;

		static Tensor<float> unary_arithmetic_operator(const Tensor<float>& in);
		static Tensor<float> binary_arithmetic_operator(const Tensor<float>& lhs, const Tensor<float>& rhs);

	};
}


Tensor<float> add(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> operator+(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> sub(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> operator-(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> mul(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> operator*(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> div(const Tensor<float>& lhs, const Tensor<float>& rhs);
Tensor<float> operator/(const Tensor<float>& lhs, const Tensor<float>& rhs);

Tensor<float> sqrt(const Tensor<float>& in);
Tensor<float> diag(const Tensor<float>& in);

Tensor<float> dot(const Tensor<float>& lhs, const Tensor<float>& rhs, size_t n = 1);

Tensor<float> sep_sign(const Tensor<float>& in);

Tensor<float> reshape(const Tensor<float>& in, const Shape& new_shape);
Tensor<float> flatten(const Tensor<float>& in);

Tensor<float> create(float value, const Shape& shape);
float mean(const Tensor<float>& in);
Tensor<float> mean1(const Tensor<float>& in);

Tensor<float> expand_in(const Tensor<float>& in, const Shape& new_dims);

Tensor<float> transpose(const Tensor<float>& in);
Tensor<float> scale(const Tensor<float>& in, float min = 0.0, float max = 1.0);

void debug_tensor(const Tensor<float>& in, const std::string& name);

#endif
