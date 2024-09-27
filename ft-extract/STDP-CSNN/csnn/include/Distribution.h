#ifndef _DISTRIBUTION_H
#define _DISTRIBUTION_H

#include <random>

template<typename T>
class Distribution {

public:
	virtual ~Distribution() {

	}

	virtual T generate(std::default_random_engine& random_generator) const = 0;
	virtual std::string to_string() const = 0;
};

namespace distribution {

	template<typename T>
	class Constant : public Distribution<T> {

	public:
		Constant(T value) : _value(value) {

		}

		virtual T generate(std::default_random_engine&) const {
			return _value;
		}

		virtual std::string to_string() const {
			return std::to_string(_value);
		}

	private:
		T _value;
	};

	template<typename T>
	class Uniform : public Distribution<T> {

	public:
		Uniform(float min = 0.0, float max = 1.0) : _min(min), _max(max) {

		}

		virtual T generate(std::default_random_engine& random_generator) const {
			std::uniform_real_distribution<T> distribution(_min, _max);
			return distribution(random_generator);
		}

		virtual std::string to_string() const {
			return "Uniform(min: "+std::to_string(_min)+", max: "+std::to_string(_max)+")";
		}

	private:
		float _min;
		float _max;
	};

	template<typename T>
	class Gaussian : public Distribution<T> {

	public:
		Gaussian(float mean = 0.0, float dev = 1.0) : _mean(mean), _dev(dev) {

		}

		virtual T generate(std::default_random_engine& random_generator) const {
			std::normal_distribution<float> distribution(_mean, _dev);
			return distribution(random_generator);
		}

		virtual std::string to_string() const {
			return "Gaussian(mean: "+std::to_string(_mean)+", dev: "+std::to_string(_dev)+")";
		}


	private:
		float _mean;
		float _dev;
	};

}

#endif
