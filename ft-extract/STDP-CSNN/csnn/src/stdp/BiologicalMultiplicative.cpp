#include "stdp/BiologicalMultiplicative.h"

using namespace stdp;

static RegisterClassParameter<BiologicalMultiplicative, STDPFactory> _register("BiologicalMultiplicative");

BiologicalMultiplicative::BiologicalMultiplicative() : STDP(_register), _alpha(0), _beta(0), _tau(0) {
	add_parameter("alpha", _alpha);
	add_parameter("beta", _beta);
	add_parameter("tau", _tau);
}


BiologicalMultiplicative::BiologicalMultiplicative(float alpha, float beta, Time tau) : BiologicalMultiplicative() {
	parameter<float>("alpha").set(alpha);
	parameter<float>("beta").set(beta);
	parameter<float>("tau").set(tau);
}

float BiologicalMultiplicative::process(float w, const Time pre, Time post) {
	float v = pre <= post ? w+_alpha*std::exp(-(post-pre)/_tau)*std::exp(-_beta*w)  :  w-_alpha*std::exp(-(pre-post)/_tau)*std::exp(_beta*(w-1.0f));
	return std::max<float>(0, std::min<float>(1, v));
}

void BiologicalMultiplicative::adapt_parameters(float factor) {
	_alpha *= factor;
}
