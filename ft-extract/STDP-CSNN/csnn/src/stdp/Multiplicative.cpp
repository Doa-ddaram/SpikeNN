#include "stdp/Multiplicative.h"

using namespace stdp;

static RegisterClassParameter<Multiplicative, STDPFactory> _register("Multiplicative");

Multiplicative::Multiplicative() : STDP(_register), _ap(0), _am(0), _beta(0) {
	add_parameter("ap", _ap);
	add_parameter("am", _am);
    add_parameter("beta", _beta);
}


Multiplicative::Multiplicative(float ap, float am, float beta) : Multiplicative() {
	parameter<float>("ap").set(ap);
	parameter<float>("am").set(am);
    parameter<float>("beta").set(beta);
}

float Multiplicative::process(float w, const Time pre, Time post) {
	float v = pre <= post ? w+_ap*std::exp(-_beta*w) :  w-_am*std::exp(_beta*(w-1.0f));
	return std::max<float>(0, std::min<float>(1, v));
}

void Multiplicative::adapt_parameters(float factor) {
	_ap *= factor;
	_am *= factor;
}
