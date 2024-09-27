#include "stdp/Proportional.h"

using namespace stdp;

static RegisterClassParameter<Proportional, STDPFactory> _register("Proportional");

Proportional::Proportional() : STDP(_register), _alpha(0) {
	add_parameter("alpha", _alpha);
}


Proportional::Proportional(float alpha) : Proportional() {
	parameter<float>("alpha").set(alpha);
}

float Proportional::process(float w, const Time pre, Time post) {
	return std::max<float>(0, std::min<float>(1, w+_alpha*(post-pre)));
}

void Proportional::adapt_parameters(float factor) {
	_alpha *= factor;
}
