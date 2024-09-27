#include "stdp/Biological.h"

using namespace stdp;

static RegisterClassParameter<Biological, STDPFactory> _register("Biological");

Biological::Biological() : STDP(_register), _ap(0), _am(0), _tau(0) {
	add_parameter("ap", _ap);
	add_parameter("am", _am);
	add_parameter("tau", _tau);
}


Biological::Biological(float ap, float am, float tau) : Biological() {
	parameter<float>("ap").set(ap);
	parameter<float>("am").set(am);
	parameter<float>("tau").set(tau);
}

float Biological::process(float w, const Time pre, Time post) {
	float v = pre <= post ? w+_ap*std::exp(-(post-pre)/_tau) :  w-_am*std::exp(-(pre-post)/_tau);
	return std::max<float>(0, std::min<float>(1, v));
}

void Biological::adapt_parameters(float factor) {
	_ap *= factor;
	_am *= factor;
}
