#include "stdp/Linear.h"

using namespace stdp;

static RegisterClassParameter<Linear, STDPFactory> _register("Linear");

Linear::Linear() : STDP(_register), _alpha_p(0), _alpha_m(0) {
	add_parameter("alpha_p", _alpha_p);
	add_parameter("alpha_m", _alpha_m);
}


Linear::Linear(float alpha_p, float alpha_m) : Linear() {
	parameter<float>("alpha_p").set(alpha_p);
	parameter<float>("alpha_m").set(alpha_m);

}

float Linear::process(float w, const Time pre, Time post) {
	float v = pre <= post ? w+_alpha_p :  w-_alpha_m;
	return std::max<float>(0, std::min<float>(1, v));
}

void Linear::adapt_parameters(float factor) {
	_alpha_p *= factor;
	_alpha_m *= factor;
}
