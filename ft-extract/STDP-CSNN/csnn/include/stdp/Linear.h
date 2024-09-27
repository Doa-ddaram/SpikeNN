#ifndef _STDP_LINEAR_H
#define _STDP_LINEAR_H

#include "Stdp.h"

namespace stdp {

	class Linear : public STDP {

	public:
		Linear();
		Linear(float alpha_p, float alpha_m);

		virtual float process(float w, const Time pre, Time post);
		virtual void adapt_parameters(float factor);
	private:
		float _alpha_p;
		float _alpha_m;
	};

}
#endif
