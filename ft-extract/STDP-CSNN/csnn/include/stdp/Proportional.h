#ifndef _STDP_PROPORTIONAL_H
#define _STDP_PROPORTIONAL_H

#include "Stdp.h"

namespace stdp {

	class Proportional : public STDP {

	public:
		Proportional();
		Proportional(float alpha);

		virtual float process(float w, const Time pre, Time post);
		virtual void adapt_parameters(float factor);
	private:
		float _alpha;
	};

}
#endif
