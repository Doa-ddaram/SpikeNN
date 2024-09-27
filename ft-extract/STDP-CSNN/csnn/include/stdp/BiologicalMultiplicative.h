#ifndef _STDP_BIOLOGICAL_MULTIPLICATIVE_H
#define _STDP_BIOLOGICAL_MULTIPLICATIVE_H

#include "Stdp.h"

namespace stdp {

	class BiologicalMultiplicative : public STDP {

	public:
		BiologicalMultiplicative();
		BiologicalMultiplicative(float alpha, float beta, Time tau);

		virtual float process(float w, const Time pre, Time post);
		virtual void adapt_parameters(float factor);
	private:
		float _alpha;
		float _beta;
		float _tau;
	};

}
#endif
