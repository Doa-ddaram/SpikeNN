#ifndef _STDP_H
#define _STDP_H

#include "ClassParameter.h"
#include "Spike.h"



class STDP : public ClassParameter {

public:
	template<typename T, typename Factory>
	STDP(const RegisterClassParameter<T, Factory>& registration) : ClassParameter(registration) {

	}

    virtual float process(float w, const Time pre, Time post) = 0;

	virtual void adapt_parameters(float factor) = 0;

};

class STDPFactory : public ClassParameterFactory<STDP, STDPFactory> {


public:
	STDPFactory() : ClassParameterFactory<STDP, STDPFactory>("STDP") {

	}

};

#endif
