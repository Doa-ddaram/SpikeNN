#ifndef _ANALYSIS_COHERENCE_H
#define _ANALYSIS_COHERENCE_H

#include "Analysis.h"

namespace analysis {
	class Coherence : public NoPassAnalysis {

	public:
		Coherence();

		virtual void resize(const Shape& shape);

		virtual void process();
	};

}

#endif
