#ifndef _PLOT_TIME_HISTOGRAM_H
#define _PLOT_TIME_HISTOGRAM_H

#ifdef ENABLE_QT

#include <iostream>
#include <QGridLayout>

#include "dep/qcustomplot/qcustomplot.h"
#include "Tensor.h"
#include "Plot.h"

class AbstractExperiment;

namespace plot {

	class TimeHistogram : public Plot {

	public:
		TimeHistogram(const std::string& name, const AbstractExperiment* experiment, size_t index, size_t n = 20, float min = 0.0, float max = 1.0);

		TimeHistogram(const TimeHistogram& that) = delete;
		TimeHistogram& operator=(const TimeHistogram& that) = delete;

		virtual void on_tick();
		virtual void on_refresh();

	private:
		QGridLayout _layout;
		QCustomPlot _plot;

		const AbstractExperiment* _experiment;
		size_t _index;

		QCPBars* _serie;
		QVector<double> _data;
		QVector<double> _ticks;

		double _max_y;
		float _min_x;
		float _max_x;

		size_t _n;
	};

}
#endif

#endif
