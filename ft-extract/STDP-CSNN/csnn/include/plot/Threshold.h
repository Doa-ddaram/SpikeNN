#ifndef _PLOT_THRESHOLD_H
#define _PLOT_THRESHOLD_H

#ifdef ENABLE_QT

#include <QGridLayout>

#include <iostream>

#include "Plot.h"
#include "dep/qcustomplot/qcustomplot.h"
#include "Tensor.h"

namespace plot {

	class Threshold : public Plot {

	public:
		Threshold(const std::string& name, const Tensor<float>& th) : _layout(this), _plot(this), _th(th), _size(0), _min_x(std::numeric_limits<float>::max()), _max_x(std::numeric_limits<float>::lowest()), _min_y(std::numeric_limits<float>::max()), _max_y(std::numeric_limits<float>::lowest()), _count(0) {
			setWindowTitle(QString((name+" thresholds").c_str()));
			setLayout(&_layout);

			_layout.addWidget(&_plot, 0, 0);
		}

		~Threshold() {
			_layout.takeAt(0);
		}


		virtual void initialize() {
			_size = _th.shape().product();
			for(size_t i=0; i<_size; i++) {
				_plot.addGraph();
			}
		}

		virtual void on_tick() {
			_count++;
		}

		virtual void on_refresh() {

			for(size_t i=0; i<_size; i++) {
				float value = _th.at(i);

				if(std::isinf(value) || std::isnan(value)) {
					value = 0;
					std::cerr << "Warning: PropertiesEvolutionViewer, Inf or NaN value" << std::endl;
				}

				_plot.graph(i)->addData(_count, value);

				_min_y = std::min(_min_y, value);
				_max_y = std::max(_max_y, value);
			}

			_plot.xAxis->setRange(_count, _count, Qt::AlignRight);
			_plot.yAxis->setRange(_max_y, _max_y-_min_y, Qt::AlignRight);

			_plot.replot();
		}

	private:
		QGridLayout _layout;
		QCustomPlot _plot;
		const Tensor<float>& _th;

		size_t _size;

		float _min_x;
		float _max_x;
		float _min_y;
		float _max_y;

		float _count;
	};

}

#endif

#endif
