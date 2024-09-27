#ifndef _PLOT_EVOLUTION_H
#define _PLOT_EVOLUTION_H

#ifdef ENABLE_QT

#include <QGridLayout>

#include <iostream>

#include "Plot.h"
#include "dep/qcustomplot/qcustomplot.h"
#include "Tensor.h"

namespace plot {

	class Evolution : public Plot {

	public:
		Evolution(const std::string& name, const Tensor<float>& t) : _layout(this), _plot(this), _t(t), _min_x(std::numeric_limits<float>::max()), _max_x(std::numeric_limits<float>::lowest()), _min_y(std::numeric_limits<float>::max()), _max_y(std::numeric_limits<float>::lowest()), _count(0) {
			setWindowTitle(QString((name+" evolution").c_str()));
			setLayout(&_layout);

			_layout.addWidget(&_plot, 0, 0);

			_plot.addGraph();
			_plot.graph(0)->setPen(QPen(Qt::blue));
			_plot.graph(0)->setName("Entropy");

			_plot.yAxis->grid()->setSubGridVisible(true);
			_plot.xAxis->grid()->setSubGridVisible(true);
			_plot.yAxis->setScaleType(QCPAxis::stLogarithmic);
			_plot.yAxis2->setScaleType(QCPAxis::stLogarithmic);
			QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
			_plot.yAxis->setTicker(logTicker);
			_plot.yAxis2->setTicker(logTicker);
			_plot.yAxis->setNumberFormat("eb");
			_plot.yAxis->setNumberPrecision(0);


			_plot.legend->setVisible(true);
			_plot.legend->setBrush(QBrush(QColor(255,255,255,150)));
			_plot.axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignRight|Qt::AlignTop);
		}


		virtual void initialize() {

		}

		virtual void on_tick() {
			_count++;
		}

		virtual void on_refresh() {

			double entropy = 0;

			size_t s = _t.shape().product();
			for(size_t i=0; i<s; i++) {
				entropy += (_t.at_index(i))*(1.0-_t.at_index(i));
			}

			entropy /= s;


			_plot.graph(0)->addData(_count, entropy);
			_min_y = std::min<float>(_min_y, entropy);
			_max_y = std::max<float>(_max_y, entropy);


			_plot.xAxis->setRange(_count, _count, Qt::AlignRight);
			_plot.yAxis->setRange(_max_y, _max_y-_min_y, Qt::AlignRight);

			_plot.replot();
		}

	private:
		QGridLayout _layout;
		QCustomPlot _plot;
		const Tensor<float>& _t;

		float _min_x;
		float _max_x;
		float _min_y;
		float _max_y;

		float _count;
	};

}

#endif

#endif
