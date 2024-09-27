#include "plot/Time.h"
#include "Experiment.h"

#ifdef ENABLE_QT
using namespace plot;

TimeHistogram::TimeHistogram(const std::string& name, const AbstractExperiment* experiment, size_t index, size_t n, float min, float max) :
	_layout(this), _plot(this),
	_experiment(experiment), _index(index),
	_serie(new QCPBars(_plot.xAxis, _plot.yAxis)), _data(),_ticks(), _max_y(0), _min_x(min), _max_x(max),_n(n) {

	setWindowTitle(QString((name+" time").c_str()));
	setLayout(&_layout);

	_layout.addWidget(&_plot, 0, 0);

	for(size_t i=0; i<n; i++) {
		_ticks.push_back(_min_x+(_max_x-_min_x)*(static_cast<float>(i)/static_cast<float>(n)));
		_data.push_back(0);
	}

	_plot.xAxis->setRange(_min_x, _max_x);
	_serie->setWidth((_max_x-_min_x)/static_cast<float>(n));
	_serie->setPen(Qt::NoPen);
	_serie->setBrush(QColor("Blue"));
}

void TimeHistogram::on_tick() {
	Tensor<Time> t = _experiment->compute_time_at(_index);
	for(float v : t) {
		if(v >= _min_x && v < _max_x)
			_data[static_cast<size_t>((v-_min_x)/(_max_x-_min_x)*static_cast<float>(_n))] += 1.0;
	}
}

void TimeHistogram::on_refresh() {
	for(size_t i=0; i<_n; i++) {
		double value = _data[i];

		if(std::isinf(value) || std::isnan(value)) {
			value = 0;
			std::cerr << "Warning: Plot, Inf or NaN value" << std::endl;
		}


		_max_y = std::max(_max_y, value);
	}

	_plot.yAxis->setRange(0, _max_y);



	_serie->setData(_ticks, _data);

	for(size_t i=0; i<_n; i++) {
		_data[i] = 0;
	}
	_max_y = 0;

	_plot.replot();
}
#endif
