#ifndef _PLOT_DELAY_H
#define _PLOT_DELAY_H

#include <QWidget>
#include <QGridLayout>
#include <QPainter>

#include <map>
#include <iostream>

#include "Tensor.h"
#include "Color.h"
#include "Plot.h"

class Layer;

namespace plot {

	template<typename ColorType>
	class DelayPanel : public QWidget {

	public:
		DelayPanel() : QWidget(), _width(0), _height(0), _data(nullptr) {

		}

		DelayPanel(const DelayPanel& that) noexcept : _width(that._width), _height(that._height), _data(_width*_height > 0 ? new QColor[_width*_height] : nullptr) {
			if(_width*_height > 0 )
				std::copy(that._data, that._data+_width*_height, _data);
		}

		DelayPanel(DelayPanel&& that) noexcept : _width(that._width), _height(that._height), _data(that._data)  {
			that._data = nullptr;
		}

		~DelayPanel() {
			delete[] _data;
		}

		DelayPanel& operator=(const DelayPanel& that) noexcept {
			delete[] _data;
			_data = _width*_height > 0 ? new QColor[_width*_height] : nullptr;
			if(_width*_height > 0 )
				std::copy(that._data, that._data+_width*_height, _data);
			return *this;
		}

		DelayPanel& operator=(DelayPanel&& that) noexcept {
			delete[] _data;
			_data = that._data;
			that._data = nullptr;
			return *this;
		}

		void update(const Tensor<float>& t, size_t i) {

			size_t input_width = t.shape().dim(0);
			size_t input_height = t.shape().dim(1);
			size_t input_depth = t.shape().dim(2);

			if(input_width != _width || input_height != _height) {
				_width = input_width;
				_height = input_height;
				delete[] _data;
				_data = new QColor[_width*_height];
			}

			for(size_t x=0; x<input_width; x++) {
				for(size_t y=0; y<input_height; y++) {

					std::vector<float> zs;
					for(size_t z=0; z<input_depth; z++) {
						zs.push_back(1.0-t.at(x, y, z, i));
					}
					_data[x*_height+y] = ColorType::get(zs);
				}
			}
		}

		void to_image(const Tensor<float>& t, const std::string& name) const {
			size_t input_width = t.shape().dim(0);
			size_t input_height = t.shape().dim(1);
			size_t input_depth = t.shape().dim(2);

			QPixmap pixmap(input_width, input_height);
			QPainter painter(&pixmap);

			for(size_t x=0; x<input_width; x++) {
				for(size_t y=0; y<input_height; y++) {

					std::vector<float> zs;
					for(size_t z=0; z<input_depth; z++) {
						zs.push_back(t.at(x, y, z));
					}
					painter.fillRect(x, y, 1, 1, ColorType::get(zs));
				}
			}

			pixmap.save(QString(name.c_str()));


		}



	protected:
		void paintEvent(QPaintEvent*) {
			size_t x_size = size().width()/_width;
			size_t y_size = size().height()/_height;

			size_t min = std::min(x_size, y_size);
			x_size = min;
			y_size = min;


			QPainter painter(this);
			painter.setRenderHint(QPainter::Antialiasing);

			for(size_t x=0; x<_width; x++) {
				for(size_t y=0; y<_height; y++) {
					painter.fillRect(x*x_size, y*y_size, x_size, y_size, _data[x*_height+y]);
				}
			}
		}

	private:
		size_t _width;
		size_t _height;
		QColor* _data;
	};

	template<typename ColorType = DefaultColor>
	class Delay : public Plot {

	public:
		Delay(const std::string& name, const Tensor<float>& d, size_t max_filter) : _layout(this), _panels(), _d(d), _max_filter(max_filter) {
			setWindowTitle(QString((name+" delays").c_str()));
			setLayout(&_layout);
		}

		~Delay() {
			for(auto& entry : _panels) {
				delete entry;
			}
		}

		Delay(const Delay& that) = delete;

		Delay& operator=(const Delay& that) = delete;

		virtual void initialize() {
			size_t width = std::ceil(std::sqrt(std::min(_max_filter, _d.shape().dim(3))));

			_panels.reserve(std::min(_max_filter, _d.shape().dim(3)));

			for(size_t i=0; i<std::min(_max_filter, _d.shape().dim(3)); i++) {
				_panels.push_back(new DelayPanel<ColorType>);
				DelayPanel<ColorType>* panel = _panels.back();
				panel->setParent(this);
				_layout.addWidget(panel, i/width, i%width);
			}

			show();
		}

		virtual void on_tick() {

		}

		virtual void on_refresh() {
			for(size_t i=0; i<std::min(_max_filter, _d.shape().dim(3)); i++) {
				_panels[i]->update(_d, i);

			}

			repaint();

		}

	private:
		QGridLayout _layout;
		std::vector<DelayPanel<ColorType>*> _panels;
		const Tensor<float>& _d;
		size_t _max_filter;
	};

}

#endif
