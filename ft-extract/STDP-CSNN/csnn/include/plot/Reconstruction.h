#ifndef _PLOT_RECONSTRUCTION_H
#define _PLOT_RECONSTRUCTION_H

#ifdef ENABLE_QT

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

	namespace priv {

		struct ReconstructionHelper {

			ReconstructionHelper() = delete;

			static Tensor<float> process(const std::vector<const Layer *>& layers, size_t i);
		};
	}

	template<typename ColorType>
	class ReconstructionPanel : public QWidget {

	public:
		ReconstructionPanel() : QWidget(), _width(0), _height(0), _data(nullptr) {

		}

		ReconstructionPanel(const ReconstructionPanel& that) noexcept : _width(that._width), _height(that._height), _data(_width*_height > 0 ? new QColor[_width*_height] : nullptr) {
			if(_width*_height > 0 )
				std::copy(that._data, that._data+_width*_height, _data);
		}

		ReconstructionPanel(ReconstructionPanel&& that) noexcept : _width(that._width), _height(that._height), _data(that._data)  {
			that._data = nullptr;
		}

		~ReconstructionPanel() {
			delete[] _data;
		}

		ReconstructionPanel& operator=(const ReconstructionPanel& that) noexcept {
			delete[] _data;
			_data = _width*_height > 0 ? new QColor[_width*_height] : nullptr;
			if(_width*_height > 0 )
				std::copy(that._data, that._data+_width*_height, _data);
			return *this;
		}

		ReconstructionPanel& operator=(ReconstructionPanel&& that) noexcept {
			delete[] _data;
			_data = that._data;
			that._data = nullptr;
			return *this;
		}

		void update(const Tensor<float>& t) {

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
						zs.push_back(t.at(x, y, z));
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
			if(_width == 0 || _height == 0 || _data == nullptr)
				return;

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
	class Reconstruction : public Plot {

	public:
		Reconstruction(const std::string& name, const std::vector<const Layer *>& layers, size_t& filter_number, size_t max_filter) : _layout(this), _panels(), _filter_number(filter_number), _layers(layers), _max_filter(max_filter) {
			setWindowTitle(QString((name+" reconstruction").c_str()));
			setLayout(&_layout);
		}

		~Reconstruction() {
			for(auto& entry : _panels) {
				delete entry;
			}
		}

		Reconstruction(const Reconstruction& that) = delete;

		Reconstruction& operator=(const Reconstruction& that) = delete;

		virtual void initialize() {
			size_t width = std::ceil(std::sqrt(std::min(_max_filter, _filter_number)));

			_panels.reserve(std::min(_max_filter, _filter_number));

			for(size_t i=0; i<std::min(_max_filter, _filter_number); i++) {
				_panels.push_back(new ReconstructionPanel<ColorType>);
				ReconstructionPanel<ColorType>* panel = _panels.back();
				panel->setParent(this);
				_layout.addWidget(panel, i/width, i%width);
			}

			show();
		}

		virtual void on_tick() {

		}

		virtual void on_refresh() {
			for(size_t i=0; i<std::min(_max_filter, _filter_number); i++) {
				_panels[i]->update(priv::ReconstructionHelper::process(_layers, i));

			}

			repaint();
		}

		void to_image(const std::string& prefix) const {
			for(size_t i=0; i<std::min(_max_filter, _filter_number); i++) {
				_panels[i]->to_image(priv::ReconstructionHelper::process(_layers, i), prefix+std::to_string(i)+".png");

			}

		}

	private:
		QGridLayout _layout;
		std::vector<ReconstructionPanel<ColorType>*> _panels;

		size_t& _filter_number;
		std::vector<const Layer*> _layers;
		size_t _max_filter;
	};

}

#endif

#endif
