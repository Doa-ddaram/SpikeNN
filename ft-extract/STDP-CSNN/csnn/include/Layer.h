#ifndef _LAYER_H
#define _LAYER_H

#include <tuple>
#include <functional>

#include "Tensor.h"
#include "Spike.h"
#include "ClassParameter.h"
#include "Plot.h"
#include "plot/Reconstruction.h"
#include "plot/Time.h"
#include "Process.h"
#include "SpikeConverter.h"

class AbstractExperiment;


class Layer : public AbstractProcess {

	friend class AbstractExperiment;

public:
	template<typename T, typename Factory>
	Layer(const RegisterClassParameter<T, Factory>& registration) : Layer(registration, 0, 0, 0) {

	}

	template<typename T, typename Factory>
	Layer(const RegisterClassParameter<T, Factory>& registration, size_t width, size_t height, size_t depth) :
		AbstractProcess(registration), _require_sorted(true),
		_width(width), _height(height), _depth(depth), _current_width(width), _current_height(height) {

	}

	Layer(const Layer& layer) = delete;

	virtual ~Layer();

	Layer& operator=(const Layer& layer) = delete;

	void set_size(size_t width, size_t height);

    size_t width() const;
    size_t height() const;
    size_t depth() const;

	bool require_sorted() const;

	virtual void train(const std::string& label, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike) = 0;
	virtual void test(const std::string& label, const std::vector<Spike>& input_spike, const Tensor<Time>& input_time, std::vector<Spike>& output_spike) = 0;

	virtual void on_epoch_start() {

	}

	virtual void on_epoch_end() {

	}

    virtual std::pair<uint16_t, uint16_t> receptive_field_of(const std::pair<uint16_t, uint16_t>& in) const = 0;
    virtual Tensor<float> reconstruct(const Tensor<float>& t) const = 0;

#ifdef ENABLE_QT
    template<typename ColorType = DefaultColor>
	void plot_reconstruction(bool only_in_train = false, size_t max_filter = std::numeric_limits<size_t>::max()) {

		add_plot<plot::Reconstruction<ColorType>>(only_in_train, _previous_layer_list(), std::ref(_depth), max_filter);
    }

	void plot_time(bool only_in_train, size_t n = 20, float min = 0.0, float max = 1.0);
#endif

protected:
#ifdef ENABLE_QT
    template<typename PlotType, typename... Args>
    void add_plot(bool only_in_train, Args&&... args) {
		_add_plot(new PlotType(name(), std::forward<Args>(args)...), only_in_train);
    }

	void _add_plot(Plot* plot, bool only_in_train);
#endif

    std::vector<const Layer*> _previous_layer_list() const;
	bool _require_sorted;

    size_t _width;
    size_t _height;
    size_t _depth;

    size_t _current_width;
    size_t _current_height;

private:
    void set_info(const std::string& name, size_t index, AbstractExperiment* experiment);
};

class Layer3D : public Layer {

public:
    template<typename T, typename Factory>
	Layer3D(const RegisterClassParameter<T, Factory>& registration) : Layer(registration),
		_filter_width(0), _filter_height(0), _filter_number(0),
					_stride_x(0), _stride_y(0), _padding_x(0), _padding_y(0) {

        add_parameter("filter_width", _filter_width);
        add_parameter("filter_height", _filter_height);
        add_parameter("filter_number", _filter_number);
        add_parameter("stride_x", _stride_x);
        add_parameter("stride_y", _stride_y);
        add_parameter("padding_x", _padding_x);
        add_parameter("padding_y", _padding_y);
    }

    template<typename T, typename Factory>
    Layer3D(const RegisterClassParameter<T, Factory>& registration,
            size_t filter_width, size_t filter_height, size_t filter_number,
            size_t stride_x, size_t stride_y,
            size_t padding_x, size_t padding_y) : Layer3D(registration) {

        parameter<size_t>("filter_width").set(filter_width);
        parameter<size_t>("filter_height").set(filter_height);
        parameter<size_t>("filter_number").set(filter_number);
        parameter<size_t>("stride_x").set(stride_x);
        parameter<size_t>("stride_y").set(stride_y);
        parameter<size_t>("padding_x").set(padding_x);
        parameter<size_t>("padding_y").set(padding_y);

    }

	virtual Shape compute_shape(const Shape& previous_shape);

    void forward(uint16_t x, uint16_t y, std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>>& output);
    std::pair<uint16_t, uint16_t> to_input_coord(uint16_t x, uint16_t y, uint16_t w_x, uint16_t w_y) const;
    bool is_valid_input_coord(const std::pair<uint16_t, uint16_t>& coord) const;

    virtual std::pair<uint16_t, uint16_t> receptive_field_of(const std::pair<uint16_t, uint16_t>& in) const;
protected:
    size_t _filter_width;
    size_t _filter_height;
    size_t _filter_number;
    size_t _stride_x;
    size_t _stride_y;
    size_t _padding_x;
    size_t _padding_y;

};

class LayerFactory : public ClassParameterFactory<Layer, LayerFactory> {

public:
    LayerFactory() : ClassParameterFactory<Layer, LayerFactory>("Layer") {

    }

};


#endif
