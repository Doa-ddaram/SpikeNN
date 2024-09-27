#ifndef _EXPERIMENT_H
#define _EXPERIMENT_H

#ifdef ENABLE_QT
#include <QApplication>
#endif

#include <string>
#include <ctime>
#include <fstream>
#include <iostream>
#include <chrono>
#include "Input.h"
#include "Process.h"
#include "Layer.h"
#include "Plot.h"
#include "Output.h"
#include "InputLayer.h"
#include "Logger.h"
#include "Monitor.h"
#include <sys/stat.h>
#include <unistd.h>

class AbstractExperiment {

	friend class InputLayer;
	friend class Layer;

public:
	AbstractExperiment(const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file);
	AbstractExperiment(int& argc, char** argv, const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file);
	AbstractExperiment(const AbstractExperiment& that) = delete;
	virtual ~AbstractExperiment();

	AbstractExperiment& operator=(const AbstractExperiment& that) = delete;

	void load(const std::string& filename);
	void save(const std::string& filename) const;

	template<typename T, typename... Args>
	void add_train(Args&&... args) {
		_train_data.push_back(new T(std::forward<Args>(args)...));
		_check_data_shape(_train_data.back()->shape());
	}

	template<typename T, typename... Args>
	void add_test(Args&&... args) {
		_test_data.push_back(new T(std::forward<Args>(args)...));
		_check_data_shape(_test_data.back()->shape());
	}

	template<typename T, typename... Args>
	T& push(Args&&... args) {
		T* process = new T(std::forward<Args>(args)...);
		process->_set_info(_process_list.size(), this);
		_process_list.push_back(process);
		return *process;
	}
/*
	template<typename T, typename... Args>
	void add_preprocessing(Args&&... args) {
		_preprocessing.push_back(new T(std::forward<Args>(args)...));
	}

	template<typename Converter, typename... Args>
	InputLayer& input(Args&&... args) {
		_input_layer = new InputLayer(this, new Converter(std::forward<Args>(args)...));
		return *_input_layer;
	}

	template<typename T, typename... Args>
	T& push_layer(const std::string& name, Args&&... args) {
		T* layer = new T(std::forward<Args>(args)...);

		layer->set_info(name, _layer.size(), this);
		_layer.push_back(layer);

		for(size_t i=0; i<_layer.size()-1; i++) {
			if(layer->name() == _layer[i]->name()) {
				throw std::runtime_error("Layer name "+layer->name()+" already exists");
			}
		}

		return *layer;
	}

	void add_train_step(Layer& layer, size_t epoch_number);
*/
	template<typename T, typename... Args>
	Output& output(const Layer& layer, Args&&... args) {
		Output* obj = new Output(this, _name+"-"+layer.name(), layer.index(), new T(std::forward<Args>(args)...));
		_outputs.push_back(obj);
		return *obj;
	}


	void remove_output(size_t index);
	void remove_output(const std::string& name);
	void remove_all_output();

	void initialize(const Shape& input_shape);
	void run(size_t refresh_interval);
	int wait();

	void tick(size_t current_layer_index, size_t sample_count);
	void refresh(size_t current_layer_index);
	void epoch(size_t current_layer_index, size_t epoch_count);

	const std::string& name() const;

	OutputStream& log() const;
	OutputStream& print() const;

	std::default_random_engine& random_generator();
/*
	InputLayer& input_layer();
	const InputLayer& input_layer() const;
*/
	const Shape& input_shape() const;

	Time time_limit() const;

	const std::vector<Input*>& train_data() const;
	const std::vector<Input*>& test_data() const;
	const std::string& output_path() const;
	const std::string& model_path() const;

	//const std::vector<Process*>& preprocessing() const;

	//const std::vector<std::pair<size_t, size_t>>& train_step() const;

	size_t process_number() const;
	AbstractProcess& process_at(size_t index);
	const AbstractProcess& process_at(size_t index) const;
/*
	Layer& layer_at(size_t i);
	const Layer& layer_at(size_t i) const;

	Layer& layer(const std::string& name);
	const Layer& layer(const std::string& name) const;

	size_t layer_count() const;
*/
	Output& output_at(size_t i);
	const Output& output_at(size_t i) const;
	size_t output_count() const;

	virtual Tensor<Time> compute_time_at(size_t i) const = 0;

protected:
	virtual void process(size_t refresh_interval) = 0;

#ifdef ENABLE_QT
	void add_plot(Plot* plot, int display);
#endif

	std::ostream& _print_date(std::ostream& stream) const;

	void _save(const std::string& filename) const;
	void _load(const std::string& filename);
	void _check_data_shape(const Shape& shape);

#ifdef ENABLE_QT
	QApplication* _app;
#endif

	Logger _logger;
	OutputStream& _log;
	OutputStream& _print;

	int _seed;
	std::string _output_path;
	std::string _model_path;
	std::string _name;
	std::default_random_engine _random_generator;

	//std::vector<Process*> _preprocessing;

	Shape* _input_shape;

	Time _time_limit;

	std::vector<Input*> _train_data;
	std::vector<Input*> _test_data;

	//InputLayer* _input_layer;
	//std::vector<Layer*> _layer;

	//std::vector<std::pair<size_t, size_t>> _train_step;
	std::vector<AbstractProcess*> _process_list;

#ifdef ENABLE_QT
	std::vector<std::pair<Plot*, int>> _plots;
#endif

	std::vector<Monitor*> _monitors;

	std::vector<Output*> _outputs;
};

template<typename ExecutionPolicy>
class Experiment : public AbstractExperiment {

public:
	template<typename... Args>
	Experiment(int& argc, char** argv, const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file, Args&&... args) :
		AbstractExperiment(argc, argv, output_path, model_path, name, seed, log_to_file), _execution(*this, std::forward<Args>(args)...), _train_set(), _test_set() {

	}

	template<typename... Args>
	Experiment(const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file, Args&&... args) :
		AbstractExperiment(output_path, model_path, name, seed, log_to_file), _execution(*this, std::forward<Args>(args)...), _train_set(), _test_set() {
	}

	virtual void process(size_t refresh_interval) {
		_execution.process(refresh_interval);
	}


	virtual Tensor<Time> compute_time_at(size_t i) const {
		return _execution.compute_time_at(i);
	}

private:
	ExecutionPolicy _execution;

	std::vector<std::pair<std::string, Tensor<float>>> _train_set;
	std::vector<std::pair<std::string, Tensor<float>>> _test_set;

};

#endif
