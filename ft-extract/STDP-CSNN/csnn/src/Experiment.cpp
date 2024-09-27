#include "Experiment.h"


#ifdef ENABLE_QT
AbstractExperiment::AbstractExperiment(int& argc, char** argv, const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file) :
	AbstractExperiment(output_path, model_path, name, seed, log_to_file) {
	_app = new QApplication(argc, argv);
}
#else
AbstractExperiment::AbstractExperiment(int&, char**, const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file) :
	AbstractExperiment(output_path, model_path, name, seed, log_to_file) {

}
#endif

AbstractExperiment::AbstractExperiment(const std::string& output_path, const std::string& model_path, const std::string& name, int seed, bool log_to_file) :
#ifdef ENABLE_QT
	_app(nullptr),
#endif
	_logger(), _log(_logger.create()), _print(_logger.create()), _output_path(output_path), _model_path(model_path), _name(name), _seed(seed), _random_generator(),
	_input_shape(nullptr), _time_limit(1.0), _train_data(), _test_data(), _process_list(),
#ifdef ENABLE_QT
	_plots(),
#endif
	_monitors(), _outputs() {

	_print.add_output(std::cout);

	size_t version = 0;

	// Create output directory if it does not exist
	mkdir(output_path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	while(true) {
		std::string filename = _output_path + "/" + ("log_" + _name + (version == 0 ? "" : "_" + std::to_string(version)) + ".txt");
		std::ifstream in_file(filename);
		if(!in_file.good()) {
			break;
		}
		version++;
	}

	if(version != 0) {
		std::cout << "Experiment " << _name << " already exists" << std::endl;
		_name += "_"+std::to_string(version);
		std::cout << "Experiment renamed in " << _name << std::endl;
	}

	std::seed_seq seed_seq{_seed};
	_random_generator.seed(seed_seq);

	_print << "Experiment " << _name << std::endl;

	_log.add_output(std::cout);
	if (log_to_file) {
		std::string filename = _output_path + "/" + "log_" +_name + ".txt";
		if(!_log.add_output<std::ofstream>(filename, std::ios::out).good()) {
			throw std::runtime_error("Can't open file " + filename);
		}
	}
	_print_date(_log) << std::endl;
	_log << "Random seed: " << _seed << std::endl;
	_log << std::endl;
}

AbstractExperiment::~AbstractExperiment() {
#ifdef ENABLE_QT
	for(auto& p : _plots) {
		delete p.first;
	}
	delete _app;
#endif
	delete _input_shape;


	for(AbstractProcess* p : _process_list) {
		delete p;
	}

	for(Output* o : _outputs) {
		delete o;
	}
}

void AbstractExperiment::load(const std::string& filename) {
	_load(filename);
}

void AbstractExperiment::save(const std::string& filename) const {
	_save(filename);
}
/*
void AbstractExperiment::add_train_step(Layer& layer, size_t epoch_number) {
	size_t layer_index = 0;
	while(layer_index < _layer.size()) {
		if(_layer[layer_index]->name() == layer.name()) {
			break;
		}
		layer_index++;
	}

	if(layer_index >= _layer.size()) {
		throw std::runtime_error("Layer not found: "+layer.name());
	}


	_train_step.emplace_back(layer_index, epoch_number);
}
*/

void AbstractExperiment::remove_output(size_t index) {
	if(index >= _outputs.size()) {
		throw std::runtime_error("remove_output: Invalid index");
	}
	auto it = _outputs.begin()+index;
	delete *it;
	_outputs.erase(_outputs.begin()+index);
}

void AbstractExperiment::remove_output(const std::string& name) {
	auto it = std::find_if(std::begin(_outputs), std::end(_outputs), [this, &name](const Output* o) {
		return o->name() == _name+"-"+name;
	});

	if(it == std::end(_outputs)) {
		throw std::runtime_error("Output not found: "+name);
	}
	delete *it;
	_outputs.erase(it);
}

void AbstractExperiment::remove_all_output() {
	for(Output* o : _outputs) {
		delete o;
	}
	_outputs.clear();
}

void AbstractExperiment::initialize(const Shape& input_shape) {
	_input_shape = new Shape(input_shape);
/*
	for(size_t i=0; i<_preprocessing.size(); i++) {
		_preprocessing[i]->_initialize(_random_generator);
	}
	_input_layer->converter()._initialize(_random_generator);*/
	for(size_t i=0; i<_process_list.size(); i++) {
		_process_list[i]->resize(i == 0 ? *_input_shape : _process_list[i-1]->shape());
		_process_list[i]->_initialize(_random_generator);
	}

	for(size_t i=0; i<_outputs.size(); i++) {
		_outputs[i]->converter()._initialize(_random_generator);

		for(size_t j=0; j<_outputs[i]->postprocessing().size(); j++) {
			_outputs[i]->postprocessing()[j]->_initialize(_random_generator);
		}

		for(size_t j=0; j<_outputs[i]->analysis().size(); j++) {
			_outputs[i]->analysis()[j]->_initialize(_random_generator);
		}
	}
}

void AbstractExperiment::run(size_t refresh_interval) {
	auto t_start = std::chrono::high_resolution_clock ::now();
	_log << "Run start at ";
	_print_date(_log) << std::endl;

	if(_input_shape == nullptr) {
		std::runtime_error("Require input data");
	}

	if(_outputs.empty()) {
		std::runtime_error("Require output(s)");
	}

	Shape current_shape = *_input_shape;

	_log << "Input data " << current_shape.to_string() << std::endl;
	_log << "Train:" << std::endl;
	for(size_t i=0; i<_train_data.size(); i++) {
		_log << "#" << (i+1) << ": " << _train_data[i]->to_string() << std::endl;
	}
	_log << "Test:" << std::endl;
	for(size_t i=0; i<_test_data.size(); i++) {
		_log << "#" << (i+1) << ": " << _test_data[i]->to_string() << std::endl;
	}
	_log << std::endl;


	for(size_t i=0; i<_process_list.size(); i++) {
		current_shape = _process_list[i]->resize(current_shape);
		_process_list[i]->_initialize(_random_generator);

		_log << _process_list[i]->class_name() << " " << (i+1) << ": " << _process_list[i]->name() << " " << current_shape.to_string() << std::endl;
		_process_list[i]->print_parameters(_log);
		_log << std::endl;
		_log << std::endl;
	}

	_log << std::endl;

	for(size_t i=0; i<_outputs.size(); i++) {
		size_t output_index = _outputs[i]->index();
		Shape current_output_shape = _process_list[output_index]->shape();
		_log << "Output " << (i+1) << " of " << _process_list[output_index]->name() << " " << current_output_shape.to_string() << ": " << _outputs[i]->name()  << std::endl;
		_outputs[i]->converter()._initialize(_random_generator);
		_outputs[i]->converter().print_parameters(_log);
		_log << std::endl;
		_log << std::endl;


		for(size_t j=0; j<_outputs[i]->postprocessing().size(); j++) {
			_outputs[i]->postprocessing()[j]->_initialize(_random_generator);
			_outputs[i]->postprocessing()[j]->resize(current_output_shape);
			current_output_shape = _outputs[i]->postprocessing()[j]->shape();
			_log << "Output " << (i+1) << ", Postprocess " << (j+1) << " " << current_output_shape.to_string() << ":" << std::endl;
			_outputs[i]->postprocessing()[j]->print_parameters(_log);
			_log << std::endl;
			_log << std::endl;
		}

		for(size_t j=0; j<_outputs[i]->analysis().size(); j++) {
			_log << "Output " << (i+1) << ", Analysis: " << (j+1) << std::endl;
			_outputs[i]->analysis()[j]->_initialize(_random_generator);
			_outputs[i]->analysis()[j]->resize(current_output_shape);
			_outputs[i]->analysis()[j]->print_parameters(_log);
			_log << std::endl;
			_log << std::endl;
		}
	}
#ifdef ENABLE_QT
	for(size_t i=0; i<_plots.size(); i++) {
		_plots[i].first->initialize();
		_plots[i].first->show();
	}
#endif
	process(refresh_interval);

	//_save(_output_path + "/" + "param"+_name);

	auto t_end = std::chrono::high_resolution_clock ::now();

	_log << "Run end at ";
	_print_date(_log) << std::endl;

	auto duration = t_end-t_start;

	auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
	auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration-hours);
	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration-hours-minutes);

	_log << "Duration: " << hours.count() << "h " << minutes.count() << "m " << seconds.count() << "s" << std::endl;


}

int AbstractExperiment::wait() {
#ifdef ENABLE_QT
	return _app->exec();
#else
	return 0;
#endif
}

#ifdef ENABLE_QT
void AbstractExperiment::tick(size_t current_layer_index, size_t sample_count) {
	for(auto& p : _plots) {
		if(p.second < 0 || static_cast<size_t>(p.second) == current_layer_index)
			p.first->on_tick();
	}

	for(Monitor* m : _monitors) {
		m->on_sample(*this, current_layer_index, sample_count);
	}
}

void AbstractExperiment::refresh(size_t current_layer_index) {
	for(auto& p : _plots) {
		if(p.second < 0 || static_cast<size_t>(p.second) == current_layer_index)
			p.first->on_refresh();
	}
	_app->processEvents();
}
#else
void AbstractExperiment::tick(size_t current_layer_index, size_t sample_count) {
	for(Monitor* m : _monitors) {
		m->on_sample(*this, current_layer_index, sample_count);
	}
}

void AbstractExperiment::refresh(size_t) {

}
#endif

void AbstractExperiment::epoch(size_t current_layer_index, size_t epoch_count) {
	for(Monitor* m : _monitors) {
		m->on_epoch(*this, current_layer_index, epoch_count);
	}
}

const std::string& AbstractExperiment::name() const {
	return _name;
}

OutputStream& AbstractExperiment::log() const {
	return _log;
}

OutputStream& AbstractExperiment::print() const {
	return _print;
}

std::default_random_engine& AbstractExperiment::random_generator() {
	return _random_generator;
}
/*
InputLayer& AbstractExperiment::input_layer() {
	return *_input_layer;
}

const InputLayer& AbstractExperiment::input_layer() const {
	return *_input_layer;
}
*/
const std::vector<Input*>& AbstractExperiment::train_data() const {
	return _train_data;
}

const std::vector<Input*>& AbstractExperiment::test_data() const {
	return _test_data;
}

const std::string& AbstractExperiment::output_path() const {
	return _output_path;
}

const std::string& AbstractExperiment::model_path() const {
	return _model_path;
}

/*const std::vector<Process*>& AbstractExperiment::preprocessing() const {
	return _preprocessing;
}

const std::vector<std::pair<size_t, size_t>>& AbstractExperiment::train_step() const {
	return _train_step;
}*/

const Shape& AbstractExperiment::input_shape() const {
	return *_input_shape;
}

Time AbstractExperiment::time_limit() const {
	return _time_limit;
}

size_t AbstractExperiment::process_number() const {
	return _process_list.size();
}

AbstractProcess& AbstractExperiment::process_at(size_t index) {
	return *_process_list.at(index);
}

const AbstractProcess& AbstractExperiment::process_at(size_t index) const {
	return *_process_list.at(index);
}

/*
Layer& AbstractExperiment::layer_at(size_t i) {
	return *_layer.at(i);
}

const Layer& AbstractExperiment::layer_at(size_t i) const {
	return *_layer.at(i);
}

Layer& AbstractExperiment::layer(const std::string& name) {
	auto it = std::find_if(std::begin(_layer), std::end(_layer), [&name](Layer* entry) {
		return entry->name() == name;
	});

	if(it == std::end(_layer)) {
		throw std::runtime_error("Layer not found: "+name);
	}

	return **it;
}

const Layer& AbstractExperiment::layer(const std::string& name) const {
	auto it = std::find_if(std::begin(_layer), std::end(_layer), [&name](Layer* entry) {
		return entry->name() == name;
	});

	if(it == std::end(_layer)) {
		throw std::runtime_error("Layer not found: "+name);
	}

	return **it;
}

size_t AbstractExperiment::layer_count() const {
	return _layer.size();
}
*/
Output& AbstractExperiment::output_at(size_t i) {
	return *_outputs.at(i);
}

const Output& AbstractExperiment::output_at(size_t i) const {
	return *_outputs.at(i);
}

size_t AbstractExperiment::output_count() const {
	return _outputs.size();
}

#ifdef ENABLE_QT
void AbstractExperiment::add_plot(Plot* plot, int display) {
	_plots.emplace_back(plot, display);
	plot->resize(800, 800);
}
#endif

std::ostream& AbstractExperiment::_print_date(std::ostream& stream) const {

	time_t t = ::time(nullptr);
	tm* local_t = ::localtime(&t);

	stream	<< (local_t->tm_hour) << ":" << (local_t->tm_min) << ":" << (local_t->tm_sec) << " "
			<< (local_t->tm_mday) << "/" << (local_t->tm_mon)+1 << "/" << (local_t->tm_year+1900);
	return stream;
}

void AbstractExperiment::_save(const std::string& filename) const {
	std::ofstream file(filename, std::ios::out | std::ios::trunc | std::ios::binary);

	if(!file.good()) {
		throw std::runtime_error("Unable to open param"+_name);
	}
/*
	uint32_t preprocessing_size = _preprocessing.size();
	file.write(reinterpret_cast<const char*>(&preprocessing_size), sizeof(uint32_t));

	for(Process* entry : _preprocessing) {
		entry->save(file);
	}

	_input_layer->converter().save(file);
*/
	uint32_t process_size = _process_list.size();
	file.write(reinterpret_cast<const char*>(&process_size), sizeof(uint32_t));

	for(AbstractProcess* entry : _process_list) {
		Persistence::save_string(entry->name(), file);
		entry->save(file);
	}

	uint32_t output_size = _outputs.size();
	file.write(reinterpret_cast<const char*>(&output_size), sizeof(uint32_t));
	for(Output* entry : _outputs) {

		uint32_t layer_index = entry->index();
		file.write(reinterpret_cast<const char*>(&layer_index), sizeof(uint32_t));

		entry->converter().save(file);

		uint32_t postprocessing_size = entry->postprocessing().size();
		file.write(reinterpret_cast<const char*>(&postprocessing_size), sizeof(uint32_t));

		for(Process* entry2 : entry->postprocessing()) {
			entry2->save(file);
		}

		uint32_t analysis_size = entry->analysis().size();
		file.write(reinterpret_cast<const char*>(&analysis_size), sizeof(uint32_t));

		for(Analysis* entry2 : entry->analysis()) {
			entry2->save(file);
		}
	}
}

void AbstractExperiment::_load(const std::string& filename) {
	std::ifstream file(filename, std::ios::in | std::ios::binary);

	if(!file.good()) {
		throw std::runtime_error("Unable to open "+filename);
	}

	uint32_t preprocessing_size;
	file.read(reinterpret_cast<char*>(&preprocessing_size), sizeof(uint32_t));
/*
	for(size_t i=0; i<preprocessing_size; i++) {
		_preprocessing.push_back(ClassParameterRegistry::load_expected<Process>(file));
	}

	_input_layer = new InputLayer(this, ClassParameterRegistry::load_expected<InputConverter>(file));
*/
	uint32_t process_size;
	file.read(reinterpret_cast<char*>(&process_size), sizeof(uint32_t));

	for(size_t i=0; i<process_size; i++) {
		std::string name = Persistence::load_string(file);
		_process_list.push_back(ClassParameterRegistry::load_expected<AbstractProcess>(file));
		//_process_list.back()->set_info(name, _layer.size()-1, this);
	}

	uint32_t output_size;
	file.read(reinterpret_cast<char*>(&output_size), sizeof(uint32_t));

	for(size_t i=0; i<output_size; i++) {
		uint32_t layer_index;
		file.read(reinterpret_cast<char*>(&layer_index), sizeof(uint32_t));

		_outputs.push_back(new Output(this, _name+"-"+_process_list.at(layer_index)->name(), layer_index, ClassParameterRegistry::load_expected<OutputConverter>(file)));

		uint32_t postprocessing_size;
		file.read(reinterpret_cast<char*>(&postprocessing_size), sizeof(uint32_t));

		for(size_t j=0; j<postprocessing_size; j++) {
			_outputs.back()->_postprocessing.push_back(ClassParameterRegistry::load_expected<Process>(file));
		}

		uint32_t analysis_size;
		file.read(reinterpret_cast<char*>(&analysis_size), sizeof(uint32_t));

		for(size_t j=0; j<analysis_size; j++) {
			_outputs.back()->_analysis.push_back(ClassParameterRegistry::load_expected<Analysis>(file));
		}

	}

}

void AbstractExperiment::_check_data_shape(const Shape& shape) {
	if(_input_shape == nullptr) {
		_input_shape = new Shape(shape);
	}
	else {
		if(*_input_shape != shape) {
			throw std::runtime_error("All data have to be in the same shape: actual: "+
									 shape.to_string()+", expected: "+_input_shape->to_string());
		}
	}
}
