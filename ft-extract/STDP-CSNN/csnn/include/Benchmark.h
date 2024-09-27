#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <thread>
#include <future>
#include <condition_variable>

#include "Experiment.h"

using namespace std::chrono_literals;

template<template<typename> class Handler, typename Execution>
class Benchmark {

public:
	Benchmark(int& argc, char** argv, const std::string& name, size_t run_number) :
		_argc(argc), _argv(argv), _name(name), _run_number(run_number),
		_variables(), _futures(), _mutex(), _cv() {

	}

	Benchmark(const Benchmark<Handler, Execution>& that) = delete;
	Benchmark<Handler, Execution>& operator=(const Benchmark<Handler, Execution>& that) = delete;

	void add_variable(const std::string& name, const std::vector<float>& values) {
		_variables.emplace(name, values);
	}

	void add_variable_range(const std::string& name, float from, float to, float step) {
		std::vector<float> values;
		for(float i=from; i<= to; i += step) {
			values.push_back(i);
		}
		add_variable(name, values);
	}

	void run(size_t thread_number) {
		std::vector<std::map<std::string, float>> configuration_list;

		std::vector<std::string> key_set;
		for(const auto& entry : _variables) {
			key_set.push_back(entry.first);
		}
		_generate_configuration_list(configuration_list, key_set, 0, {});
		std::unique_lock<std::mutex> lock(_mutex);

		for(size_t i=0; i<_run_number; i++) {
			for(const std::map<std::string, float>& configuration : configuration_list) {
				_futures.emplace_back(std::async(std::launch::async, &Benchmark::_run_configuration, this, configuration, i));


				while(_futures.size() >= thread_number) {
					_cv.wait_for(lock, 1000ms);

					_futures.erase(std::remove_if(std::begin(_futures), std::end(_futures), [this](const std::future<bool>& future) {
						return future.wait_for(0ms) == std::future_status::ready;
					}), std::end(_futures));

				}
			}
		}

		while(!_futures.empty()) {
			_cv.wait_for(lock, 1000ms);

			_futures.erase(std::remove_if(std::begin(_futures), std::end(_futures), [this](const std::future<bool>& future) {
				return future.wait_for(0ms) == std::future_status::ready;
			}), std::end(_futures));
		}
	}



private:
	void _generate_configuration_list(std::vector<std::map<std::string, float>>& configuration_list, std::vector<std::string>& key_index, size_t depth, const std::map<std::string, float>& current) {
		if(depth >= key_index.size()) {
			configuration_list.push_back(current);
			return;
		}

		for(float v : _variables.at(key_index[depth])) {
			std::map<std::string, float> current_next(current);
			current_next.emplace(key_index[depth], v);
			_generate_configuration_list(configuration_list, key_index, depth+1, current_next);
		}
	}

	static bool _run_configuration(Benchmark* benchmark, const std::map<std::string, float>& configuration, size_t run) {
		std::string name = benchmark->_name;
		for(const auto& entry : configuration) {
			name += "-"+entry.first+"_"+std::to_string(entry.second);
		}
		name += "-"+std::to_string(run);


		std::cout << "Lauch benchmark configuration (" << name << "):" << std::endl;
		std::cout << "run: " << run << std::endl;
		for(const auto& entry : configuration) {
			std::cout << entry.first <<": " << entry.second << std::endl;
		}

		try {
			Experiment<Execution> experiment(name);
			Handler<Execution>::run(experiment, configuration);
			experiment.run(std::numeric_limits<size_t>::max());
		} catch(std::exception& e) {
			std::cerr << "Exception caught in configuration " << name << std::endl;
			std::cerr << e.what() << std::endl;

			std::unique_lock<std::mutex> lock(benchmark->_mutex);
			benchmark->_cv.notify_all();

			return false;
		}

		std::cout << "End benchmark configuration " << name << std::endl;

		benchmark->_mutex.lock();
		benchmark->_cv.notify_all();
		benchmark->_mutex.unlock();

		return true;
	}


	int& _argc;
	char** _argv;
	std::string _name;
	size_t _run_number;

	std::map<std::string, std::vector<float>> _variables;
	std::vector<std::future<bool>> _futures;


	std::mutex _mutex;
	std::condition_variable _cv;
};

#endif
