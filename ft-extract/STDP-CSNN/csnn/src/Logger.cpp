#include "Logger.h"

Logger::Logger() : _streams() {

}

Logger::~Logger() {
	for(OutputStream* stream : _streams) {
		delete stream;
	}
}

OutputStream& Logger::create() {
	_streams.push_back(new OutputStream);
	return *_streams.back();
}
