#ifndef _LOGGER_H
#define _LOGGER_H

#include <vector>
#include <algorithm>
#include <streambuf>
#include <ostream>

template <typename char_type,  typename traits = std::char_traits<char_type>>
class BasicStreamBuf : public std::basic_streambuf<char_type, traits> {

public:
	typedef typename traits::int_type int_type;


	BasicStreamBuf() : _bufs() {

	}

	~BasicStreamBuf() {

	}

	void add_output(std::basic_streambuf<char_type, traits>* buf) {
		_bufs.push_back(buf);
	}

private:
	int sync() {
		return std::all_of(std::begin(_bufs), std::end(_bufs), [](std::basic_streambuf<char_type, traits>* buf) {
			return buf->pubsync() == 0;
		}) ? 0 : -1;
	}

	int_type overflow(int_type c) {
		int_type const eof = traits::eof();

		if(traits::eq_int_type(c, eof)) {
			return traits::not_eof(c);
		}
		else {
			char_type const ch = traits::to_char_type(c);

			return std::any_of(std::begin(_bufs), std::end(_bufs), [ch, eof](std::basic_streambuf<char_type, traits>* buf) {
				return traits::eq_int_type(buf->sputc(ch), eof);
			}) ? eof : c;
		}

	}

	std::vector<std::basic_streambuf<char_type, traits>*> _bufs;

};



template <typename char_type,  typename traits = std::char_traits<char_type>>
class BasicOutputStream : public std::basic_ostream<char_type, traits> {
public:
	BasicOutputStream() : std::basic_ostream<char_type, traits>(&_buf), _outputs(), _buf() {

	}

	~BasicOutputStream() {
		for(std::basic_ostream<char_type, traits>* output : _outputs) {
			delete output;
		}
	}

	template<typename T, typename... Args>
	T& add_output(Args&&... args) {
		T* obj = new T(std::forward<Args>(args)...);
		_outputs.push_back(obj);
		_buf.add_output(obj->rdbuf());
		return *obj;
	}

	void add_output(std::basic_ostream<char_type, traits>& output) {
		_buf.add_output(output.rdbuf());
	}

private:
	std::vector<std::basic_ostream<char_type, traits>*> _outputs;
	BasicStreamBuf<char_type, traits> _buf;
};


typedef BasicOutputStream<char> OutputStream;

class Logger {

public:
	Logger();
	~Logger();

	OutputStream& create();


private:
	std::vector<OutputStream*> _streams;

};

#endif
