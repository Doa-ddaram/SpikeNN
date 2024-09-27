#ifndef _NUMPY_STRUCT_H
#define _NUMPY_STRUCT_H

#include <map>
#include <memory>
#include <fstream>
#include <cstring>

#include "Tensor.h"
#include "Endianness.h"

#define ZIP_FILE_MAGIC 0x04034b50
#define ZIP_DICTIONNARY_MAGIC 0x02014b50

#define NPY_MAGIC_1 0x93
#define NPY_MAGIC_2 "NUMPY"

enum NumpyHeaderObjectType {
	NUMPY_HEADER_BOOL,
	NUMPY_HEADER_INT,
	NUMPY_HEADER_STRING,
	NUMPY_HEADER_TUPLE,
	NUMPY_HEADER_MAP
};

class NumpyHeaderObject {

public:
	NumpyHeaderObject(NumpyHeaderObjectType p_type) : _type(p_type) {

	}

	virtual ~NumpyHeaderObject() {

	}

	virtual void print(std::ostream& stream) = 0;


	static std::string print_next_token(const std::string& str, size_t cursor) {
		return cursor < str.size() ? "'"+str.substr(cursor)+"'" : "EOF";
	}

	static std::unique_ptr<NumpyHeaderObject> read(const std::string& str, size_t& cursor);

private:
	NumpyHeaderObjectType _type;
};

class NumpyHeaderBool : public NumpyHeaderObject {

public:

	NumpyHeaderBool(bool p_value) : NumpyHeaderObject(NUMPY_HEADER_BOOL), _value(p_value) {

	}

	virtual ~NumpyHeaderBool() {

	}

	bool value() const {
		return _value;
	}

	virtual void print(std::ostream& stream) {
		stream << (_value ? "True" : "False");
	}

	static std::unique_ptr<NumpyHeaderObject> read(const std::string& str, size_t& cursor) {
		static std::string true_symbol = "true";
		static std::string false_symbol = "false";

		std::string str_true = str.substr(cursor, true_symbol.size());
		std::transform(std::begin(str_true), std::end(str_true), std::begin(str_true), ::tolower);

		if(str_true == true_symbol) {
			cursor += true_symbol.size();
			return std::unique_ptr<NumpyHeaderObject>(new NumpyHeaderBool(true));
		}

		std::string str_false = str.substr(cursor, false_symbol.size());
		std::transform(std::begin(str_false), std::end(str_false), std::begin(str_false), ::tolower);
		if(str_false == false_symbol) {
			cursor += false_symbol.size();
			return std::unique_ptr<NumpyHeaderObject>(new NumpyHeaderBool(false));
		}

		return std::unique_ptr<NumpyHeaderObject>();
	}

private:
	bool _value;
};

class NumpyHeaderInt : public NumpyHeaderObject {

public:
	NumpyHeaderInt(int p_value) : NumpyHeaderObject(NUMPY_HEADER_INT), _value(p_value) {

	}

	virtual ~NumpyHeaderInt() {

	}

	int value() const {
		return _value;
	}

	virtual void print(std::ostream& stream) {
		stream << _value;
	}

	static std::unique_ptr<NumpyHeaderObject> read(const std::string& str, size_t& cursor) {
		size_t current_cursor = cursor;

		while(current_cursor < str.size() && str.at(current_cursor) >= '0' && str.at(current_cursor) <= '9')
			current_cursor++;

		if(current_cursor > cursor) {
			size_t start_cursor = cursor;
			cursor = current_cursor;
			std ::string int_str = str.substr(start_cursor, current_cursor-start_cursor);
			return std::unique_ptr<NumpyHeaderObject>(new NumpyHeaderInt(::atoi(int_str.c_str())));
		}

		return std::unique_ptr<NumpyHeaderObject>();
	}

private:
	int _value;
};

class NumpyHeaderString : public NumpyHeaderObject {

public:
	NumpyHeaderString(std::string&& p_value) : NumpyHeaderObject(NUMPY_HEADER_STRING), _value(p_value) {

	}

	virtual ~NumpyHeaderString() {

	}

	const std::string& value() const {
		return _value;
	}

	virtual void print(std::ostream& stream) {
		stream << '"' << _value << '"';
	}

	static std::unique_ptr<NumpyHeaderObject> read(const std::string& str, size_t& cursor) {
		size_t start_cursor = cursor;
		size_t current_cursor = cursor;
		char start_quote = str.at(current_cursor);

		if(start_quote != '"' && start_quote != '\'') {
			return std::unique_ptr<NumpyHeaderObject>();
		}

		current_cursor++;

		while(current_cursor < str.size() && str.at(current_cursor) != start_quote)
			current_cursor++;

		if(str.at(current_cursor) != start_quote) {
			return std::unique_ptr<NumpyHeaderObject>();
		}

		cursor = current_cursor+1;

		return std::unique_ptr<NumpyHeaderObject>(new NumpyHeaderString(str.substr(start_cursor+1, current_cursor-start_cursor-1)));
	}

private:
	std::string _value;
};

class NumpyHeaderTuple: public NumpyHeaderObject {

public:
	NumpyHeaderTuple(std::vector<std::unique_ptr<NumpyHeaderObject>>&& p_value) : NumpyHeaderObject(NUMPY_HEADER_TUPLE), _value(std::move(p_value)) {

	}

	virtual ~NumpyHeaderTuple() {

	}

	virtual void print(std::ostream& stream) {
		stream << '(';
		for(size_t i=0; i<_value.size(); i++) {
			if(i != 0)
				stream << ", ";
			_value.at(i)->print(stream);
		}
		stream << ')';
	}

	const std::vector<std::unique_ptr<NumpyHeaderObject>>& value() const {
		return _value;
	}

	static std::unique_ptr<NumpyHeaderObject> read(const std::string& str, size_t& cursor) {
		size_t current_cursor = cursor;

		if(str.at(current_cursor) != '(') {
			return std::unique_ptr<NumpyHeaderObject>();
		}

		current_cursor++;

		std::vector<std::unique_ptr<NumpyHeaderObject>> elements;


		while(current_cursor < str.size() && str.at(current_cursor) != ')') {
			elements.emplace_back(std::move(NumpyHeaderObject::read(str, current_cursor)));

			while(current_cursor < str.size() && std::isspace(str.at(current_cursor)))
				current_cursor++;

			if(str.at(current_cursor) == ',') {
				current_cursor++;
			}
			else {
				break;
			}

			while(current_cursor < str.size() && std::isspace(str.at(current_cursor)))
				current_cursor++;

		}

		if(str.at(current_cursor) != ')') {
			throw std::runtime_error("Expected ')', got "+NumpyHeaderObject::print_next_token(str, current_cursor));
		}

		cursor = current_cursor+1;

		return std::unique_ptr<NumpyHeaderObject>(new NumpyHeaderTuple(std::move(elements)));
	}

private:
	std::vector<std::unique_ptr<NumpyHeaderObject>> _value;
};

class NumpyHeaderMap : public NumpyHeaderObject {

public:
	NumpyHeaderMap(std::map<std::string, std::unique_ptr<NumpyHeaderObject>>&& p_value) : NumpyHeaderObject(NUMPY_HEADER_MAP), _value(std::move(p_value)) {

	}

	virtual ~NumpyHeaderMap() {

	}

	const std::map<std::string, std::unique_ptr<NumpyHeaderObject>>& value() const {
		return _value;
	}

	virtual void print(std::ostream& stream) {
		stream << '{';
		size_t i = 0;
		for(const auto& entry : _value) {
			if(i != 0)
				stream << ", ";
			stream << entry.first << ": ";
			entry.second->print(stream);
			i++;
		}
		stream << '}';
	}

	static std::unique_ptr<NumpyHeaderObject> read(const std::string& str, size_t& cursor) {
		size_t current_cursor = cursor;

		if(str.at(current_cursor) != '{') {
			return std::unique_ptr<NumpyHeaderObject>();
		}

		current_cursor++;

		std::map<std::string, std::unique_ptr<NumpyHeaderObject>> elements;

		while(current_cursor < str.size() && str.at(current_cursor) != '}') {
			std::unique_ptr<NumpyHeaderObject> key = NumpyHeaderString::read(str, current_cursor);

			if(!key)
				throw std::runtime_error("Expected string, got "+NumpyHeaderObject::print_next_token(str, current_cursor));

			while(current_cursor < str.size() && std::isspace(str.at(current_cursor)))
				current_cursor++;

			if(str.at(current_cursor) != ':') {
				throw std::runtime_error("Expected ':', got "+NumpyHeaderObject::print_next_token(str, current_cursor));
			}

			current_cursor++;

			elements.emplace(std::piecewise_construct, std::forward_as_tuple(static_cast<NumpyHeaderString&>(*key).value()), std::forward_as_tuple(std::move(NumpyHeaderObject::read(str, current_cursor))));

			while(current_cursor < str.size() && std::isspace(str.at(current_cursor)))
				current_cursor++;

			if(str.at(current_cursor) == ',') {
				current_cursor++;
			}
			else {
				break;
			}

			while(current_cursor < str.size() && std::isspace(str.at(current_cursor)))
				current_cursor++;

		}


		if(str.at(current_cursor) != '}') {
			throw std::runtime_error("Expected '}', got "+NumpyHeaderObject::print_next_token(str, current_cursor));
		}

		cursor = current_cursor+1;

		return std::unique_ptr<NumpyHeaderObject>(new NumpyHeaderMap(std::move(elements)));
	}

private:
	std::map<std::string, std::unique_ptr<NumpyHeaderObject>> _value;
};

class NumpyArray {

public:
	NumpyArray(const std::vector<size_t>& dimension) :
		_dimension(dimension),
		_size(std::accumulate(std::begin(dimension), std::end(dimension), 1, std::multiplies<size_t>())),
		_array(new double[_size]) {

	}

	template<typename... Args>
	NumpyArray(Args... args) : _dimension(),_size(0), _array(nullptr) {
		_initialize(args...);
	}

	NumpyArray(const NumpyArray& that) noexcept : _dimension(that._dimension), _size(that._size), _array(new double[_size]) {
		std::copy(that._array, that._array+_size, _array);
	}

	NumpyArray(NumpyArray&& that) noexcept : _dimension(std::move(that._dimension)), _size(that._size), _array(that._array) {
		that._size = 0;
		that._array = nullptr;
	}

	~NumpyArray() {
		delete[] _array;
	}

	NumpyArray& operator=(const NumpyArray& that) noexcept {
		delete[] _array;
		_dimension = that._dimension;
		_size = that._size;
		_array = new double[_size];
		std::copy(that._array, that._array+_size, _array);
		return *this;
	}

	NumpyArray& operator=(NumpyArray&& that) noexcept {
		delete[] _array;
		_dimension = std::move(that._dimension);
		_size = that._size;
		_array = that._array;
		that._size = 0;
		that._array = nullptr;
		return *this;
	}

	template<typename T>
	Tensor<T> to_tensor() const {
		Tensor<T> out(_dimension);

		size_t size = out.shape().product();

		for(size_t i=0; i<size; i++) {
			out.at_index(i) = _array[i];
		}

		return out;
	}

	size_t dimension_number() const {
		return _dimension.size();
	}

	size_t dimension(size_t index) const {
		return _dimension.at(index);
	}

	double& at_index(size_t index) {
		return _array[index];
	}

	const double& at_index(size_t index) const {
		return _array[index];
	}

	template<typename... Index>
	double& at(Index... index) {
		return _array[to_index<Index...>(index...)];
	}

	template<typename... Index>
	const double& at(Index... index) const {
		return _array[to_index<Index...>(index...)];
	}

	size_t size() const {
		return _size;
	}

	void reshape(const std::vector<size_t>& new_dimension) {
		if(std::accumulate(std::begin(_dimension), std::end(_dimension), 1, std::multiplies<size_t>()) !=
				std::accumulate(std::begin(new_dimension), std::end(new_dimension), 1, std::multiplies<size_t>())) {
			throw std::runtime_error("Incompatible dimension");
		}
		_dimension = new_dimension;
	}

	void reorder(const std::vector<size_t>& dst_index) {
		double* new_array = new double[_size];

		std::vector<size_t> new_dimension;
		for(size_t i=0; i<dst_index.size(); i++) {
			new_dimension.push_back(_dimension.at(dst_index.at(i)));
		}

		_reorder(0, 0, 0, dst_index, new_dimension, new_array);
		delete[] _array;
		std::swap(new_dimension, _dimension);


		_array = new_array;
	}

	template<typename Function>
	void apply(Function f) {
		for(size_t i=0; i<_size; i++)
			_array[i] = f(_array[i]);
	}

	void print(std::ostream& stream) const {
		stream << "NumpyArray[";
		for(size_t i=0; i<_dimension.size(); i++) {
			if(i != 0)
				stream << ", ";
			stream << _dimension.at(i);
		}
		stream << "] (";
		for(size_t i=0; i<_size; i++) {
			if(i != 0)
				stream << ", ";
			stream << _array[i];
		}
		stream << ")";
	}


private:
	template<typename... Tail>
	void _initialize(size_t head, Tail... tail) {
		_dimension.push_back(head);
		_initialize(tail...);
	}

	void _initialize() {
		_size = std::accumulate(std::begin(_dimension), std::end(_dimension), 1, std::multiplies<size_t>()),
		_array = new double[_size];
	}

	template<typename... Index>
	size_t to_index(Index... index) const {
		if(sizeof...(Index) != _dimension.size()) {
			throw std::runtime_error("Bad index");
		}
		return _to_index<0>(index...);
	}

	template<size_t I, typename... TailIndex>
	typename std::enable_if<sizeof...(TailIndex) >= 1, size_t>::type _to_index(size_t head_index, TailIndex... tail_index) const {
		return std::accumulate(std::begin(_dimension)+I+1, std::end(_dimension), 1, std::multiplies<size_t>())*head_index+_to_index<I+1>(tail_index...);
	}

	template<size_t I, typename... TailIndex>
	typename std::enable_if<sizeof...(TailIndex) == 0, size_t>::type _to_index(size_t head_index, TailIndex... tail_index) const {
		return head_index;
	}


	void _reorder(size_t dst_index, size_t src_index, size_t src_dim, const std::vector<size_t>& dst_order, const std::vector<size_t>& dst_dimension, double* dst) const {

		if(src_dim == _dimension.size()) {
			dst[dst_index] = _array[src_index];
		}
		else {
			size_t dst_product = std::accumulate(std::begin(dst_dimension)+std::distance(std::begin(dst_order), std::find(std::begin(dst_order), std::end(dst_order), src_dim))/*dst_order.at(src_dim)*/+1, std::end(dst_dimension), 1, std::multiplies<size_t>());
			size_t  src_product = std::accumulate(std::begin(_dimension)+src_dim+1, std::end(_dimension), 1, std::multiplies<size_t>());

			for(size_t i=0; i<_dimension.at(src_dim); i++) {
				_reorder(dst_index+i*dst_product, src_index+i*src_product, src_dim+1, dst_order, dst_dimension, dst);
			}
		}
	}


	std::vector<size_t> _dimension;
	size_t _size;
	double* _array;
};

class NumpyArchive {

public:
	NumpyArchive() : _list() {

	}

	void add(const std::string& name, const NumpyArray& array) {
		_list.emplace(std::piecewise_construct, std::forward_as_tuple(name), std::forward_as_tuple(array));
	}

	NumpyArray& at(const std::string& name) {
		return _list.at(name);
	}

	const NumpyArray& at(const std::string& name) const {
		return _list.at(name);
	}

	std::map<std::string, NumpyArray>& list() {
		return _list;
	}

	const std::map<std::string, NumpyArray>& list() const {
		return _list;
	}

	void clear() {
		_list.clear();
	}

private:
	std::map<std::string, NumpyArray> _list;
};

#endif
