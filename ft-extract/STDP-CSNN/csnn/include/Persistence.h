#ifndef _PERSISTENCE_H
#define _PERSISTENCE_H

#include <cstdint>
#include <sstream>

#include "Tensor.h"

typedef uint8_t PersistenceType;

#define _PERSISTENCE_INT8	0x01
#define _PERSISTENCE_UINT8	0x02
#define _PERSISTENCE_INT16	0x03
#define _PERSISTENCE_UINT16 0x04
#define _PERSISTENCE_INT32	0x05
#define _PERSISTENCE_UINT32 0x06
#define _PERSISTENCE_INT64	0x07
#define _PERSISTENCE_UINT64 0x08
#define _PERSISTENCE_FLOAT	0x09
#define _PERSISTENCE_DOUBLE 0x0A
#define _PERSISTENCE_BOOL 0x0B

class Persistence {

public:
	Persistence() = delete;

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, int8_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_INT8;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, uint8_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_UINT8;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, int16_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_INT16;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, uint16_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_UINT16;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, int32_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_INT32;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, uint32_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_UINT32;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, int64_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_INT64;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, uint64_t>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_UINT64;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, float>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_FLOAT;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, double>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_DOUBLE;
	}

	template<typename T>
	static constexpr typename std::enable_if<std::is_same<T, bool>::value, PersistenceType>::type to_indetifier() {
		return _PERSISTENCE_BOOL;
	}


	template<template<typename> class T, typename Ret, typename... Args>
	static Ret call(PersistenceType type, Args&&... args) {
		switch(type) {

		case _PERSISTENCE_INT8:
			return T<int8_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_UINT8:
			return T<uint8_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_INT16:
			return T<int16_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_UINT16:
			return T<uint16_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_INT32:
			return T<int32_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_UINT32:
			return T<uint32_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_INT64:
			return T<int64_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_UINT64:
			return T<uint64_t>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_FLOAT:
			return T<float>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_DOUBLE:
			return T<double>::apply(std::forward<Args>(args)...);
		case _PERSISTENCE_BOOL:
			return T<bool>::apply(std::forward<Args>(args)...);
		default:
			throw std::runtime_error("Unkown type "+std::to_string(static_cast<size_t>(type)));
		}
	}

	static void save_string(const std::string& str, std::ostream& stream) {
		uint32_t size = str.size();
		stream.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
		stream.write(str.c_str(), size);
	}

	static std::string load_string(std::istream& stream) {
		uint32_t size = 0;
		stream.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));

		std::string str(size, 0);
		stream.read(reinterpret_cast<char*>(&str[0]), size);
		return str;
	}

	template<typename T>
	static void save_tensor(const Tensor<T>& tensor, std::ostream& stream) {
		PersistenceType type = to_indetifier<T>() | 0x80;
		stream.write(reinterpret_cast<const char*>(&type), sizeof(PersistenceType));
		uint32_t number_dim = tensor.shape().number();
		stream.write(reinterpret_cast<const char*>(&number_dim), sizeof(uint32_t));
		for(size_t i=0; i<number_dim; i++) {
			uint32_t dim = tensor.shape().dim(i);
			stream.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
		}

		stream.write(reinterpret_cast<const char*>(tensor.begin()), tensor.shape().product()*sizeof(T));
	}

	template<typename T>
	static Tensor<T> load_tensor(std::istream& stream) {
		PersistenceType type;
		stream.read(reinterpret_cast<char*>(&type), sizeof(PersistenceType));

		if(type != (Persistence::to_indetifier<T>() | 0x80)) {
			throw std::runtime_error("Incompatible variable type");
		}

		uint32_t number_dim;
		stream.read(reinterpret_cast<char*>(&number_dim), sizeof(uint32_t));

		std::vector<size_t> dims;
		for(size_t i=0; i<number_dim; i++) {
			uint32_t dim;
			stream.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
			dims.push_back(dim);
		}

		Shape s(dims);
		Tensor<T> value(s);
		stream.read(reinterpret_cast<char*>(value.begin()), s.product()*sizeof(T));

		return value;
	}
};

#endif
