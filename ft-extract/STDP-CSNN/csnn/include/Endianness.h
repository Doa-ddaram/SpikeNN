#ifndef _ENDIANNESS_H
#define _ENDIANNESS_H

#include <istream>
#include <algorithm>
#include <type_traits>

template<typename From, typename To>
class DataReinterpret {

public:
    DataReinterpret(const From& from) : _from(from) {
        static_assert(sizeof(From) == sizeof(To), "Different type size");
    }

    From& from() {
        return _from;
    }

    const From& from() const {
        return _from;
    }

    To& to() {
        return _to;
    }

    const To& to() const {
        return _to;
    }

private:
    union {
        From _from;
        To _to;
    };

};

class ByteOrer {

private:
	enum EndiannessType : uint32_t {
		_LITTLE_ENDIAN   = 0x00000001,
		_BIG_ENDIAN      = 0x01000000,
		_PDP_ENDIAN      = 0x00010000
	};

public:
	ByteOrer() = delete;

	static constexpr bool is_big_endian() {
		return (1 & 0xFFFFFFFF) == _BIG_ENDIAN;
	}

	static constexpr bool is_little_endian() {
		return (1 & 0xFFFFFFFF) == _LITTLE_ENDIAN;
	}

	static constexpr bool is_pdp_endian() {
		return (1 & 0xFFFFFFFF) == _PDP_ENDIAN;
	}
};


class SwapBytes {

public:
	SwapBytes() = delete;


	static int8_t swap(int8_t v) {
		return v;
	}

	static uint8_t swap(uint8_t v) {
		return v;
	}

	static int16_t swap(int16_t v) {
		return ((v & 0xFF) << 8) | ((v & 0xFF00) >> 8);
	}

	static uint16_t swap(uint16_t v) {
		return ((v & 0xFF) << 8) | ((v & 0xFF00) >> 8);
	}

	static int32_t swap(int32_t v) {
		return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
	}

	static uint32_t swap(uint32_t v) {
		return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
	}

	static int64_t swap(int64_t v) {
		return ((v & 0xFF) << 56) | ((v & 0xFF00) << 24) | ((v & 0xFF0000) << 8) | ((v & 0xFF000000) >> 8) | ((v & 0xFF00000000) >> 24) | ((v & 0xFF0000000000) >> 56);
	}

	static uint64_t swap(uint64_t v) {
		return ((v & 0xFF) << 56) | ((v & 0xFF00) << 24) | ((v & 0xFF0000) << 8) | ((v & 0xFF000000) >> 8) | ((v & 0xFF00000000) >> 24) | ((v & 0xFF0000000000) >> 56);
    }

    static float swap(float v) {
        DataReinterpret<float, uint32_t> cast(v);
        cast.to() = swap(cast.to());
        return cast.from();
    }

    static double swap(double v) {
        DataReinterpret<double, uint64_t> cast(v);
        cast.to() = swap(cast.to());
        return cast.from();
    }

};

//
//	Endian Types
//
class BigEndian {

public:
	BigEndian() = delete;

	template<typename T>
	static typename std::enable_if<std::is_arithmetic<T>::value && ByteOrer::is_big_endian(), T>::type
	convert(T v) {
		return v;
	}

	template<typename T>
	static typename std::enable_if<std::is_arithmetic<T>::value && ByteOrer::is_little_endian(), T>::type
	convert(T v) {
		return SwapBytes::swap(v);
	}
};

class LittleEndian {

public:
	LittleEndian() = delete;

	template<typename T>
	static typename std::enable_if<std::is_arithmetic<T>::value && ByteOrer::is_big_endian(), T>::type
	convert(T v) {
		return SwapBytes::swap(v);
	}

	template<typename T>
	static typename std::enable_if<std::is_arithmetic<T>::value && ByteOrer::is_little_endian(), T>::type
	convert(T v) {
		return v;
	}
};

template<typename EndianType>
class Endianness {

public:
	Endianness() = delete;

	template<typename T>
	static typename std::enable_if<std::is_trivial<T>::value, T>::type
	read(std::istream& stream) {
		T data;
		stream.read(reinterpret_cast<char*>(&data), sizeof(T));
		return EndianType::convert(data);
	}

private:

};
#endif
