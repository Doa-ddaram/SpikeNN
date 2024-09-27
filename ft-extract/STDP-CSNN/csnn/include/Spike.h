#ifndef _SPIKE_H
#define _SPIKE_H

#include <cstdint>
#include <limits>

typedef float Time;

#define INFINITE_TIME std::numeric_limits<Time>::max()

struct Spike {

	Spike(Time p_time, uint16_t p_x, uint16_t p_y, uint16_t p_z) : x(p_x), y(p_y), z(p_z), time(p_time)  {

	}


	uint8_t x;
	uint8_t y;
	uint16_t z;
	Time time;

};


struct TimeComparator {

	bool operator()(const Spike& e1, const Spike& e2) const {
		return e1.time < e2.time;
	}
};


#endif
