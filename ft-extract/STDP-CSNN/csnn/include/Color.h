#ifndef _COLOR_H
#define _COLOR_H

#ifdef ENABLE_QT
#include <QColor>

#include <cmath>

class HSL {

public:
	static QColor compute(float h, float s, float l, float alpha = 1.0) {
		float c = (1.0-std::abs(2.0*l-1.0))*s;
		float x = c*(1.0-std::abs(std::fmod(h/60.0, 2.0)-1.0));
		float m = l-c/2.0;

		uint8_t cn = static_cast<uint16_t>(c*255.0+m);
		uint8_t xn = static_cast<uint16_t>(x*255.0+m);

		uint8_t a = alpha*255.0;

		if(h < 60)
		  return QColor(cn, xn, 0, a);
		else if(h < 120)
		  return QColor(xn, cn, 0, a);
		else if(h < 180)
		  return QColor(0, cn, xn, a);
		else if(h < 240)
		  return QColor(0, xn, cn, a);
		else if(h < 300)
		  return QColor(xn, 0, cn, a);
		else
		  return QColor(cn, 0, xn, a);
	  }
};

class HeatMap {

public:
	static QColor get(float value, float alpha = 1.0) {
	  if(value < 0)
		return QColor(0, 0, 0, alpha*255);
	  else if(value > 1)
		return QColor(255, 255, 255, alpha*255);
	  else if(value == 0)
		return QColor(0, 0, 255, alpha*255);
	  else
		return HSL::compute((1.0-value)*240.0, 1, 0.5, alpha);
	}

};

class NegPos {

public:
	static QColor get(float v1, float v2) {
	  return QColor(
				  std::max<float>(0, std::min<float>(1, v1))*255,
				  std::max<float>(0, std::min<float>(1, v2))*255,
				  0
				);
	}

};

class RGB {

public:
	static QColor get(float r, float g, float b) {
	  return QColor(
				  std::max<float>(0, std::min<float>(1, r))*255,
				  std::max<float>(0, std::min<float>(1, g))*255,
				  std::max<float>(0, std::min<float>(1, b))*255
				);
	}

};

class HeatMapWithNet {

public:
	static QColor get(float value) {
	  if(value <= -1.0+std::numeric_limits<float>::epsilon())
		return QColor(0, 0, 255);
	  else if(value >= 1-std::numeric_limits<float>::epsilon())
		return QColor(255, 0, 0);
	  else
		return HSL::compute((1-(value*2.0-1.0))*240, 1, 0.5);
	}
};

class GreyScale {

public:
	static QColor get(float value) {
	  if(value <= 0)
		return QColor(0, 0, 0);
	  else if(value >= 1)
		return QColor(255, 255, 255);
	  else
		return QColor(value*255, value*255, value*255);
	}

};

class Heat2 {
public:
	static QColor get(float value) {
		if (value > 1.0) {
			return QColor(255, 255, 255);
		}
		else if (value < 0.0) {
			return QColor(255, 255, 255);
		}
		else if (value < 0.5) {
			return QColor(0, 0, 2*(127 - value*255));
		}
		else {
			return QColor(2* (value*255 - 128), 0, 0);
		}
	}
};

class DefaultColor {

public:
	static QColor get(const std::vector<float>& list) {
		if(list.size() == 1) {
			return GreyScale::get(list[0]);
		}
		else if(list.size() == 2) {
			return NegPos::get(list[0], list[1]);
		}
		else if(list.size() == 3) {
			return RGB::get(list[0], list[1], list[2]);
		}
		else {
			size_t max_i = 0;
			float max_v = 0;

			for(size_t i=0; i<list.size(); i++) {
				if(max_v < list[i]) {
					max_i = i;
					max_v = list[i];
				}
			}

			if(max_v == 0) {
				return QColor(0, 0, 0, max_v);
			}
			else {
				return HeatMap::get(static_cast<float>(max_i)/static_cast<float>(list.size()-1), max_v);
			}
		}
	}

};
#endif

#endif
