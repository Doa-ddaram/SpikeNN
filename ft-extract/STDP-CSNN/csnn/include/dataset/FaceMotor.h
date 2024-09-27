#ifndef _DATASET_FACE_MOTOR_H
#define _DATASET_FACE_MOTOR_H

#ifdef ENABLE_QT
#include <QDir>
#include <QImage>

#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>

#include "Tensor.h"
#include "Input.h"

#define FACE_MOTOR_WIDTH 250
#define FACE_MOTOR_HEIGHT 160
#define FACE_MOTOR_DEPTH 1

namespace dataset {

	class FaceMotor : public Input {

	public:
		FaceMotor(const std::string& learn_dir);
		FaceMotor(const std::string& face_dir, const std::string& motor_dir);

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();


		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape& shape() const;

	private:
		void _read(const QString& filename, const std::string& label);

		std::string _dir1;
		std::string _dir2;

		uint32_t _cursor;

		Shape _shape;

		std::vector<std::pair<std::string, Tensor<InputType>>> _data;
	};

}
#endif

#endif
