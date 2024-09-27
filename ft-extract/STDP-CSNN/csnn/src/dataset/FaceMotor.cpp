#include "dataset/FaceMotor.h"

#ifdef ENABLE_QT
using namespace dataset;

FaceMotor::FaceMotor(const std::string& learn_dir) :
	_dir1(learn_dir), _dir2(), _cursor(0), _shape({FACE_MOTOR_WIDTH, FACE_MOTOR_HEIGHT, FACE_MOTOR_DEPTH}), _data() {
	QDir dir(learn_dir.c_str());
	QStringList learn_images = dir.entryList(QStringList() << "*.jpg" << "*.JPG" << "*.bmp", QDir::Files);
	for(int i=0; i<learn_images.count();  i++) {
		_read(QString("%1/%2").arg(dir.absolutePath()).arg(learn_images.at(i)), "unk");
	}

}

FaceMotor::FaceMotor(const std::string& face_dir, const std::string& motor_dir) :
	_dir1(face_dir), _dir2(motor_dir), _cursor(0), _shape({FACE_MOTOR_WIDTH, FACE_MOTOR_HEIGHT, FACE_MOTOR_DEPTH}), _data()  {
	QDir dir1(face_dir.c_str());
	QStringList face_images = dir1.entryList(QStringList() << "*.jpg" << "*.JPG" << "*.bmp", QDir::Files);
	for(int i=0; i<face_images.count();  i++) {
		_read(QString("%1/%2").arg(dir1.absolutePath()).arg(face_images.at(i)), "1");
	}

	QDir dir2(motor_dir.c_str());
	QStringList motor_images = dir2.entryList(QStringList() << "*.jpg" << "*.JPG" << "*.bmp", QDir::Files);
	for(int i=0; i<motor_images.count();  i++) {
		_read(QString("%1/%2").arg(dir2.absolutePath()).arg(motor_images.at(i)), "2");
	}
}

bool FaceMotor::has_next() const {
	return _cursor < _data.size();
}

std::pair<std::string, Tensor<InputType>> FaceMotor::next() {
	return _data.at(_cursor++);
}

void FaceMotor::reset() {
	_cursor = 0;
}

void FaceMotor::close() {
	_data.clear();
}


size_t FaceMotor::size() const {
	return _data.size();
}
std::string FaceMotor::to_string() const {
	if(_dir2.empty()) {
		return "FaceMotor(learn_set: "+_dir1+")";
	}
	else {
		return "FaceMotor(face: "+_dir1+", motor: "+_dir2+")";
	}
}

const Shape& FaceMotor::shape() const {
	return _shape;
}

void FaceMotor::_read(const QString& filename, const std::string& label) {
	QImage image(filename);
	image = image.scaled(FACE_MOTOR_WIDTH, FACE_MOTOR_HEIGHT, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

	_data.emplace_back(label, _shape);

	for(int x=0; x<FACE_MOTOR_WIDTH; x++) {
		for(int y=0; y<FACE_MOTOR_HEIGHT; y++) {
			QRgb pixel = image.pixel(x, y);
			_data.back().second.at(x, y, 0) = static_cast<InputType>(qBlue(pixel));
		}
	}
}
#endif
