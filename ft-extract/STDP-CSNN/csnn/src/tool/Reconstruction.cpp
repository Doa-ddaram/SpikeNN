#include "tool/Reconstruction.h"

#ifdef ENABLE_QT
using namespace tool;


void Reconstruction::process(const std::string& output, const AbstractExperiment& experiment, const Layer& layer, size_t oversampling) {
	std::cout << "Saving " << layer.depth() << " to " << output << std::endl;
	for(size_t i=0; i<layer.depth(); i++) {
		Tensor<float> out(Shape({1, 1, layer.depth()}));
		out.fill(0);
		out.at(0, 0, i) = 1.0;

		for(int j=layer.index(); j>=0; j--) {
			if(dynamic_cast<const Layer*>(&experiment.process_at(j))) {
				out = dynamic_cast<const Layer*>(&experiment.process_at(j))->reconstruct(out);
			}
		}

		out.range_normalize();

		size_t width = out.shape().dim(0);
		size_t height = out.shape().dim(1);
		size_t depth = out.shape().dim(2);

		QImage image(width*oversampling, height*oversampling, QImage::Format_ARGB32_Premultiplied);
		QPainter painter(&image);

		for(size_t x=0; x<width; x++) {
			for(size_t y=0; y<height; y++) {
				std::vector<float> values;
				for(size_t z=0; z<depth; z++) {
					values.push_back(out.at(x, y, z));
				}
				painter.fillRect(x*oversampling, y*oversampling, oversampling, oversampling, DefaultColor::get(values));

			}
		}
		std::string output_file(output+std::to_string(i)+".png");
		image.save(QString(output_file.c_str()));
		std::cout << "Save " << output_file << std::endl;

	}
}
#endif
