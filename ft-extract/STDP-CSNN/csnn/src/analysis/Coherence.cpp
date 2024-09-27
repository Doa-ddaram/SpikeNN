#include "analysis/Coherence.h"
#include "Experiment.h"

using namespace analysis;

static RegisterClassParameter<Coherence, AnalysisFactory> _register("Coherence");


Coherence::Coherence() : NoPassAnalysis(_register) {

}

void Coherence::resize(const Shape&) {

}

void Coherence::process() {
	experiment().log() << "===Coherence===" << std::endl;

	if(experiment().process_at(layer_index()).has_parameter("w") && experiment().process_at(layer_index()).is_type<Tensor<float>>("w")) {
		const Tensor<float>& w = experiment().process_at(layer_index()).parameter<Tensor<float>>("w").get();

		if(w.shape().number() == 4) {


			size_t width = w.shape().dim(0);
			size_t height = w.shape().dim(1);
			size_t depth = w.shape().dim(2);
			size_t n = w.shape().dim(3);

			std::vector<float> list;
			
			for(size_t i=0; i<n; i++) {
				for(size_t j=i+1; j<n;j++) {
					float value = 0;
					float ni = 0;
					float nj = 0;

					for(size_t x=0; x<width; x++) {
						for(size_t y=0; y<height; y++) {
							for(size_t z=0; z<depth; z++) {
								value += w.at(x, y, z, i)*w.at(x, y, z, j);
								ni += w.at(x, y, z, i)*w.at(x, y, z, i);
								nj += w.at(x, y, z, j)*w.at(x, y, z, j);
							}
						}
					}

					list.push_back(value/(std::numeric_limits<float>::epsilon()+std::sqrt(ni)*std::sqrt(nj)));
				}
			}

			// Mean weights
			float mean_w = 0;
			for(size_t i=0; i<n; i++) {
				for(size_t x=0; x<width; x++) {
					for(size_t y=0; y<height; y++) {
						for(size_t z=0; z<depth; z++) {
							mean_w += w.at(x, y, z, i);
						}
					}
				}
			}
			mean_w = mean_w / w.shape().product();


			std::sort(std::begin(list), std::end(list));

			experiment().log() << "Mean weights: " << mean_w << std::endl;
			experiment().log() << "------" << std::endl;
			experiment().log() << "N: " << list.size() << std::endl;
			experiment().log() << "Min: " << list.front() << std::endl;
			experiment().log() << "Q1: " << list.at(std::min(list.size()-1, (list.size()*1)/4)) << std::endl;
			experiment().log() << "Q2: " << list.at(std::min(list.size()-1, (list.size()*2)/4)) << std::endl;
			experiment().log() << "Q3: " << list.at(std::min(list.size()-1, (list.size()*3)/4)) << std::endl;
			experiment().log() << "Max: " << list.back() << std::endl;
		}
		else {
			experiment().log() << "Incompatible w shape." << std::endl;
		}

	}
	else {
		experiment().log() << "No w parameter in this layer." << std::endl;
	}

	experiment().log() << std::endl;
}
