#include "monitor/SaveNetwork.h"
#include "Experiment.h"

using namespace monitor;

static RegisterClassParameter<SaveNetwork, MonitorFactory> _register("SaveNetwork");

SaveNetwork::SaveNetwork() : Monitor(_register)  {

}



void SaveNetwork::on_sample(const AbstractExperiment&, size_t, size_t) {

}

void SaveNetwork::on_epoch(const AbstractExperiment& experiment, size_t layer_index, size_t epoch_count) {
	experiment.save(experiment.name()+"-layer-"+std::to_string(layer_index)+"-epoch-"+std::to_string(epoch_count));
}
