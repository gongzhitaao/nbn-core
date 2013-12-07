#include "nbn.h"

#include <iostream>

#define EIGEN_NO_DEBUG

NBN::NBN() :
  num_input_(0),
  num_output_(0),
  num_weight_(0),
  train_func_{&NBN::ebp, &NBN::nbn}
{
}

NBN::~NBN()
{
}

#define CALL_MEMBER_FUNC(object, func_ptr) ((object).*(func_ptr))

void NBN::set_topology(const nbn_index_t &topology)
{
  topology_ = topology;
  neuron_index_.clear();
  layer_index_.clear();

  neuron_index_.push_back(num_input_);
  layer_index_.push_back(topology_[0]);
  int size = topology_.size();
  for (int i = 1,
	 max_last_layer_neuron_id = 2,
	 updated = 0;
       i < size; ++i) {
    if (topology_[i] > topology_[neuron_index_.back() - num_input_]) {
      neuron_index_.push_back(i + num_input_);
      updated = 0;
    } else if (topology_[i] > max_last_layer_neuron_id && !updated) {
      max_last_layer_neuron_id = topology_[neuron_index_[neuron_index_.size() - 2] - num_input_];
      layer_index_.push_back(topology_[neuron_index_.back() - num_input_]);
      updated = 1;
    }
  }

  num_input_ = topology_[0] - 1;
  num_output_ = topology_[neuron_index_.back()] - layer_index_.back() + 1;
  num_weight_ = size - neuron_index_.size();

  neuron_index_.push_back(size); // for convinience
}

bool NBN::train(std::vector<double> inputs, std::vector<double> desired_outputs,
		int max_iteration, double max_error)
{
  return CALL_MEMBER_FUNC(*this, train_func_[training_algorithm_])(max_iteration, max_error);
}

bool NBN::nbn(int max_iteration, double max_error)
{
  return true;
}

bool NBN::ebp(int max_iteration, double max_error)
{
  return true;
}
