#include "nbn.h"

#include <algorithm>
#include <iostream>

#define EIGEN_NO_DEBUG
#define CALL_MEMBER_FUNC(object_ptr, func_ptr) ((object_ptr)->*(func_ptr))

void NBN::set_topology(const std::vector<int> &topology, const std::vector<int> &output)
{
  // Topology in config files are 1-based, while in program, 0-based
  // are more convenient.
  topology_.reserve(topology.size());
  std::transform(topology.begin(), topology.end(), topology_.begin(),
                 [](int i) -> int { return i - 1; });

  // 1-based to 0-based
  output_id_.reserve(output.size());
  std::transform(output.begin(), output.end(), output_id_.begin(),
                 [](int i) -> int { return i - 1; });

  neuron_index_.clear();
  layer_index_.clear();

  neuron_index_.push_back(0);
  layer_index_.push_back(topology_[0]);
  int size = topology_.size();

  for (int i = 1, j = 2, updated = 0; i < size; ++i) {
    if (topology_[i] > topology_[neuron_index_.back()]) {
      neuron_index_.push_back(i);
      updated = 0;
    } else if (topology_[i] > j && !updated) {
      j = topology_[neuron_index_[neuron_index_.size() - 2]];
      layer_index_.push_back(topology_[neuron_index_.back()]);
      updated = 1;
    }
  }

  // both for convinience
  neuron_index_.push_back(size);
  layer_index_.push_back(get_num_neuron());
}

bool NBN::train(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
                int max_iteration, double max_error)
{
  return CALL_MEMBER_FUNC(this, train_func_[training_algorithm_])(
      inputs, desired_outputs, max_iteration, max_error);
}

bool NBN::ebp(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
              int max_iteration, double max_error)
{
  return true;
}

bool NBN::nbn(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
              int max_iteration, double max_error)
{
  return true;
}
