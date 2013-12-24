#include "nbn.h"

#include <algorithm>
#include <limits>
#include <iostream>

void NBN::set_topology(const std::vector<int> &topology, const std::vector<int> &output)
{
  // Topology in config files are 1-based, while in program, 0-based
  // are more convenient.
  topology_.resize(topology.size());
  std::transform(topology.begin(), topology.end(), topology_.begin(),
                 [](int i) -> int { return i - 1; });

  // 1-based to 0-based
  output_id_.resize(output.size());
  std::transform(output.begin(), output.end(), output_id_.begin(),
                 [](int i) -> int { return i - 1; });

  neuron_index_.clear();
  layer_index_.clear();

  neuron_index_.push_back(0);
  layer_index_.push_back(0);
  int size = topology_.size();
  for (int i = 1, updated = 0; i < size; ++i) {
    if (topology_[i] > topology_[neuron_index_.back()]) {
      // sort inputs to each neuron ascendingly
      std::sort(&topology_[neuron_index_.back() + 1], &topology_[i]);
      neuron_index_.push_back(i);
      updated = 0;
    } else if (topology_[i] >= layer_index_.back() && !updated) {
      layer_index_.push_back(topology_[neuron_index_.back()]);
      updated = 1;
    }
  }
  // sort last neuron inputs
  std::sort(topology_.begin() + neuron_index_.back() + 1, topology_.end());

  for (unsigned i = 0; i < topology_.size(); ++i)
    std::cout << topology_[i] << ' ';
  std::cout << std::endl;

  // both for convinience
  neuron_index_.push_back(size);
  layer_index_.push_back(get_num_neuron());

  // update index in lookup table
  int num_neuron = get_num_neuron();
  lookup_ = -1 * Eigen::MatrixXi::Ones(num_neuron, num_neuron);
  for (int i = 0, j = 0; i < size; ++i) {
    int neuron = topology_[i];
    if (neuron > j) j = neuron;
    lookup_(neuron, j) = i;
  }
}

bool NBN::ebp(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
              int max_iteration, double max_error)
{
  return true;
}

bool NBN::nbn(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
              int max_iteration, double max_error)
{
  int num_input = get_num_input();
  int num_output = get_num_output();
  int num_neuron = get_num_neuron();
  int num_weight = get_num_weight();
  int num_pattern = inputs.size() / num_input;

  Eigen::MatrixXd hessian(num_weight, num_weight);

  // Same structure as topology_, so neuron_index_ applies here.
  Eigen::RowVectorXd jacobian(num_weight);
  Eigen::VectorXd gradient(num_weight);

  // Contains output for each neuron, including input neurons
  Eigen::VectorXd output(num_neuron);

  // Stores the signal gain for backward computation.  This vector has
  // same structurs as weight_.
  Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(num_neuron, num_neuron);

  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(num_weight, num_weight);
  double last_error = std::numeric_limits<double>::max();
  int fail_count = 0;

  for (int iteration = 0; iteration < max_iteration; ++iteration) {

    double error = 0.0;
    for (int ind_pattern = 0; ind_pattern < num_pattern; ++ind_pattern) {

      // output of the input neurons
      output.head(num_input) = Eigen::Map<const Eigen::VectorXd>(inputs.data() + ind_pattern * num_input,
                                                                 num_input);

      // Forward computation.
      for (int i = 0; i < num_neuron - num_input; ++i) {
        const int beg = neuron_index_[i],
                  end = neuron_index_[i + 1],
                  cur = topology_[beg];

        // gather inputs
        double net = weight_[beg];
        for (int j = beg + 1; j < end; ++j)
          net += weight_[j] * output[topology_[j]];
        output[cur] = CALL_MEMBER_FUNC(this, activation_func_[activation_[cur]])(net, gain_[cur]);

        // update signal gain
        double derivative = CALL_MEMBER_FUNC(this, activation_func_d_[activation_[cur]])
                            (output[cur], gain_[cur]);

        delta(cur, cur) = derivative;
        int layer = get_neuron_layer(cur);
        for (int j = 0; j < layer_index_[layer]; ++j) {
          int layerj = get_neuron_layer(j);
          double signal = 0.0;
          for (int k = beg + 1; k < end; ++k)
            if (topology_[k] > layer_index_[layerj])
              signal += weight_[lookup_(topology_[k], cur)] * delta(topology_[k], j);
          delta(cur, j) = derivative * signal;
          std::cout << cur << ' ' << j << std::endl;
        }
      }

      // Backward propagation
      for (int ind_output = 0; ind_output < num_output; ++ind_output) {
        int i = output_id_[ind_output];
        int layeri = get_neuron_layer(i);
        double e = desired_outputs[ind_output] - output[i];
        jacobian.setZero();
        for (int j = 0; j < layer_index_[layeri]; ++j)
          jacobian[lookup_(j, i)] = output[j] * delta(i, j);

        gradient += jacobian.transpose() * e;
        hessian += jacobian.transpose() * jacobian;
        error += e * e;
      }
    }

    // std::cout << delta << std::endl;
    // std::cout << output.transpose() << std::endl;
    // std::cout << gradient.transpose() << std::endl;
    // std::cout << jacobian << std::endl;

    // if (error > last_error) {
    //   ++fail_count;
    //   param_.mu *= param_.scale_up;
    //   if (param_.mu > param_.mu_max) param_.mu = param_.mu_max;
    // } else {

    //   if (error < max_error) return true;

    //   param_.mu /= param_.scale_down;
    //   if (param_.mu < param_.mu_min) param_.mu = param_.mu_min;

    //   fail_count = 0;
    //   w = weight_;
    // }

    // if (fail_count > param_.fail_max) weight_ = w;
    // else weight_ -= (hessian + param_.mu * Eigen::MatrixXd::Identity(num_weight, num_weight)).inverse() * gradient;
    weight_ -= (hessian + param_.mu * Eigen::MatrixXd::Identity(num_weight, num_weight)).inverse() * gradient;

    last_error = error;
  }

  return false;
}

std::vector<double> NBN::run(const std::vector<double> &inputs)
{
  int num_input = get_num_input();
  int num_neuron = get_num_neuron();
  int num_weight = get_num_weight();
  int num_output = get_num_output();
  int num_pattern = inputs.size() / num_input;

  Eigen::VectorXd output(num_neuron);
  std::vector<double> ret(num_output * num_pattern);

  for (int ind_pattern = 0; ind_pattern < num_pattern; ++ind_pattern) {
    output.head(num_input) = Eigen::Map<const Eigen::VectorXd>(inputs.data() + ind_pattern * num_input,
                                                               num_input);

    for (int i = 0; i < num_neuron - num_input; ++i) {
      const int beg = neuron_index_[i],
                end = neuron_index_[i + 1],
                cur = topology_[beg];
      double net = weight_[beg];
      for (int j = beg + 1; j < end; ++j)
        net += weight_[j] * output[topology_[j]];
      output[cur] = CALL_MEMBER_FUNC(this, activation_func_[activation_[cur]])(net, gain_[cur]);
    }

    for (int i = 0, offset = ind_pattern * num_output; i < num_output; ++i)
      ret[i + offset] = output[output_id_[i]];
  }

  return ret;
}
