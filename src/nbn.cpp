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
  layer_index_.push_back(topology_[0]);
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

  // // both for convinience
  neuron_index_.push_back(size);

  int num_input = get_num_input();
  layer_index_.push_back(get_num_neuron() + num_input);

  // update index in lookup table
  int num_neuron = get_num_neuron();
  lookup_ = -1 * Eigen::MatrixXi::Ones(num_neuron + num_input, num_neuron + num_input);
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
  Eigen::MatrixXd Identity = Eigen::MatrixXd::Identity(num_weight, num_weight);

  // Same structure as topology_, so neuron_index_ applies here.
  Eigen::RowVectorXd jacobian(num_weight);
  Eigen::VectorXd gradient(num_weight);

  // Contains output for each neuron, including input
  Eigen::VectorXd output(num_neuron + num_input);

  // Stores the signal gain for backward computation.
  Eigen::MatrixXd delta(num_neuron, num_neuron);

  double last_error = std::numeric_limits<double>::max();
  int fail_count = 0;

  for (int iteration = 0; iteration < max_iteration; ++iteration) {

    double error = 0.0;
    hessian.setZero();
    gradient.setZero();

    for (int ind_pattern = 0; ind_pattern < num_pattern; ++ind_pattern) {

      delta.setZero();
      output.setZero();

      // output of the input neurons
      output.head(num_input) = Eigen::Map<const Eigen::VectorXd>(inputs.data() + ind_pattern * num_input,
                                                                 num_input);

      // Forward computation.
      for (int i = 0; i < num_neuron; ++i) {
        const int beg = neuron_index_[i],     // start postion for current neuron in topology_
                  end = neuron_index_[i + 1], // past-the-end position for current neuron in topology_
                 ind0 = topology_[beg],       // index of current neuron considering input
                 ind1 = ind0 - num_input;     // index of current neuron without considering input

        // calculate output of current neuron
        double net = weight_[beg];
        for (int j = beg + 1; j < end; ++j)
          net += weight_[j] * output[topology_[j]];
        output[ind0] = CALL_MEMBER_FUNC(this, activation_func_[activation_[ind1]])(net, gain_[ind1]);

        double derivative = CALL_MEMBER_FUNC(this, activation_func_d_[activation_[ind1]])
                            (output[ind0], gain_[ind1]);

        // Update delta table.  Don't try to change anything here
        // unless you know what you're doing.
        delta(ind1, ind1) = derivative;
        int cur_layer = get_neuron_layer(ind0);
        for (int j0 = num_input; j0 < layer_index_[cur_layer]; ++j0) {
          int j1 = j0 - num_input;
          int layerj = get_neuron_layer(j0);
          double signal = lookup_(j0, ind0) >= 0 ? weight_[lookup_(j0, ind0)] * delta(j1, j1) : 0.0;
          for (int k = bisearch_ge(layer_index_[layerj + 1],
                                   topology_.data() + beg + 1, end - beg) + beg + 1; k < end; ++k)
            signal += weight_[lookup_(topology_[k], ind0)] * delta(topology_[k] - num_input, j1);
          delta(ind1, j1) = derivative * signal;
        }
      }

      // Backward propagation
      for (int ind_output = 0; ind_output < num_output; ++ind_output) {
        int i0 = output_id_[ind_output],
            i1 = i0 - num_input,
        layeri = get_neuron_layer(i0);
        double e = desired_outputs[ind_pattern * num_output + ind_output] - output[i0];
        jacobian.setZero();

        // Update jacobian for neurons in previous layers
        for (int j = 0; j < layer_index_[layeri] - num_input; ++j) {
          const int beg = neuron_index_[j],     // start postion for current neuron in topology_
                    end = neuron_index_[j + 1]; // past-the-end position for current neuron in topology_

          jacobian[beg] = delta(i1, j);
          for (int k = beg + 1; k < end; ++k)
            jacobian[k] = output[topology_[k]] * delta(i1, j);
        }

        // Update jacobian for inputs to the current output.  This is
        // neccessary because we might have outputs from hidden
        // neurons.
        const int beg = neuron_index_[i1],
                  end = neuron_index_[i1 + 1],
                 ind1 = topology_[beg] - num_input;

        jacobian[beg] = delta(i1, ind1);
        for (int k = beg + 1; k < end; ++k)
          jacobian[k] = output[topology_[k]] * delta(i1, ind1);

        gradient += jacobian.transpose() * e;
        hessian += jacobian.transpose() * jacobian;
        error += e * e;
      }
    }

    if (error > last_error) {
      ++fail_count;
      // if (fail_count > param_.fail_max) return false;

      param_.mu *= param_.scale_up;
      if (param_.mu > param_.mu_max) param_.mu = param_.mu_max;
    } else {

      if (error < max_error) return true;

      param_.mu /= param_.scale_down;
      fail_count = 0;
    }
    last_error = error;

    weight_ -= (hessian + param_.mu * Identity).inverse() * gradient;

    std::cout << error << std::endl;
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

  Eigen::VectorXd output(num_neuron + num_input);
  std::vector<double> ret(num_output * num_pattern);

  for (int ind_pattern = 0; ind_pattern < num_pattern; ++ind_pattern) {
    output.head(num_input) = Eigen::Map<const Eigen::VectorXd>(inputs.data() + ind_pattern * num_input,
                                                               num_input);
    for (int i = 0; i < num_neuron; ++i) {
      const int beg = neuron_index_[i],
                end = neuron_index_[i + 1],
                cur = topology_[beg] - num_input;
      double net = weight_[beg];
      for (int j = beg + 1; j < end; ++j)
        net += weight_[j] * output[topology_[j]];
      output[cur + num_input] = CALL_MEMBER_FUNC(this, activation_func_[activation_[cur]])(net, gain_[cur]);
    }

    for (int i = 0, offset = ind_pattern * num_output; i < num_output; ++i)
      ret[i + offset] = output[output_id_[i]];
  }

  return ret;
}
