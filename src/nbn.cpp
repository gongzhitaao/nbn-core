#include "nbn.h"

#include <algorithm>
#include <limits>

std::mt19937 NBN::gen_;

void NBN::set_topology(const std::vector<int> &topology, const std::vector<int> &output)
{
  // Topology in config files are 1-based, while in program, 0-based
  // are more convenient.
  topology_.resize(topology.size());
  std::transform(topology.begin(), topology.end(), topology_.begin(),
                 [](int i) -> int { return i - 1; });

  neuron_index_.clear();
  layer_index_.clear();

  // Parse the topology vector to get general architecture information
  // of the neural network, number of inputs, number of outputs, layer
  // sizes and etc.
  neuron_index_.push_back(0);
  layer_index_.push_back(0);
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
  int num_neuron = get_num_neuron();

  // Most of the cases, outputs appear in the last layer, i.e., output
  // layers.  But in rare cases, we might want outputs from hidden
  // layers.
  output_id_.clear();
  if (output.size() > 0) {
      // 1-based to 0-based
      output_id_.resize(output.size());
      std::transform(output.begin(), output.end(), output_id_.begin(),
                     [](int i) { return i - 1; });
  } else {
      for (int i = layer_index_.back(); i < num_neuron + num_input; ++i)
          output_id_.push_back(i);
  }

  layer_index_.push_back(get_num_neuron() + num_input);

  if (output.size() > 0) {
      // 1-based to 0-based
      output_id_.resize(output.size());
      std::transform(output.begin(), output.end(), output_id_.begin(),
                     [](int i) { return i - 1; });
  } else {
      int num_neuron = get_num_neuron();
      for (int i = layer_index_.back(); i < num_neuron + num_input; ++i)
          output_id_.push_back(i);
  }

  // update index in lookup table
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
  Timer timer;

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

  Eigen::VectorXd weight_backup(num_weight);

  errors_.clear();
  errors_.push_back(calcerror(desired_outputs, run(inputs)));

  for (int iteration = 0; iteration < max_iteration; ++iteration) {

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
                  ind = topology_[beg];       // index of current neuron considering input

        // calculate output of current neuron
        double net = weight_[beg];
        for (int j = beg + 1; j < end; ++j)
          net += weight_[j] * output[topology_[j]];
        output[ind] = activation_func_[activation_[i]](net, gain_[i]);

        double derivative = activation_func_d_[activation_[i]](output[ind], gain_[i]);

        // Update delta table.
        delta(i, i) = derivative;
        for (int j0 = num_input; j0 < ind; ++j0) {
          int j1 = j0 - num_input;
          double signal = lookup_(j0, ind) >= 0 ? weight_[lookup_(j0, ind)] * delta(j1, j1) : 0.0;
          for (int k = beg + 1; k < end; ++k) {
            if (topology_[k] > j0)
              signal += weight_[lookup_(topology_[k], ind)] * delta(topology_[k] - num_input, j1);
          }
          delta(i, j1) = derivative * signal;
        }
      }

      // Backward propagation
      for (int ind_output = 0; ind_output < num_output; ++ind_output) {
        int i0 = output_id_[ind_output],
            i1 = i0 - num_input,
        layeri = get_neuron_layer(i0 + 1);
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
      }
    }

    weight_backup = weight_;
    int fail_count = 0;
    while (true) {
      // update weight
      weight_ = weight_backup + (hessian + param_.mu * Identity).inverse() * gradient;

      // calculate error again
      double error = calcerror(desired_outputs, run(inputs)),
        last_error = errors_.back();

      if (error < last_error) {
        if (param_.mu > param_.mu_min)
          param_.mu *= param_.scale_down;
        errors_.push_back(error);
        break;
      } else {
        ++fail_count;
        if (fail_count > param_.fail_max) {
          errors_.push_back(error);
          break;
        }
        if (param_.mu < param_.mu_max)
          param_.mu *= param_.scale_up;
      }
    }

    if (errors_.back() < max_error)
      break;
  }

  weightcopy_.clear();
  weightcopy_.insert(weightcopy_.end(), weight_.data(), weight_.data() + weight_.size());

  elapsed_ = timer.elapsed();

  return errors_.back() < max_error;
}

std::vector<double> NBN::run(const std::vector<double> &inputs) const
{
  int num_input = get_num_input();
  int num_neuron = get_num_neuron();
  int num_output = get_num_output();
  int num_pattern = inputs.size() / num_input;

  Eigen::VectorXd output(num_neuron + num_input);
  std::vector<double> ret(num_output * num_pattern);

  for (int ind_pattern = 0; ind_pattern < num_pattern; ++ind_pattern) {
    output.head(num_input) = Eigen::Map<const Eigen::VectorXd>(inputs.data() + ind_pattern * num_input,
                                                               num_input);

    for (int i = 0; i < num_neuron; ++i) {
      const int beg = neuron_index_[i],
                end = neuron_index_[i + 1];
      double net = weight_[beg];
      for (int j = beg + 1; j < end; ++j)
        net += weight_[j] * output[topology_[j]];
      output[i + num_input] = activation_func_[activation_[i]](net, gain_[i]);
    }

    for (int i = 0, offset = ind_pattern * num_output; i < num_output; ++i)
      ret[i + offset] = output[output_id_[i]];
  }

  return ret;
}
