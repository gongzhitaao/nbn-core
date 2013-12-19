#ifndef NBN_H_
#define NBN_H_

#include <cmath>
#include <vector>

#include <Eigen/Dense>

class NBN
{
  typedef bool (NBN::*nbn_train_func_t) (const std::vector<double> &, const std::vector<double> &,
                                         int, double);
  typedef double (NBN::*nbn_activation_func_t) (double, double);

 public:

  enum nbn_train_enum {
    NBN_EBP = 0,		// standard error-backpropagation
    NBN_NBN,			// NBN algorithm
    NBN_TRAIN_ENUM
  };

  enum nbn_activation_func_enum {
    NBN_LINEAR = 0,
    NBN_THRESHOLD,
    NBN_THRESHOLD_SYMMETRIC,
    NBN_SIGMOID,
    NBN_SIGMOID_SYMMETRIC,
    NBN_ACTIVATION_ENUM
  };

  NBN()
      : activation_func_{&NBN::linear, &NBN::threshold, &NBN::threshold_s, &NBN::sigmoid, &NBN::tanh}
      , activation_func_d_{&NBN::linear_d, &NBN::threshold, &NBN::threshold_s, &NBN::sigmoid_d, &NBN::tanh_d}
      , train_func_{&NBN::ebp, &NBN::nbn} {}
  ~NBN() {}

  bool set_training_algorithm(nbn_train_enum training_algorithm);

  nbn_train_enum get_training_algorithm() const {
    return training_algorithm_;
  }

  void set_topology(const std::vector<int> &topology, const std::vector<int> &output);

  int get_num_input() const {
    return topology_[0];
  }

  // Refer to comment of OUTPUT_ID_
  int get_num_output() const {
    return output_id_.size();
  }

  int get_num_layer() const {
    return layer_index_.size() - 1;
  }

  // Number of weights, *including* the bias
  int get_num_weight() const {
    return topology_.size();
  }

  // Number of connections, *excluding* the bias
  int get_num_connection() const {
    return topology_.size() - neuron_index_.size() + 1;
  }

  // Number of neurons, *including* the input neurons.
  int get_num_neuron() const {
    return neuron_index_.size() - 1 + get_num_input();
  }

  std::vector<int> get_layer_size() const {
    std::vector<int> layer_array;
    for (unsigned i = 0; i < layer_index_.size() - 1; ++i)
      layer_array.push_back(layer_index_[i+1] - layer_index_[i]);
    return layer_array;
  }

  bool train(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
             int max_iteration, double max_error);

  bool run(const std::vector<double> input);

 private:

  // The func_d is the derivative of func.  The func_s is the
  // symmetric version of func.  And note for sigmoid and tanh, the
  // derivative is based on output of sigmoid and tanh respectively.

  double linear(double x, double k) { return k * x; }
  double linear_d(double x, double k) { return k; }

  // They are not supposed to be used for training, so no derivatives
  // are defined here.
  double threshold(double x, double k) { return x < 0 ? 0 : 1; }
  double threshold_s(double x, double k) { return x < 0 ? -1 : 1; }

  double sigmoid(double x, double k) { return 1 / (1 + std::exp(-k * x)); }
  double sigmoid_d(double y, double k) { return k * y * (1 - y); }

  double tanh(double x, double k) { return std::tanh(k * x); }
  double tanh_d(double y, double k) { return k * (1 - y * y); }

  bool ebp(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
           int max_iteration, double max_error);
  bool nbn(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
           int max_iteration, double max_error);

  // A long vector that contains the topology of the whole network.
  // Each neuron id is followed by its input.  Assumption is that the
  // neurons in the network are number from left to right, i.e., layer
  // by layer, and from top to bottom in the same layer.
  std::vector<int> topology_;

  // A helper index that records the start position for each neuron's
  // definition in TOPOLOGY_.
  std::vector<int> neuron_index_;

  // A helper index that records the start positon of each layer in
  // the neuron id sequence 0, 1, 2, ..., num_neuron - 1.
  std::vector<int> layer_index_;

  // A helper index that records the neuron id that generate
  // outputs. Most of the time, the output neurons appear in the last
  // layer, i.e., output layer.  But there are rares cases where we
  // have output from hidden layers.
  std::vector<int> output_id_;

  // Gain for each neuron, usually it's all 1's.
  Eigen::VectorXd gain_;

  // Weight for each connection
  Eigen::VectorXd weight_;

  // Activation function for each neuron, for input layer, it's always
  // linear.  Other layers are different based on user settings.
  std::vector<nbn_activation_func_enum> activation_;
  const nbn_activation_func_t activation_func_[NBN_ACTIVATION_ENUM];
  const nbn_activation_func_t activation_func_d_[NBN_ACTIVATION_ENUM];

  nbn_train_enum training_algorithm_;
  const nbn_train_func_t train_func_[NBN_TRAIN_ENUM];
};

#endif	// NBN_H_
