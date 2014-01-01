#ifndef NBN_H_
#define NBN_H_

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

class NBN
{
#define CALL_MEMBER_FUNC(object_ptr, func_ptr) ((object_ptr)->*(func_ptr))

  typedef double (NBN::*nbn_activation_func_t) (double, double) const;
  typedef bool (NBN::*nbn_train_func_t) (const std::vector<double> &, const std::vector<double> &, int, double);
  typedef std::vector<double> (NBN::*nbn_run_func_t) (const std::vector<double> &) const;

 public:

  enum nbn_train_enum {
    NBN_EBP = 0,  // standard error-backpropagation
    NBN_NBN,      // NBN(revised LM) algorithm
    NBN_TRAIN_ENUM
  };

  enum nbn_topology_enum {
    NBN_MLP = 0,  // multilayer perceptron
    NBN_ACN,      // arbitrary connected network, including bridged-MLP and FCC
    NBN_TOPOLOGY_ENUM
  };

  enum nbn_activation_func_enum {
    NBN_LINEAR = 0,
    NBN_THRESHOLD,
    NBN_THRESHOLD_SYMMETRIC,
    NBN_SIGMOID,
    NBN_SIGMOID_SYMMETRIC,
    NBN_ACTIVATION_ENUM
  };

  struct nbn_param {
    double alpha;
    double mu;
    double mu_min, mu_max;
    double scale_down, scale_up;
    int fail_max;
  };

  NBN()
      : activation_func_{&NBN::linear, &NBN::threshold, &NBN::threshold_s, &NBN::sigmoid, &NBN::tanh}
      , activation_func_d_{&NBN::linear_d, &NBN::threshold, &NBN::threshold_s, &NBN::sigmoid_d, &NBN::tanh_d}
      , train_func_{{&NBN::mlp_ebp, &NBN::mlp_nbn}, {&NBN::acn_ebp, &NBN::acn_nbn}}
      , run_func_{&NBN::run_mlp, &NBN::run_acn} {}
  ~NBN() {}

  bool set_training_algorithm(nbn_train_enum training_algorithm);

  nbn_train_enum get_training_algorithm() const { return training_algorithm_; }

<<<<<<< Updated upstream
  void set_topology(const std::vector<int> &topology, const std::vector<int> &output);
  const std::vector<int> &get_topology() const { return topology_; }
=======
  void set_topology(const std::vector<int> &topology,
                    const std::vector<int> &output = std::vector<int>());
>>>>>>> Stashed changes

  void set_gains(const std::vector<double> &gains) { gain_ = gains; }
  const std::vector<double> &get_gains() const { return gain_; }

  void set_weight(const std::vector<double> &weight) { weight_ = Eigen::Map<const Eigen::VectorXd>(weight.data(),weight.size()); }
  const std::vector<double> &get_weight() const { return weightcopy_; }

  void set_activations(const std::vector<nbn_activation_func_enum> &activations) { activation_ = activations; }
  const std::vector<nbn_activation_func_enum> &get_activations() const { return activation_; }

  void set_algorithm(nbn_train_enum training_algorithm) { training_algorithm_ = training_algorithm; }
  nbn_train_enum get_algorithm() const { return training_algorithm_; }

  void set_learning_const(double alpha) { param_.alpha = alpha; }
  double get_learning_const() const { return param_.alpha; }

  void set_mu(double mu) { param_.mu = mu; }
  double get_mu() const { return param_.mu; }

  void set_mu_min(double mu_min) { param_.mu_min = mu_min; }
  double get_mu_min() const { return param_.mu_min; }

  void set_mu_max(double mu_max) { param_.mu_max = mu_max; }
  double get_mu_max() const { return param_.mu_max; }

  void set_scale_up(double scale_up) { param_.scale_up = scale_up; }
  double get_scale_up() const { return param_.scale_up; }

  void set_scale_down(double scale_down) { param_.scale_down = scale_down; }
  double get_scale_down() const { return param_.scale_down; }

  void init_default() {
    int num_weight = get_num_weight();
    int num_neuron = get_num_neuron();

    // default activation, bipolar
    activation_.resize(num_neuron);
    std::fill(activation_.begin(), activation_.end(), NBN_SIGMOID_SYMMETRIC);

    // random initialized weights
    weight_ = Eigen::VectorXd::Ones(num_weight);

    // gain will usually be all 1's.
    gain_.resize(num_neuron);
    std::fill(gain_.begin(), gain_.end(), 1.0);

    // nbn parameters
    param_.mu = 0.01;
    param_.mu_min = 1e-15;
    param_.mu_max = 1e15;
    param_.scale_up = 10;
    param_.scale_down = 0.1;
    param_.fail_max = 10;

    // default using NBN algorithm
    training_algorithm_ = NBN_NBN;

    topology_type_ = NBN_ACN;
  }

  int get_num_input() const {
    return topology_[0];
  }

  // Refer to comment of OUTPUT_ID_
  int get_num_output() const {
    return output_id_.size();
  }

  int get_num_layer() const {
    return layer_index_.size() - 2;
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
    return neuron_index_.size() - 1;
  }

  std::vector<int> get_layer_size() const {
    std::vector<int> layer_array;
    for (unsigned i = 1; i < layer_index_.size() - 1; ++i)
      layer_array.push_back(layer_index_[i+1] - layer_index_[i]);
    return layer_array;
  }

  // Get the layer number the neuron i is in.  Both are 1-based.
  int get_neuron_layer(int i) const {
    return bisearch_le(i - 1, layer_index_.data(), get_num_layer() + 1);
  }

  bool train(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
             int max_iteration, double max_error) {
    return CALL_MEMBER_FUNC(this, train_func_[topology_type_][training_algorithm_])(
        inputs, desired_outputs, max_iteration, max_error);
  }

  std::vector<double> run(const std::vector<double> &inputs) const {
    return CALL_MEMBER_FUNC(this, run_func_[topology_type_])(inputs);
  }

 private:

  // The func_d is the derivative of func.  The func_s is the
  // symmetric version of func.  And note for sigmoid and tanh, the
  // derivative is based on output of sigmoid and tanh respectively.

  double linear(double x, double k) const { return k * x; }
  double linear_d(double /* x */, double k) const { return k; }

  // They are not supposed to be used for training, so no derivatives
  // are defined here.
  double threshold(double x, double /* k */) const { return x < 0 ? 0 : 1; }
  double threshold_s(double x, double /* k */) const { return x < 0 ? -1 : 1; }

  double sigmoid(double x, double k) const { return 1 / (1 + std::exp(-k * x)); }
  double sigmoid_d(double y, double k) const { return k * y * (1 - y); }

  double tanh(double x, double k) const { return std::tanh(k * x); }
  double tanh_d(double y, double k) const { return k * (1 - y * y); }

  bool mlp_ebp(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
               int max_iteration, double max_error);
  bool mlp_nbn(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
               int max_iteration, double max_error);
  bool acn_ebp(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
               int max_iteration, double max_error);
  bool acn_nbn(const std::vector<double> &inputs, const std::vector<double> &desired_outputs,
               int max_iteration, double max_error);

  std::vector<double> run_mlp(const std::vector<double> &inputs) const;
  std::vector<double> run_acn(const std::vector<double> &inputs) const;

  int bisearch_le(int key, const int *arr, int size) const {
    int low = 0, high = size - 1;
    if (key >= arr[high]) return high;

    while (low < high) {
      int mid = (low + high) / 2;
      if (arr[mid] > key) high = mid;
      else low = mid + 1;
    }
    return high - 1;
  }

  int bisearch_ge(int key, const int *arr, int size) const {
    int low = 0, high = size - 1;
    if (key > arr[high]) return size;
    while (low < high) {
      int mid = (low + high) / 2;
      if (arr[mid] >= key) high = mid;
      else low = mid + 1;
    }
    return high;
  }

  double calcerror(const std::vector<double> desired_outputs, const std::vector<double outputs) {
    int num_output = get_num_output();
    int num_pattern = outputs.size() / num_input;
    double error = 0.0;
    for (int i = 0; i < num_pattern; ++i) {
      double e = 0.0;
      int offset = i * num_output;
      for (int j = 0; j < num_output; ++j)
        e += desired_outputs[offset + j] - outputs[offset + j];
      error += e * e;
    }
    return error;
  }

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
  std::vector<double> gain_;

  // Stores weight for every connection in a long vector which has the
  // same structure as topology_.
  Eigen::VectorXd weight_;

  std::vector<double> weightcopy_;

  std::vector<double> errors_;

  // Index mapping from (i, j) to index in topology_.  Since the
  // connection is undirected, this is a symmetric matrix.  We only
  // use the upper triangle so that the index (i, j) should always
  // satisfy i <= j.
  Eigen::MatrixXi lookup_;

  std::vector<double> error;

  nbn_param param_;

  // Activation function for each neuron, for input layer, it's always
  // linear.  Other layers are different based on user settings.
  std::vector<nbn_activation_func_enum> activation_;
  const nbn_activation_func_t activation_func_[NBN_ACTIVATION_ENUM];
  const nbn_activation_func_t activation_func_d_[NBN_ACTIVATION_ENUM];

  nbn_train_enum training_algorithm_;
  nbn_topology_enum topology_type_;

  const nbn_train_func_t train_func_[NBN_TOPOLOGY_ENUM][NBN_TRAIN_ENUM];
  const nbn_run_func_t run_func_[NBN_TOPOLOGY_ENUM];
};

#endif	// NBN_H_
