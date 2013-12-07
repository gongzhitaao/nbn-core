#ifndef NBN_H_
#define NBN_H_

#include <vector>
#include <Eigen/Dense>

class NBN
{
  typedef bool (NBN::*nbn_train_func_t) (int, double);
  typedef std::vector<int> nbn_index_t;

public:

  enum nbn_train_enum {
    NBN_EBP = 0,		// standard error-backpropagation
    NBN_NBN,			// NBN algorithm
    NBN_TRAIN_ENUM
  };

  NBN();
  ~NBN();

  bool set_training_algorithm(nbn_train_enum training_algorithm);

  nbn_train_enum get_training_algorithm() const {
    return training_algorithm_;
  }

  void set_topology(const nbn_index_t &topology);

  nbn_index_t get_layer_size() const {
    nbn_index_t layer_array;
    layer_array.push_back(num_input_);
    for (unsigned i = 0; i < layer_index_.size() - 1; ++i)
      layer_array.push_back(layer_index_[i+1] - layer_index_[i]);
    layer_array.push_back(num_output_);
    return layer_array;
  }

  int get_connection_count() const {
    return num_weight_;
  }

  int get_neuron_count() const {
    return neuron_index_.size() + 1;
  }

  void initialize_weight();

  bool train(std::vector<double> inputs, std::vector<double> desired_outputs,
	     int max_iteration, double max_error);

  bool run(const std::vector<double> input);

private:
  bool ebp(int max_iteration, double max_error);
  bool nbn(int max_iteration, double max_error);

  int num_input_;
  int num_output_;
  int num_weight_;

  int num_pattern_;

  Eigen::VectorXd gain_;
  Eigen::VectorXi activation_;

  Eigen::VectorXd weight_;
  Eigen::VectorXd output_;
  Eigen::VectorXd desired_output_;

  Eigen::MatrixXd hessian;

  nbn_index_t topology_;
  nbn_index_t neuron_index_;
  nbn_index_t layer_index_;

  nbn_train_enum training_algorithm_;
  const nbn_train_func_t train_func_[NBN_TRAIN_ENUM];
};

#endif	// NBN_H_
