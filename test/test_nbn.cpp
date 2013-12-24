#include <gtest/gtest.h>

#include <vector>
#include <iostream>

#include "nbn.h"

class NbnTopologyTest00 : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    const std::vector<int> topology = {
      3, 2, 1,
      4, 1, 2,
      5, 1, 3, 2, 4,
      6, 1, 3, 2, 4,
      7, 1, 2, 3, 5, 4, 6,
      8, 1, 2, 3, 4, 6, 5};
    const std::vector<int> outputs = {5, 7, 8};
    nbn.set_topology(topology, outputs);
  }

  static NBN nbn;
};

NBN NbnTopologyTest00::nbn;

TEST_F(NbnTopologyTest00, get_num_input)
{
  ASSERT_EQ(2, nbn.get_num_input());
}

TEST_F(NbnTopologyTest00, get_num_output)
{
  ASSERT_EQ(3, nbn.get_num_output());
}

TEST_F(NbnTopologyTest00, get_num_connection)
{
  ASSERT_EQ(24, nbn.get_num_connection());
}

TEST_F(NbnTopologyTest00, get_num_weight)
{
  ASSERT_EQ(30, nbn.get_num_weight());
}

TEST_F(NbnTopologyTest00, get_num_neuron)
{
  ASSERT_EQ(8, nbn.get_num_neuron());
}

TEST_F(NbnTopologyTest00, get_layer_size)
{
  const std::vector<int> layer_size = {2, 2, 2, 2};
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}


class NbnTopologyTest01 : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    const std::vector<int> topology = {2, 1};
    const std::vector<int> outputs = {2};
    nbn.set_topology(topology, outputs);
  }

  static NBN nbn;
};

NBN NbnTopologyTest01::nbn;

TEST_F(NbnTopologyTest01, get_num_input)
{
  ASSERT_EQ(1, nbn.get_num_input());
}

TEST_F(NbnTopologyTest01, get_num_output)
{
  ASSERT_EQ(1, nbn.get_num_output());
}

TEST_F(NbnTopologyTest01, get_num_connection)
{
  ASSERT_EQ(1, nbn.get_num_connection());
}

TEST_F(NbnTopologyTest01, get_num_weight)
{
  ASSERT_EQ(2, nbn.get_num_weight());
}

TEST_F(NbnTopologyTest01, get_num_neuron)
{
  ASSERT_EQ(2, nbn.get_num_neuron());
}

TEST_F(NbnTopologyTest01, get_layer_size)
{
  const std::vector<int> layer_size = {1, 1};
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}

// class NbnCoreTest : public ::testing::Test {
//  protected:
//   static void SetUpTestCase() {
//     const std::vector<int> topology = {
//       2, 1,
//       3, 1, 2,
//       4, 1, 2, 3};
//     const std::vector<int> outputs = {4};
//     nbn.set_topology(topology, outputs);
//     nbn.init_default();
//   }

//   static NBN nbn;
// };

// NBN NbnCoreTest::nbn;

// TEST_F(NbnCoreTest, run)
// {
//   const std::vector<double> inputs = {-5, -3, -1, 1, 3, 5};
//   std::vector<double> output = nbn.run(inputs);
//   for (unsigned i = 0; i < output.size(); ++i)
//     std::cout << output[i] << ' ';
//   std::cout << std::endl;
// }

// TEST_F(NbnCoreTest, get_layer_size)
// {
//   const std::vector<int> layer_size = {1, 1, 1, 1};
//   ASSERT_EQ(layer_size, nbn.get_layer_size());
// }

// TEST_F(NbnCoreTest, train)
// {
//   const std::vector<double> inputs = {-5.};
//   const std::vector<double> outputs = {-1.};
//   nbn.train(inputs, outputs, 1, 0.001);

//   std::vector<double> output = nbn.run(inputs);
//   for (unsigned i = 0; i < output.size(); ++i)
//     std::cout << output[i] << ' ';
//   std::cout << std::endl;
// }

// TEST_F(NbnCoreTest, train)
// {
//   const std::vector<double> inputs = {-5., -3., -1., 1., 3., 5.};
//   const std::vector<double> outputs = {-1., 1., -1., 1., -1., 1.};
//   nbn.train(inputs, outputs, 1, 0.001);

//   std::vector<double> output = nbn.run(inputs);
//   for (unsigned i = 0; i < output.size(); ++i)
//     std::cout << output[i] << ' ';
//   std::cout << std::endl;
// }
