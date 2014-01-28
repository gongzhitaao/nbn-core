#include <gtest/gtest.h>

#include <vector>
#include <iostream>

#include "nbn.h"

/* --------------------------------------------------------------------
 * set_topology test
 * --------------------------------------------------------------------
 */

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
    // const std::vector<int> outputs = {7, 8};
    nbn.set_topology(topology);
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
  ASSERT_EQ(2, nbn.get_num_output());
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
  ASSERT_EQ(6, nbn.get_num_neuron());
}

TEST_F(NbnTopologyTest00, get_layer_size)
{
  const std::vector<int> layer_size = {2, 2, 2, 2};
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}

TEST_F(NbnTopologyTest00, get_neuron_layer)
{
  EXPECT_EQ(0, nbn.get_neuron_layer(1));
  EXPECT_EQ(0, nbn.get_neuron_layer(2));
  EXPECT_EQ(1, nbn.get_neuron_layer(3));
  EXPECT_EQ(1, nbn.get_neuron_layer(4));
  EXPECT_EQ(2, nbn.get_neuron_layer(5));
  EXPECT_EQ(2, nbn.get_neuron_layer(6));
  EXPECT_EQ(3, nbn.get_neuron_layer(7));
  EXPECT_EQ(3, nbn.get_neuron_layer(8));
}

class NbnTopologyTest01 : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    const std::vector<int> topology = {2, 1};
    // const std::vector<int> outputs = {2};
    nbn.set_topology(topology);
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
  ASSERT_EQ(1, nbn.get_num_neuron());
}

TEST_F(NbnTopologyTest01, get_layer_size)
{
  const std::vector<int> layer_size = {1, 1};
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}

TEST_F(NbnTopologyTest01, get_neuron_layer)
{
  EXPECT_EQ(0, nbn.get_neuron_layer(1));
  EXPECT_EQ(1, nbn.get_neuron_layer(2));
}

class NbnTopologyTest02 : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    const std::vector<int> topology = {
      2, 1,
      3, 2, 1,
      4, 1, 2, 3};
    const std::vector<int> outputs = {3, 4};
    nbn.set_topology(topology, outputs);
  }

  static NBN nbn;
};

NBN NbnTopologyTest02::nbn;

TEST_F(NbnTopologyTest02, get_num_input)
{
  ASSERT_EQ(1, nbn.get_num_input());
}

TEST_F(NbnTopologyTest02, get_num_output)
{
  ASSERT_EQ(2, nbn.get_num_output());
}

TEST_F(NbnTopologyTest02, get_num_connection)
{
  ASSERT_EQ(6, nbn.get_num_connection());
}

TEST_F(NbnTopologyTest02, get_num_weight)
{
  ASSERT_EQ(9, nbn.get_num_weight());
}

TEST_F(NbnTopologyTest02, get_num_neuron)
{
  ASSERT_EQ(3, nbn.get_num_neuron());
}

TEST_F(NbnTopologyTest02, get_layer_size)
{
  const std::vector<int> layer_size = {1, 1, 1, 1};
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}

TEST_F(NbnTopologyTest02, get_neuron_layer)
{
  EXPECT_EQ(0, nbn.get_neuron_layer(1));
  EXPECT_EQ(1, nbn.get_neuron_layer(2));
  EXPECT_EQ(2, nbn.get_neuron_layer(3));
  EXPECT_EQ(3, nbn.get_neuron_layer(4));
}

class NbnTopologyTest03 : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    const std::vector<int> topology = {
      3, 2, 1,
      4, 2, 1,
      5, 2, 1,
      6, 3, 4, 5,
      7, 3, 4, 5,
      8, 3, 4, 5,
      9, 6, 7, 8};
    const std::vector<int> outputs = {9};
    nbn.set_topology(topology, outputs);
  }

  static NBN nbn;
};

NBN NbnTopologyTest03::nbn;

TEST_F(NbnTopologyTest03, get_num_input)
{
  ASSERT_EQ(2, nbn.get_num_input());
}

TEST_F(NbnTopologyTest03, get_num_output)
{
  ASSERT_EQ(1, nbn.get_num_output());
}

TEST_F(NbnTopologyTest03, get_num_connection)
{
  ASSERT_EQ(18, nbn.get_num_connection());
}

TEST_F(NbnTopologyTest03, get_num_weight)
{
  ASSERT_EQ(25, nbn.get_num_weight());
}

TEST_F(NbnTopologyTest03, get_num_neuron)
{
  ASSERT_EQ(7, nbn.get_num_neuron());
}

TEST_F(NbnTopologyTest03, get_layer_size)
{
  const std::vector<int> layer_size = {2, 3, 3, 1};
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}

TEST_F(NbnTopologyTest03, get_neuron_layer)
{
  EXPECT_EQ(0, nbn.get_neuron_layer(1));
  EXPECT_EQ(0, nbn.get_neuron_layer(2));
  EXPECT_EQ(1, nbn.get_neuron_layer(3));
  EXPECT_EQ(1, nbn.get_neuron_layer(4));
  EXPECT_EQ(1, nbn.get_neuron_layer(5));
  EXPECT_EQ(2, nbn.get_neuron_layer(6));
  EXPECT_EQ(2, nbn.get_neuron_layer(7));
  EXPECT_EQ(2, nbn.get_neuron_layer(8));
  EXPECT_EQ(3, nbn.get_neuron_layer(9));
}

/* --------------------------------------------------------------------
 * binary search test
 * --------------------------------------------------------------------
 */

int bisearch_ge(int key, const int *arr, int size) {
  int low = 0, high = size - 1;
  if (key > arr[high]) return size;
  while (low < high) {
    int mid = (low + high) / 2;
    if (arr[mid] >= key) high = mid;
    else low = mid + 1;
  }
  return high;
}

TEST(bisearch, bisearch_ge)
{
  std::vector<int> arr = {1, 3, 5, 7, 9};

  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(i, bisearch_ge(2 * i + 1, arr.data(), 5));

  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(i + 1, bisearch_ge(2 * i + 2, arr.data(), 5));
}

int bisearch_le(int key, const int *arr, int size) {
  int low = 0, high = size - 1;
  // only one layer
  if (low == high) return low;
  if (key >= arr[high]) return high;

  while (low < high) {
    int mid = (low + high) / 2;
    if (arr[mid] > key) high = mid;
    else low = mid + 1;
  }
  return high - 1;
}

TEST(bisearch, bisearch_le)
{
  std::vector<int> arr = {1, 3, 5, 7, 9};

  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(i, bisearch_le(2 * i + 1, arr.data(), 5));

  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(i, bisearch_le(2 * i + 2, arr.data(), 5));
}

/* --------------------------------------------------------------------
 * training test
 * --------------------------------------------------------------------
 */

TEST(NbnTrainingTest, parity3)
{
  const std::vector<int> topology = {
    2, 1,
    3, 1, 2,
    4, 1, 2, 3};
  const std::vector<int> out = {4};
  const std::vector<double> inputs = {-3., -1., 1., 3.};
  const std::vector<double> outputs = {-1., 1., -1., 1.};

  NBN nbn;
  nbn.set_topology(topology, out);

  nbn.train(inputs, outputs, 100, 0.001);

  std::vector<double> output = nbn.run(inputs);
  for (unsigned i = 0; i < output.size(); ++i)
    std::cout << output[i] << ' ';
  std::cout << std::endl;
}

TEST(NbnTrainingTest, parity5)
{
  const std::vector<int> topology = {
    2, 1,
    3, 1, 2,
    4, 1, 2, 3};
  const std::vector<int> out = {4};
  const std::vector<double> inputs = {-5., -3., -1., 1., 3., 5.};
  const std::vector<double> outputs = {-1., 1., -1., 1., -1., 1.};

  NBN nbn;
  nbn.set_topology(topology, out);

  nbn.train(inputs, outputs, 100, 0.001);

  std::vector<double> output = nbn.run(inputs);
  for (unsigned i = 0; i < output.size(); ++i)
    std::cout << output[i] << ' ';
  std::cout << std::endl;
}

TEST(NbnTrainingTest, parity7)
{
  const std::vector<int> topology = {
    2, 1,
    3, 1, 2,
    4, 1, 2, 3,
    5, 1, 2, 3, 4};
  const std::vector<int> out = {5};
  const std::vector<double> inputs = {-7., -5., -3., -1., 1., 3., 5., 7.};
  const std::vector<double> outputs = {-1., 1., -1., 1., -1., 1., -1., 1.};

  NBN nbn;
  nbn.set_topology(topology, out);

  nbn.train(inputs, outputs, 1000, 0.001);

  std::vector<double> output = nbn.run(inputs);
  for (unsigned i = 0; i < output.size(); ++i)
    std::cout << output[i] << ' ';
  std::cout << std::endl;
}

TEST(NbnTrainingTest, logic_xor)
{
  const std::vector<int> topology = {
    3, 1, 2,
    4, 1, 2, 3};
  const std::vector<int> out = {4};
  const std::vector<double> inputs = {-1, -1, -1, 1, 1, -1, 1, 1};
  const std::vector<double> outputs = {-1, 1, 1, -1};

  NBN nbn;
  nbn.set_topology(topology, out);

  nbn.train(inputs, outputs, 50, 0.001);

  std::vector<double> output = nbn.run(inputs);
  for (unsigned i = 0; i < output.size(); ++i)
    std::cout << output[i] << ' ';
  std::cout << std::endl;
}

TEST(NbnTrainingTest, logic_and)
{
  const std::vector<int> topology = {
    3, 1, 2,
    4, 1, 2, 3};
  const std::vector<int> out = {4};
  const std::vector<double> inputs = {-1, -1, -1, 1, 1, -1, 1, 1};
  const std::vector<double> outputs = {-1, -1, -1, 1};

  NBN nbn;
  nbn.set_topology(topology, out);

  nbn.train(inputs, outputs, 50, 0.001);

  std::vector<double> output = nbn.run(inputs);
  for (unsigned i = 0; i < output.size(); ++i)
    std::cout << output[i] << ' ';
  std::cout << std::endl;
}

TEST(NbnTrainingTest, logic_or)
{
  const std::vector<int> topology = {
    3, 1, 2,
    4, 1, 2, 3};
  const std::vector<int> out = {4};
  const std::vector<double> inputs = {-1, -1, -1, 1, 1, -1, 1, 1};
  const std::vector<double> outputs = {-1, 1, 1, 1};

  NBN nbn;
  nbn.set_topology(topology, out);

  nbn.train(inputs, outputs, 50, 0.001);

  std::vector<double> output = nbn.run(inputs);
  for (unsigned i = 0; i < output.size(); ++i)
    std::cout << output[i] << ' ';
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  ::testing::GTEST_FLAG(filter) = "NbnTrainingTest*";
  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
