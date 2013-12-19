#include <gtest/gtest.h>

#include <vector>
#include <iostream>

#include "nbn.h"

class NbnTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    const std::vector<int> topology = {
      3, 2, 1,
      4, 1, 2,
      5, 1, 2, 3, 4,
      6, 1, 2, 3, 4,
      7, 1, 2, 3, 4, 5, 6,
      8, 1, 2, 3, 4, 5, 6};
    const std::vector<int> outputs = {5, 7, 8};
    nbn.set_topology(topology, outputs);
  }

  static NBN nbn;
};

NBN NbnTest::nbn;

TEST_F(NbnTest, get_num_input)
{
  ASSERT_EQ(2, nbn.get_num_input());
}

TEST_F(NbnTest, get_num_output)
{
  ASSERT_EQ(3, nbn.get_num_output());
}

TEST_F(NbnTest, get_num_connection)
{
  ASSERT_EQ(24, nbn.get_num_connection());
}

TEST_F(NbnTest, get_num_weight)
{
  ASSERT_EQ(30, nbn.get_num_weight());
}

TEST_F(NbnTest, get_num_neuron)
{
  ASSERT_EQ(8, nbn.get_num_neuron());
}

TEST_F(NbnTest, get_layer_size)
{
  const std::vector<int> layer_size = {2, 2, 2, 2};
  std::vector<int> test = nbn.get_layer_size();
  ASSERT_EQ(layer_size, nbn.get_layer_size());
}
