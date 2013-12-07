#include <gtest/gtest.h>

#include <vector>
#include <iostream>

#include "nbn.h"

class NbnTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    const std::vector<int> topology = {
      3, 2, 1,
      4, 1, 2,
      5, 1, 2, 3, 4,
      6, 1, 2, 3, 4,
      7, 1, 2, 3, 4, 5, 6,
      8, 1, 2, 3, 4, 5, 6};
    nbn.set_topology(topology);
  }

  NBN nbn;
};

TEST_F(NbnTest, get_layer_size)
{
  const std::vector<int> layer_size = {2, 2, 2, 2};
  std::vector<int> test = nbn.get_layer_size();
  EXPECT_EQ(layer_size, nbn.get_layer_size());
}

TEST_F(NbnTest, get_connection_count)
{
  EXPECT_EQ(24, nbn.get_connection_count());
}

TEST_F(NbnTest, get_neuron_count)
{
  EXPECT_EQ(8, nbn.get_neuron_count());
}
