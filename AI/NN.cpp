//
//  NN.cpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#include "NN.hpp"
#include "mlx/array.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/array.h"
using namespace mlx::core;

array relu(const array& input) {
  return maximum(input, 0.0); // Applies element-wise maximum [1]
}

double forward(mlp* ml, array input) {
  // Layer 1
  array z1 = matmul(input, ml->w1) + ml->b1; // Linear transformation [2]
  array a1 = relu(z1);  // ReLU activation

  // Layer 2
  array z2 = matmul(a1, ml->w2) + ml->b2;
  array a2 = relu(z2);

  // Layer 3
  array z3 = matmul(a2, ml->w3) + ml->b3;
  array a3 = relu(z3);

  // Output Layer
  array z4 = matmul(a3, ml->w4) + ml->b4;
  array output = (z4); // Assuming a sigmoid output

  return output; // Assuming a single output value
}


void updateNN(mlp *ml, double loss) {

}
