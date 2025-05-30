//
//  NN.cpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#include "NN.h"
#include "game.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "tetrisrandom.hpp"
#include <_stdlib.h>
#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <random>
#include <stdio.h>

using namespace mlx::core;


array leakyRelu(const array& input) {
  return maximum(input, {0.01 * input});
}


void printArray(array a) {
    printf("SUM: %f\n", sum(a).item<float>());
    printf("Mean: %f\n", mean(a).item<float>());
    printf("Max: %f\n", max(a).item<float>());
    printf("Min: %f\n", min(a).item<float>());
    for (auto s: a) {
        for (auto j: s) {
            printf(" %f ", j.item<float>());
        }
        printf("\n");
    }
}
void printVect(array x) {
    for (auto i: x) {
        printf("%f ", i.item<float>());
    }
    printf("\n");
}
double summed(int* arr, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}
array stateToArray(tetrisState s) {
    evars* ev = s.ev;
    bestc b = s.pos;
    int lines = s.linesCleared;

    std::vector<double> input = {
        // float(ev->hMax) / 20,
        double(ev->numHoles),
        // float(ev->minMax) / 20,
        // float(meaned(ev->colHeights, 10)),
        // float(meaned(ev->deltaColHeights, 9)),
        summed(ev->colHeights, 10),
        summed(ev->deltaColHeights, 9),
        double(lines),
        // float(numberOfCaveats) / 10,
        // float(b.col)/10,
    };

    std::vector<int> shape = {int(input.size())};
    mlx::core::array inputArray = mlx::core::array(input.data(), shape, float32);
    return inputArray;
}

array relu(const array& input) {
    // return input;
    return maximum(input, array(0)); // Applies element-wise maximum [1]
}

array generalizedForward(const array& x, const std::vector<array> params) {
    std::vector<array> xs = {x};

    for (int i = 0; i < params.size(); i+=2) {
        array x1 = matmul(xs.back(), transpose(params[i])) + params[i+1];
        if (i < params.size() - 2) {
            array x2 = leakyRelu(x1);
            xs.push_back(x2);
        }
        else {
            xs.push_back(x1);
        }
    }
    return xs.back();
}

std::vector<tetrisState> possibleStates(mat m, block s, evars* previousEvars) {
    std::vector<tetrisState> states;

    // Early validation
    if (m.cols <= 0 || s.numberOfShapes <= 0) {
        return states;
    }

    states.reserve(m.cols * s.numberOfShapes); // Preallocate for better performance

    for (int i = 0; i < m.cols; i++) {
        for (int r = 0; r < s.numberOfShapes; r++) {
            s.currentShape = r;
            int numCleared = 0;
            s.position[1] = i;

            mat* preview = previewMatIfPushDown(&m, s, &numCleared);
            if (preview && preview != NULL) {
                evars* ev = retrieveEvars(*preview, previousEvars);
                if (ev) {
                    bestc config = {.col = i, .shapeN = r};
                    states.push_back({
                        .ev = ev,
                        .pos = config,
                        .linesCleared = numCleared
                    });
                }
                freeMat(preview);
            }
        }
    }

    return states;
}
