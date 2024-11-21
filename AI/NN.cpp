//
//  NN.cpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#include "NN.hpp"
#include "game.h"
#include <_stdlib.h>
#include <algorithm>
#include <assert.h>
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <random>

using namespace mlx::core;

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
array stateToArray(tetrisState s) {
    evars* ev = std::get<0>(s);
    bestc b = std::get<1>(s);
    int lines = std::get<2>(s);

    std::vector<float> input = {
        float(ev->hMax) / 20,
        float(ev->numHoles) / 200,
        float(ev->minMax) / 20,
        float(meaned(ev->colHeights, 10)),
        float(meaned(ev->deltaColHeights, 9)),
        float(lines) / float(4.0),
    };

    std::vector<int> shape = {int(input.size())};
    mlx::core::array inputArray = mlx::core::array(input.data(), shape, float32);

    return inputArray;
}

array relu(const array& input) {
  return maximum(input, {0.0}); // Applies element-wise maximum [1]
}
array generalizedForward(const array& x, const std::vector<array> params) {
    std::vector<array> xs = {x};

    for (int i = 0; i < params.size(); i+=2) {
        array x1 = matmul(xs.back(), transpose(params[i])) + params[i+1];
        if (i < params.size() - 2) {
            array x2 = relu(x1);
            xs.push_back(x2);
        }
        else {
            xs.push_back(x1);
        }
    }
    return xs.back();
}

array DQN::generalizedForward(const array& x, const std::vector<array> params) {
    std::vector<array> xs = {x};

    for (int i = 0; i < params.size(); i+=2) {
        array x1 = matmul(xs.back(), transpose(params[i])) + params[i+1];
        if (i < params.size() - 2) {
            array x2 = relu(x1);
            xs.push_back(x2);
        }
        else {
            xs.push_back(x1);
        }
    }
    return xs.back();
}

array heuristic(const array& input) {
    array hMax = (*input.begin());
    array numHoles = (*(input.begin() + 1));
    array numCleared = (*(input.begin() + 5));
    return hMax * (-2) + numHoles * (-5) + numCleared * 4;
}

array heuristic2(const array& input) {
    array hMax = (*input.begin());
    array numHoles = (*(input.begin() + 1));
    array minMax = (*(input.begin() + 2));
    array colHeights = (*(input.begin() + 3));
    array deltaColHeights = (*(input.begin() + 4));
    array numCleared = (*(input.begin() + 5));

    return hMax * (-2) + numHoles * (-5) + numCleared * 4 + deltaColHeights * (-1) + minMax * (-1) + colHeights * (-1.5);
}

array heuristic3(const array& input) {
    array hMax = (*input.begin());
    array numHoles = (*(input.begin() + 1));
    array minMax = (*(input.begin() + 2));
    array colHeights = (*(input.begin() + 3));
    array deltaColHeights = (*(input.begin() + 4));
    array numCleared = (*(input.begin() + 5));

    return -0.3 * colHeights + 8 * numCleared - 7.5 * numHoles - 5 * deltaColHeights;
}

array heuristic4(const array& input) {
    array hMax = (*input.begin());
    array numHoles = (*(input.begin() + 1));
    array minMax = (*(input.begin() + 2));
    array colHeights = (*(input.begin() + 3));
    array deltaColHeights = (*(input.begin() + 4));
    array numCleared = (*(input.begin() + 5));

    return -0.3 * colHeights + 8 * numCleared - 7.5 * numHoles - 5 * deltaColHeights - hMax*hMax;
}
std::vector<array> DQN::batchHeuristic(std::vector<array> states) {
    std::vector<array> ys = {};

    for (array input : states) {
        array y = heuristic4(input);
        ys.push_back(y);
    }
    return ys;
}

std::vector<tetrisState> possibleStates(mat m, block s, evars* previousEvars) {
    std::vector<std::tuple<evars *, bestc, int>> states;

    for (int i = 0; i < m.cols; i++) {
        for (int r = 0; r < s.numberOfShapes; r++) {
            s.currentShape = r;
            int numCleared = 0;
            s.position[1] = i;

            mat *preview = previewMatIfPushDown(&m, s, &numCleared);
            if (preview && preview != NULL) {
                evars* ev = retrieveEvars(*preview, previousEvars);
                bestc config = {
                    .col = i,
                    .shapeN = r
                };
                states.push_back(std::tuple<evars *, bestc, int>(ev, config, numCleared));
                freeMat(preview);
            }
        }
    }
    return states;
}

void DQN::train(std::vector<array> states, std::vector<array> yTruth) {

    std::vector<int> argnums(ml->params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    for (int epoch = 0; epoch < 10; epoch++) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> randI(0, states.size() - 1);

        int r = randI(rng);

        assert(r < states.size());
        printf("Random sample: %d\n", r);

        std::vector<array> input = ml->params;
        input.push_back(states[r]);

        auto loss_fn = [&yTruth, &r, this](const std::vector<array>& input) {
            array trueScore = yTruth[r];
            std::vector<array> params(input.begin(), input.end() - 1);
            array x = input.back();

            array predictions = generalizedForward(x, params);
            return mean(square(predictions - trueScore));
        };

        auto [loss, grads] = value_and_grad(loss_fn, argnums)(input);

        eval(loss);
        ml->update_parameters(grads, 0.001);
        printf("Epoch %d, loss: %f\n", epoch, loss.item<float>());
    }

    // Update epsilon for exploration
    epsilon = std::max(eps_min, epsilon * eps_decay);
}

bool DQN::tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, bool userMode, block** BASIC_BLOCKS) {
    int down = downShape(*m, s);

    // *score += 1;

    if (down == -1) {
        computeDownPos(*m, s);

        int numCleared = pushToMat(m, *s);
        *score += 200 * pow(numCleared, 2);
        *linesCleared += numCleared;

        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, randomBlock(BASIC_BLOCKS));

        std::vector<tetrisState> maybeStates = possibleStates(*m, *s, e);

        bestc compo = this->act(maybeStates);

        // 50% chance to add a random state

        if (compo.shapeN == -1) { return false; }

        // if (generateRandomDouble(0, 1) < 0.5) {
        //     mem.push_back(maybeStates[randomIntBetween(0, maybeStates.size() - 1)]);
        // }
        int nextPosX = compo.col;

        assert(nextPosX >= 0 && nextPosX < m->cols + 2);

        s->position[1] = nextPosX;
        s->currentShape = compo.shapeN;
        computeDownPos(*m, s);

        bool canInsert = canInsertShape(*m, *s);

        if (canInsert) {
            // great!
        } else {
            printf("Can't insert shape...");
            // printMat(m, *s);
            printMatrix(m->data, m->rows, m->cols);
            printMatrix(s->shape[compo.shapeN], 4, 4);
        }
        return canInsert;
    }
    return true;
}

bestc DQN::act(std::vector<tetrisState>& possibleStates) {
    if (generateRandomDouble(0, 1) < epsilon) {
        printf("\nEXPLORING --- \n");
        tetrisState randomState = possibleStates[generateRandomNumber(0, possibleStates.size() - 1)];
        return std::get<1>(randomState);
    }
    float max_rating = -std::numeric_limits<float>::infinity();

    bestc best_action = {
        .col = -1,
        .shapeN = -1
    };
    int best_index = -1;
    std::vector<array> ratings = batchForward(possibleStates);

    for (int i = 0; i < ratings.size(); i++) {
        float rating = ratings[i].item<float>();
        if (rating > max_rating) {
            if (best_index != -1) {
                evars* previousBest = std::get<0>(possibleStates[best_index]);
                if (previousBest != nullptr) {
                    free(previousBest->colHeights);
                    free(previousBest->deltaColHeights);
                    free(previousBest);
                }
            }

            max_rating = rating;
            best_action = std::get<1>(possibleStates[i]);
            best_index = i;
        } else {
            evars* e = std::get<0>(possibleStates[i]);
            free(e->colHeights);
            free(e->deltaColHeights);
            free(e);
        }
    }
    if (best_action.shapeN != -1 && generateRandomDouble(0, 1) < 0.5) {
        mem.push_back(stateToArray(possibleStates[best_index]));
        if (mem.size() >= memCapacity) {
            mem.erase(mem.begin());
        }
        evars* best = std::get<0>(possibleStates[best_index]);
        free(best->colHeights);
        free(best->deltaColHeights);
        free(best);
    }
    return best_action;
}
