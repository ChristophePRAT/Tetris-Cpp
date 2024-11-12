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
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"
#include <cstdint>
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

/*
void train() {
    MultiLayer ml = MultiLayer(2, {16,16,1});

    array x = linspace(0, 3.141592, 100);
    array y = cos(x);
    array targetZ = sin(x) + 3*y;
    eval(targetZ);

    array inputs = transpose(stack({x,y}));

    auto loss_fn = [&targetZ, &inputs](const std::vector<array>& input) {
            int index = input.back().item<int>();
            array inp = *(inputs.begin() + index);
            array zTrue = *(targetZ.begin() + index);

            std::vector<array> p = input;
            p.pop_back();

            array predictions = generalizedForward(inp, p);

            return mean(square(predictions - zTrue));
        };

    std::vector<int> argnums = {};

    for (int i = 0; i < ml.params.size(); i++) {
        argnums.push_back(i);
    }

    for (int epoch = 0; epoch < 1000; epoch++) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> randI(0,inputs.size()/2 - 1);

        int r = randI(rng);

        assert(r<inputs.size()/2);
        printf("Random sample: %d\n", r);

        std::vector<array> input = ml.params;

        input.push_back(array(r));

        auto [loss, grads] = value_and_grad(loss_fn, argnums)(input);;

        eval(loss);
        ml.update_parameters(grads, 0.01);
        printf("Epoch %d, loss: %f\n", epoch, loss.item<float>());
    }
    exit(0);
}
*/
float heuristic(evars e, int numCleared) {
    return e.hMax * (-2) + e.numHoles * (-5) + numCleared * 4;
}

std::vector<tetrisState> possibleStates(mat m, block s, evars* previousEvars) {
    std::vector<std::tuple<evars, bestc, int>> states;

    for (int i = 0; i < m.cols; i++) {
        for (int r = 0; r < s.numberOfShapes; r++) {
            s.currentShape = r;
            int numCleared = 0;
            mat *preview = previewMatIfPushDown(&m, s, &numCleared);
            if (preview == NULL) {
                printf("Move not possible: { %d, %d }", i,r);
                continue;
            }
            evars* ev = retrieveEvars(*preview, previousEvars);
            bestc config = {
                .col = i,
                .shapeN = r
            };
            states.push_back(std::tuple<evars, bestc, int>(*ev, config, numCleared));
            freeMat(preview);
            free(ev->colHeights);
            free(ev->deltaColHeights);
            free(ev);

        }
    }
    return states;
}
/*

// Modified training step with improved learning
void trainStep(mat m, block s, evars* e, MultiLayer ml) {
    int numCleared = 0;
    mat *preview = previewMatIfPushDown(&m, s, &numCleared);
    if (!preview) return;

    evars *ev = retrieveEvars(*preview, e);

    // Modified heuristic scaling
    float heurScore = heuristic(*ev, numCleared) / 100.0;  // Scale down the target values

    // Add positional information to input features
    std::vector<float> input = {
        float(ev->hMax) / m.rows,
        float(ev->numHoles) / (m.rows * m.cols),
        float(ev->minMax) / m.rows,
        float(meaned(ev->colHeights, m.cols)) / m.rows,
        float(meaned(ev->deltaColHeights, m.cols)) / m.rows,
        float(numCleared) / float(4.0),
        float(s.position[1]) / m.cols,  // Add normalized column position
        float(s.currentShape) / s.numberOfShapes  // Add normalized shape number
    };

    std::vector<int> shape = {8};  // Updated input size
    mlx::core::array inputArray = mlx::core::array(input.data(), shape, float32);

    auto loss_fn = [&heurScore](const std::vector<array>& input) {
        array trueScore = array({heurScore});
        std::vector<array> params(input.begin(), input.end() - 1);
        array x = input.back();

        array predictions = generalizedForward(x, params);
        return mean(square(predictions - trueScore));
    };

    std::vector<int> argnums(ml.params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    std::vector<array> inputs = ml.params;
    inputs.push_back(inputArray);

    auto [loss, grads] = value_and_grad(loss_fn, argnums)(inputs);
    ml.update_parameters(grads, 0.01);

    // printf("Loss: %f, Learning Rate: %f, Patience: %d\n",
           // loss.item<float>(), 0.01, 1);

    freeMat(preview);
    free(ev->colHeights);
    free(ev->deltaColHeights);
    free(ev);
    }*/

void DQN::train(std::vector<array> states, std::vector<array> yTruth) {

    std::vector<int> argnums(ml->params.size());
    std::iota(argnums.begin(), argnums.end(), 0);

    for (int epoch = 0; epoch < 1000; epoch++) {
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
        ml->update_parameters(grads, 0.01);
        printf("Epoch %d, loss: %f\n", epoch, loss.item<float>());
    }

    // Update epsilon for exploration
    epsilon = std::max(eps_min, epsilon * eps_decay);
}

// Modified decision function with position-aware scoring
bestc theFinestDecision(mat m, block s, evars* previousEvars, MultiLayer ml) {
    std::vector<std::tuple<evars, bestc, int>> states = possibleStates(m, s, previousEvars);
    float maxScore = -std::numeric_limits<float>::infinity();
    bestc bestConfig = {
        .col = 0,
        .shapeN = 0
    };

    if (states.size() == 0) {
        printf("NO MOVES POSSIBLE");
        return {0, -1};
    }

    // Improved exploration strategy
    static int totalMoves = 0;
    totalMoves++;
    float explorationRate = 0.2 * exp(-totalMoves / 20000.0);  // Slower decay

    if (true) {
        int index = randomIntBetween(0, states.size() - 1);
        auto [e, b, i] = states[index];
        // printf("Exploring random move (rate: %f)\n", explorationRate);
        return b;
    }

    for (const auto& [ev, config, numCleared] : states) {
        std::vector<float> input = {
            float(ev.hMax) / m.rows,
            float(ev.numHoles) / (m.rows * m.cols),
            float(ev.minMax) / m.rows,
            float(meaned(previousEvars->colHeights, m.cols)) / m.rows,
            float(meaned(previousEvars->deltaColHeights, m.cols)) / m.rows,
            float(numCleared) / float(4.0),
            float(config.col) / m.cols,  // Add column position
            float(config.shapeN) / s.numberOfShapes  // Add shape information
        };

        std::vector<int> shape = {8};
        mlx::core::array inputArray = mlx::core::array(input.data(), shape, float32);

        array prediction = generalizedForward(inputArray, ml.params);
        float score = prediction.item<float>();

        // Add position-based bonus/penalty
        float heightPenalty = ev.hMax / float(m.rows);  // Penalize high stacks
        float holePenalty = ev.numHoles * 0.1;  // Penalize holes
        float clearBonus = numCleared * 0.2;  // Bonus for clearing lines

        score = score - heightPenalty - holePenalty + clearBonus;

        printf("Col: %d, Shape: %d, Raw Score: %f, Final Score: %f\n",
               config.col, config.shapeN, prediction.item<float>(), score);

        if (score > maxScore) {
            maxScore = score;
            bestConfig = config;
        }
    }

    return bestConfig;
}
bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, MultiLayer ml, unsigned int index, bool userMode, block** BASIC_BLOCKS) {
    int down = downShape(*m, s);

    // *score += 1;

    if (down == -1) {
        computeDownPos(*m, s);
        printf("Down pos: %d \n", s->downPos);

        int numCleared = pushToMat(m, *s);
        *score += 200 * pow(numCleared, 2);

        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, randomBlock(BASIC_BLOCKS));

        bestc compo = theFinestDecision(*m, *s, e, ml);

        if (compo.shapeN == -1) { return false; }
        int nextPosX = compo.col;

        assert(nextPosX >= 0 && nextPosX < m->cols + 2);

        s->position[1] = nextPosX;
        s->currentShape = compo.shapeN;
        computeDownPos(*m, s);

        bool canInsert = canInsertShape(*m, *s);

        if (canInsert) {
            //trainStep(*m, *s, e, ml);
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

bool DQN::tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int index, bool userMode, block** BASIC_BLOCKS) {
    int down = downShape(*m, s);

    // *score += 1;

    if (down == -1) {
        computeDownPos(*m, s);
        printf("Down pos: %d \n", s->downPos);

        int numCleared = pushToMat(m, *s);
        *score += 200 * pow(numCleared, 2);

        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, randomBlock(BASIC_BLOCKS));

        std::vector<tetrisState> maybeStates = possibleStates(*m, *s, e);

        bestc compo = this->act(maybeStates);

        // 50% chance to add a random state
        if (random::uniform(0,1,{1}).item<float>() < 0.5) {
            mem.push_back(maybeStates[random::uniform(0, maybeStates.size() - 1, {1}).item<int>()]);
        }

        if (compo.shapeN == -1) { return false; }
        int nextPosX = compo.col;

        assert(nextPosX >= 0 && nextPosX < m->cols + 2);

        s->position[1] = nextPosX;
        s->currentShape = compo.shapeN;
        computeDownPos(*m, s);

        bool canInsert = canInsertShape(*m, *s);

        if (canInsert) {
            //trainStep(*m, *s, e, ml);
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
