//
//  NN.hpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#ifndef NN_hpp
#define NN_hpp
// #include <stdio.h>
#include "game.h"
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include <cstddef>
#include <mlx/mlx.h>
#include <vector>
#include "assert.h"
const int NUM_WEIGHTS = 6;
const int NUM_LAYERS = 2;


using namespace mlx::core;
// a tuple representing the env variables, the combination it came from and the lines cleared
typedef std::tuple<evars *, bestc, int> tetrisState;

class Linear {
    public:
        int input_dims;
        int output_dims;
        array *weights = nullptr;
        array *bias = nullptr;
        int layer_id;
        Linear(int input_dims, int output_dims, int layer_id) { // y = x * transpose(W) + B

            this->input_dims = input_dims;
            this->output_dims = output_dims;

            float k = sqrt(1.0 / input_dims);
            this->weights = new array({output_dims, input_dims});
            *this->weights = random::uniform(-k, k, {output_dims, input_dims});

            this->bias = new array({output_dims});
            *this->bias = random::uniform(-k, k, {output_dims});
            // *this->bias = zeros({output_dims});
            this->layer_id = layer_id;
            eval(*this->weights);
            eval(*this->bias);
            return;
        }
        array forward(array x) {
            return matmul(x,transpose(*this->weights)) + *this->bias;
        }
        void update_layer(const array& weightsGrads,const array& biasGrads, float lr) {

            *this->weights = *this->weights - weightsGrads * lr;
            *this->bias = *this->bias - biasGrads * lr;
            eval(*this->weights);
            eval(*this->bias);
        }
};

class MultiLayer {
public:
    int input_size;
    std::vector<Linear> layers;
    std::vector<array> params;

    MultiLayer(int input_size, std::vector<int> hidden_sizes) {
        this->input_size = input_size;
        // this->output_size = output_size;

        for (int i = 0; i < hidden_sizes.size(); i++) {
            if (i == 0) {
                layers.push_back(Linear(input_size, hidden_sizes[i], i));
            } else {
                layers.push_back(Linear(hidden_sizes[i - 1], hidden_sizes[i], i));
            }
        }

        for (const auto& layer : layers) {
            params.push_back(*layer.weights);
            params.push_back(*layer.bias);
        }
        eval(params);
    }

    array forward(const array& x);

    void update(const std::vector<array>& new_params) {
        if (new_params.size() != params.size()) {
            throw std::runtime_error("Mismatch in parameter count");
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            *layers[i].weights = new_params[2 * i];
            *layers[i].bias = new_params[2 * i + 1];
        }
        params = new_params;
        eval(params);
        assert(mean(*layers[0].weights).item<float>() == mean(params[0]).item<float>());
    }

    void update_parameters(const std::vector<array>& grads, float lr) {
        if (grads.size() != params.size()) {
            throw std::runtime_error("Mismatch in gradient count");
        }
        for (size_t i = 0; i < params.size(); i++) {
            params[i] = params[i] - grads[i] * lr;
        }
        update(params);
    }
};


class DQN {
    public:
        float discount;
        float epsilon;
        float eps_decay;
        float eps_min;
        float lr;
        MultiLayer* ml = nullptr;
        std::vector<array> mem = {};
        unsigned int memCapacity;

    DQN(int input_size, float discount, float epsilon, float eps_decay, float eps_min, float lr, unsigned int memCapacity) {
        this->discount = discount;
        this->epsilon = epsilon;
        this->eps_decay = eps_decay;
        this->eps_min = eps_min;
        this->lr = lr;
        this->ml = new MultiLayer(input_size, {64, 64, 32, 1});
        this->memCapacity = memCapacity;
    }
    bestc act(std::vector<tetrisState>& possibleStates) {
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

    std::tuple<std::vector<array>, std::vector<array>> gatherTrainingData(std::vector<tetrisState> memory) {
        std::vector<array> inputs = batchStateToArray(memory);
        std::vector<array> targets;
        return std::make_tuple(inputs, targets);
    }
    void trainNN() {
        train(mem, batchHeuristic(mem));
    }

    bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, bool userMode, block** BASIC_BLOCKS);

    private:
    std::vector<array> batchHeuristic(std::vector<array>);

    void train(std::vector<array> states, std::vector<array> yTruth);

std::vector<array> batchGetTrue(std::vector<tetrisState> states) {
        std::vector<array> predictions;
        for (int i = 0; i < states.size(); i++) {
            array input = stateToArray(states[i]);

            array trueValue = mlx::core::matmul(input, transpose(array({-1,-2,-0.4,-0.5,4})));
            predictions.push_back(trueValue);
        }
        return predictions;
    }
    array stateToArray(tetrisState s) {
        evars* ev = std::get<0>(s);
        bestc b = std::get<1>(s);
        int lines = std::get<2>(s);

        std::vector<float> input = {
            float(ev->hMax) / 20,
            float(ev->numHoles) / 200,
            float(ev->minMax) / 20,
            float(meaned(ev->colHeights, 10)) / 20,
            float(meaned(ev->deltaColHeights, 9)) / 20,
            float(lines) / float(4.0),
        };

        std::vector<int> shape = {int(input.size())};
        mlx::core::array inputArray = mlx::core::array(input.data(), shape, float32);

        return inputArray;
    }
    std::vector<array> batchStateToArray(std::vector<tetrisState> states) {
        std::vector<array> inputs;
        for (const auto& state : states) {
            inputs.push_back(stateToArray(state));
        }
        return inputs;
    }

    std::vector<array> batchForward(std::vector<tetrisState> states) {
        std::vector<array> inputs;
        for (const auto& state : states) {
            inputs.push_back(stateToArray(state));
        }
        return batchForward(inputs);
    }
    std::vector<array> batchForward(std::vector<array> batchInput) {
        std::vector<array> batchOutput;
        for (const array& input : batchInput) {
            // batchOutput.push_back(this->ml->forward(input));
            batchOutput.push_back(
                // generalizedForward(input, this->ml->params)
                generalizedForward(input, this->ml->params)
            );
        }
        return batchOutput;
    }
    array generalizedForward(const array& x, const std::vector<array> params);
};

// void train();
// bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, MultiLayer ml, unsigned int index, bool userMode, block** BASIC_BLOCKS);
void printArray(array a);

#endif /* NN_hpp */
