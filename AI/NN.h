//
//  NN.h
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#ifndef NN_h
#define NN_h

#include "mlx/mlx.h"
// #include <stdio.h>
#include "game.h"
#include <cstddef>
#include <vector>
#include "assert.h"
const int NUM_WEIGHTS = 6;
const int NUM_LAYERS = 2;


using namespace mlx::core;
// a tuple representing the env variables, the combination it came from and the lines cleared
typedef std::tuple<evars *, bestc, int> tetrisState;
array stateToArray(tetrisState s);


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

    void update_parameters(const std::vector<array>& grads, double lr) {
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
        // float epsilon;
        float eps_decay;
        float eps_min;
        MultiLayer* ml = nullptr;
        std::vector<array> mem = {};
        unsigned int memCapacity;
        int step = 0;

        float beta1 = 0.9;    // Exponential decay rate for first moment
        float beta2 = 0.999;  // Exponential decay rate for second moment
        float epsilon = 1e-8; // Small constant to prevent division by zero

    DQN(int input_size, unsigned int memCapacity) {
        // this->epsilon = 0.2;
        this->eps_decay = 0.99;
        this->eps_min = 0.001;
        this->ml = new MultiLayer(input_size, {1});
        this->memCapacity = memCapacity;
    }
    bestc act(std::vector<tetrisState>& possibleStates);

    std::tuple<std::vector<array>, std::vector<array>> gatherTrainingData(std::vector<tetrisState> memory) {
        std::vector<array> inputs = batchStateToArray(memory);
        std::vector<array> targets;
        return std::make_tuple(inputs, targets);
    }
    void trainNN(unsigned int linesCleared) {
        step += 1;
        unsigned int epochs = 5;
        for (int i = 0; i < epochs; i++) {
            printf("Epoch %d: \n", i);
            trainWithBatch(mem, batchHeuristic(mem), linesCleared);
        }
    }

    bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, block** BASIC_BLOCKS);

    private:
    // Adam optimizer state
    std::vector<array> m_t; // First moment estimate
    std::vector<array> v_t; // Second moment estimate
    int t = 0;
    void initializeAdam();
    void adamUpdate(const std::vector<array>& grads, double lr);

    std::vector<array> batchHeuristic(std::vector<array>);

    void train(std::vector<array> states, std::vector<array> yTruth, unsigned int linesCleared);
    void trainWithBatch(std::vector<array> states, std::vector<array> yTruth, unsigned int linesCleared);

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

void printArray(array a);
std::vector<tetrisState> possibleStates(mat m, block s, evars* previousEvars);
array generalizedForward(const array& x, const std::vector<array> params);

#endif /* NN_h */
