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
#include "mlx/transforms.h"
#include "tetrisrandom.hpp"

using namespace mlx::core;

// a struct representing the env variables, the combination it came from and the lines cleared
typedef struct tetrisState {
    evars* ev;
    bestc pos;
    int linesCleared;
} tetrisState;

array stateToArray(tetrisState s);
array leakyRelu(const array& input);
std::vector<tetrisState> possibleStates(mat m, block s, evars* previousEvars);
array generalizedForward(const array& x, const std::vector<array> params);
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
            this->weights = new array(random::uniform(-k, k, {output_dims, input_dims}));

            this->bias = new array(random::uniform(-k, k, {output_dims}));

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




#endif /* NN_h */
