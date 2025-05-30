#ifndef rl_hpp
#define rl_hpp

#include <vector>
#include <array>
#include <tuple>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include "NN.h"

class RL {
    public:
        float discount;
        float explorationRate = 0.3;
        float expDecay;
        float expMin;
        MultiLayer* ml = nullptr;
        std::vector<array> mem = {};
        unsigned int memCapacity;
        int step = 0;

        float beta1 = 0.9;    // Exponential decay rate for first moment
        float beta2 = 0.999;  // Exponential decay rate for second moment
        float epsilon = 1e-8; // Small constant to prevent division by zero

    RL(int input_size, unsigned int memCapacity) {
        this->expDecay = 0.99;
        this->expMin = 0.001;
        this->ml = new MultiLayer(input_size, {1});
        this->memCapacity = memCapacity;
    }
    bestc act(std::vector<tetrisState>& possibleStates);
    bestc actWithMat(std::vector<tetrisState>& possibleStates);

    std::tuple<std::vector<array>, std::vector<array>> gatherTrainingData(std::vector<tetrisState> memory) {
        std::vector<array> inputs = batchStateToArray(memory);
        std::vector<array> targets;
        return std::make_tuple(inputs, targets);
    }
    void trainNN(unsigned int linesCleared) {
        step += 1;
        unsigned int epochs = 1;
        for (int i = 0; i < epochs; i++) {
            printf("Epoch %d: \n", i);
            trainWithBatch(mem, batchHeuristic(mem), linesCleared);
        }
    }

    bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, block** BASIC_BLOCKS, TetrisRandom& tetrisRand, bool instant);

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
        inputs.reserve(states.size());
        for (const auto& state : states) {
            inputs.push_back(stateToArray(state));
        }
        mlx::core::eval(inputs);
        return batchForward(inputs);
    }
    std::vector<array> batchForward(std::vector<array> batchInput) {
        std::vector<array> batchOutput;
        batchOutput.reserve(batchInput.size());
        for (const array& input : batchInput) {
            // batchOutput.push_back(this->ml->forward(input));
            batchOutput.push_back(
                // generalizedForward(input, this->ml->params)
                generalizedForward(input, this->ml->params)
            );
        }
        mlx::core::eval(batchOutput);
        return batchOutput;
    }
    array generalizedForward(const array& x, const std::vector<array> params);
};

void printArray(array a);
std::vector<tetrisState> possibleStates(mat m, block s, evars* previousEvars);
array generalizedForward(const array& x, const std::vector<array> params);

#endif
