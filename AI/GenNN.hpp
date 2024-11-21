#ifndef GenNN_hpp
#define GenNN_hpp
// #include <stdio.h>
#include "game.h"
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/random.h"
// #include <cstddef>
#include <mlx/mlx.h>
#include <vector>
#include "assert.h"
#include "NN.hpp"

using namespace mlx::core;
// a tuple representing the env variables, the combination it came from and the lines cleared
typedef std::tuple<evars *, bestc, int> tetrisState;
class NNIndividual {
    public:
    MultiLayer *mlp = nullptr;
    unsigned int id;
    unsigned int score = 0;

    NNIndividual(int input_size, std::vector<int> hidden_sizes, unsigned int id) {
        this->mlp = new MultiLayer(input_size, hidden_sizes);
        this->id = id;
    }

    void updateParams(const std::vector<array>& params) {
        this->mlp->update(params);
    }
};
class GeneticNN {
    public:
    std::vector<NNIndividual> population;
    unsigned int count;
    std::vector<int> hidden_sizes;

    unsigned int populationID = 0;

    GeneticNN(unsigned int count, int input_size, std::vector<int> hidden_sizes) {
        this->hidden_sizes = hidden_sizes;
        this->count = count;
        for (int i = 0; i < count; i ++) {
            population.push_back(NNIndividual(input_size, hidden_sizes, i));
        }
    }
    void setResult(unsigned int id, unsigned int score) {
        population[id].score = score;
    }
    void udpatePopulation() {
        printf("UPDATING POPULATION \n");
        std::sort(population.begin(), population.end(), [](NNIndividual a, NNIndividual b) {
            return a.score > b.score;
        });

        for (int i = this->count / 2; i < this->count; i++) {
            int parent1 = randomIntBetween(0, -1 +this->count / 2);
            int parent2 = randomIntBetween(0, -1 +this->count / 2);
            printf("Breeding %d and %d and replacing %d with score %d\n", parent1, parent2, i, population[i].score);

            breed(parent1, parent2, i);
        }

        for (int i = 0; i < this->count; i++) {
            population[i].score = 0;
        }
        populationID += 1;
    }

    void breed(int parent1, int parent2, int child) {
        std::vector<array> newParams;
        for (int i = 0; i < population[child].mlp->params.size(); i++) {
            double r1 = randomProba();
            if (randomProba() > 0.05) {
                newParams.push_back(population[parent1].mlp->params[i]* r1 + population[parent2].mlp->params[i] * (1-r1));
            } else {
                printf("choosing random params\n");
                array newParam = random::uniform(-1, 1, population[parent1].mlp->params[i].shape());
                newParams.push_back(newParam);
            }
        }
        population[child].mlp->update(newParams);
    }

    bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, bool userMode, block** BASIC_BLOCKS);

    std::vector<array> batchForward(std::vector<tetrisState> states, int index) {
        std::vector<array> inputs;
        for (const auto& state : states) {
            inputs.push_back(stateToArray(state));
        }
        return batchForward(inputs, index);
    }
    std::vector<array> batchForward(std::vector<array> batchInput, int index) {
        std::vector<array> batchOutput;
        for (const array& input : batchInput) {
            // batchOutput.push_back(this->ml->forward(input));
            batchOutput.push_back(
                // generalizedForward(input, this->ml->params)
                generalizedForward(input, this->population[index].mlp->params)
            );
        }
        return batchOutput;
    }

    bestc act(std::vector<tetrisState>& possibleStates, int index);
};
#endif
