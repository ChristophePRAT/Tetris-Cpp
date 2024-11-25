#ifndef GenNN_hpp
#define GenNN_hpp
#include "game.h"
#include "mlx/array.h"
#include <mlx/mlx.h>
#include <vector>
#include "assert.h"
#include "NN.hpp"

std::string getCurrentDateTime();

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
    std::string createDate;
    unsigned int populationID = 0;

    GeneticNN(unsigned int count, int input_size, std::vector<int> hidden_sizes) {
        this->hidden_sizes = hidden_sizes;
        this->count = count;
        this->createDate = getCurrentDateTime();

        mkdir(createDate.c_str(), 0777);

        for (int i = 0; i < count; i ++) {
            population.push_back(NNIndividual(input_size, hidden_sizes, i));
        }
    }
    void setResult(unsigned int id, unsigned int score) {
        population[id].score = score;
    }
    void udpatePopulation();

    void breed(int parent1, int parent2, int child);

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
    bestc act2(std::vector<std::tuple<array, bestc>> possibleBoards, int index);
    void loadPrevious(int genID, std::string date);
};
#endif
