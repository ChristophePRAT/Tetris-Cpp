#include "game.h"
#include <_stdlib.h>
#include <assert.h>
#include "mlx/array.h"
#include "mlx/dtype.h"
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>
#include "GenNN.hpp"
#include "../Helpers/loader.hpp"
#include <ctime>
#include <sstream>
#include <iomanip>

std::string getCurrentDateTime() {
    // Get the current time
    std::time_t now = std::time(nullptr);
    // Convert to local time structure
    std::tm *localTime = std::localtime(&now);

    // Create a string stream to format the date and time
    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y-%m-%d_%H:%M"); // Format: YYYY-MM-DD HH:MM:SS

    return oss.str();
}

void GeneticNN::udpatePopulation() {
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
    saveGen(*this);
}
void GeneticNN::loadPrevious(int genID, std::string date) {
    loadGen(*this, genID, date);
}

void GeneticNN::breed(int parent1, int parent2, int child) {
    std::vector<array> newParams;
    for (int i = 0; i < population[child].mlp->params.size(); i++) {
        double r1 = randomProba();
        if (randomProba() > 0.05) {
            newParams.push_back(population[parent1].mlp->params[i]* r1 + population[parent2].mlp->params[i] * (1-r1));
        } else {
            printf("choosing random params\n");
            float meanArr = mean(population[parent1].mlp->params[i]).item<float>();
            array newParam = population[parent1].mlp->params[i] + random::uniform(-meanArr, meanArr, population[parent1].mlp->params[i].shape());

            // array newParam = random::uniform(-1, 1, population[parent1].mlp->params[i].shape());
            newParams.push_back(newParam);
        }
    }
    population[child].mlp->update(newParams);

}

bestc GeneticNN::act(std::vector<tetrisState>& possibleStates, int index) {
    float max_rating = -std::numeric_limits<float>::infinity();

    bestc best_action = {
        .col = -1,
        .shapeN = -1
    };
    int best_index = -1;
    std::vector<array> ratings = batchForward(possibleStates, index);

    for (int i = 0; i < ratings.size(); i++) {
        float rating = ratings[i].item<float>();
        if (rating > max_rating) {

            max_rating = rating;
            best_action = std::get<1>(possibleStates[i]);
            best_index = i;

            evars* e = std::get<0>(possibleStates[i]);
            free(e->colHeights);
            free(e->deltaColHeights);
            free(e);
        }
    }
    return best_action;
}

std::vector<std::tuple<array, bestc>> possibleBoards(mat m, block s, evars* previousEvars) {

    std::vector<std::tuple<array, bestc>> states;

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
                std::vector<float> inp;
                for (int j = 0; j < m.cols; j++) {
                    inp.push_back(float(ev->colHeights[j])/20);
                }
                inp.push_back(numCleared);
                inp.push_back(ev->numHoles);

                std::vector<int> shape = {int(inp.size())};

                array input = array(inp.data(), shape, float32);
                states.push_back(std::make_tuple(input, config));

                free(ev->colHeights);
                free(ev->deltaColHeights);
                free(ev);
                freeMat(preview);
            }
        }
    }
    return states;
}

bestc GeneticNN::act2(std::vector<std::tuple<array, bestc>> possibleBoards, int index) {
    float max_rating = -std::numeric_limits<float>::infinity();

    bestc best_action = {
        .col = -1,
        .shapeN = -1
    };
    int best_index = -1;
    std::vector<array> inputArrays;

    for (int i = 0; i < possibleBoards.size(); i++) {
        inputArrays.push_back(std::get<0>(possibleBoards[i]));
    }
    std::vector<array> ratings = batchForward(inputArrays, index);

    for (int i = 0; i < ratings.size(); i++) {
        float rating = ratings[i].item<float>();
        if (rating > max_rating) {

            max_rating = rating;
            best_action = std::get<1>(possibleBoards[i]);
            best_index = i;
        }
    }
    return best_action;
}


bool GeneticNN::tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, bool userMode, block** BASIC_BLOCKS) {
    int down = downShape(*m, s);

    // If the shape is at the bottom
    if (down == -1) {
        computeDownPos(*m, s);

        int numCleared = pushToMat(m, *s);
        *score += 200 * pow(numCleared, 2) + 10;
        *linesCleared += numCleared;

        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, randomBlock(BASIC_BLOCKS));

         std::vector<tetrisState> maybeStates = possibleStates(*m, *s, e);

         bestc compo = this->act(maybeStates, index);

        //std::vector<std::tuple<array, bestc>> maybeStates = possibleBoards(*m, *s, e);

        //bestc compo = this->act2(maybeStates, index);

        if (compo.shapeN == -1) { return false; }

        int nextPosX = compo.col;

        assert(nextPosX >= 0 && nextPosX < m->cols + 2);

        s->position[1] = nextPosX;
        s->currentShape = compo.shapeN;
        computeDownPos(*m, s);

        bool canInsert = canInsertShape(*m, *s);

        return canInsert;
    }
    return true;
}
