#include "NN.h"
// #include "agent.h"
#include "game.h"
#include <_stdlib.h>
#include <algorithm>
#include <assert.h>
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>
#include "GenNN.hpp"
#include "loader.hpp"
#include "mlx/ops.h"
#include "mlx/stream.h"
// #include "mlx/random.h"
#include <ctime>
#include <sstream>
#include <iomanip>
#include <stdbool.h>
// #include <thread>
#include <chrono>
#include <random>
// #include "tetrisrandom.hpp"

#define THREADS_COUNT 8

using std::chrono::high_resolution_clock;

std::string getCurrentDateTime() {
    // Get the current time
    std::time_t now = std::time(nullptr);
    // Convert to local time structure
    std::tm *localTime = std::localtime(&now);

    // Create a string stream to format the date and time
    std::ostringstream oss;
    oss << std::put_time(localTime, "%d-%m_%H-%M"); // Format: YYYY-MM-DD HH:MM:SS

    return oss.str();
}

// void GeneticNN::udpatePopulation() {
//     printf("UPDATING POPULATION \n");
//     std::sort(population.begin(), population.end(), [](NNIndividual a, NNIndividual b) {
//         return a.score > b.score;
//     });

//     // printf("Previous gen best indiv score: %d\n", this->population[0].score);
//     for (int i = this->count / 2; i < this->count; i++) {
//         int parent1 = randomIntBetween(0, -1 +this->count / 2);
//         int parent2 = randomIntBetween(0, -1 +this->count / 2);
//         // printf("Breeding %d and %d and replacing %d with score %d\n", parent1, parent2, i, population[i].score);

//         breed(parent1, parent2, i);
//     }

//     for (int i = 0; i < this->count; i++) {
//         population[i].score = 0;
//     }
//     populationID += 1;
//     saveGen(*this);
//     this->seed = time(NULL);
//     srand(this->seed);
// }

void GeneticNN::udpatePopulation() {
    int nWorst = count * 0.3;
    int nSelection = count * 0.1;

    std::sort(population.begin(), population.end(), [](NNIndividual a, NNIndividual b) {
        return a.score < b.score;
    });

    printf("------------------------------------------------------------------------\n");
    for (int i = 0; i < nWorst; i++) {
        std::vector<int> potentialParents;

        for (int j = 0; j < nSelection; j++) {
            potentialParents.push_back(generateRandomNumber(0, this->count - 1));
        }

        std::sort(potentialParents.begin(), potentialParents.end(), [this](int a, int b) {
            return population[a].score > population[b].score;
        });

        breed(potentialParents[0], potentialParents[1], i);
        printf("| %s ❤️ %s => %s |\n", population[potentialParents[0]].name.c_str(), population[potentialParents[1]].name.c_str(), population[i].name.c_str());
    }

    printf("------------------------------------------------------------------------\n");
    for (int s = 0; s < this->count; s++) {
        population[s].score = 0;
    }

    populationID += 1;
    saveGen(*this);

    this->seed = seed_gen(); // update the seed for the next game
}

// void GeneticNN::breed(int parent1, int parent2, int child) {
//     std::vector<array> newParams;
//     for (int i = 0; i < population[child].mlp->params.size(); i++) {
//         unsigned int score1 = population[parent1].score;
//         unsigned int score2 = population[parent2].score;
//         newParams.push_back(
//             (population[parent1].mlp->params[i] * score1 + population[parent2].mlp->params[i] * score2) / (score1 + score2 + 1e-10)
//         );
//     }
//     population[child].mlp->update(newParams);
// }

void GeneticNN::loadPrevious(int genID, std::string date) {
    loadGen(*this, genID, date);
    for (int i = 0; i < count; i++) {
        population[i].score = 0;
        population[i].name = NAMES[i % count];
    }
}

void GeneticNN::breed(int parent1, int parent2, int child) {
    std::vector<array> newParams;

    // On effectue une mutation sur l'ensemble des paramètres
    for (int i = 0; i < population[child].mlp->params.size(); i++) {
        double r1 = randomProba();
        if (randomProba() > 0.05) {
            // On prend une moyenne pondérée aléatoirement des deux parents
            array weightedMean = population[parent1].mlp->params[i] * r1 + population[parent2].mlp->params[i] * (1-r1);
            mlx::core::eval(weightedMean);

            array mutation = mlx::core::random::normal(weightedMean.shape(), 0, mean(weightedMean).item<float>()/3);
            mlx::core::eval(mutation);

            newParams.push_back(mutation + weightedMean);
        } else {
            // 5% de chance de créer un individu complètement aléatoire
            array newParam1 = mlx::core::random::normal(population[parent1].mlp->params[i].shape(), 0, mean(population[parent1].mlp->params[i]).item<float>()/2);
            array newParam2 = mlx::core::random::normal(population[parent2].mlp->params[i].shape(), 0, mean(population[parent2].mlp->params[i]).item<float>()/2);

            mlx::core::eval({newParam1, newParam2});
            newParams.push_back(newParam1 + newParam2);
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
            best_action = possibleStates[i].pos;
            best_index = i;

        }
    }
    for (auto& state : possibleStates) {
        evars* e = state.ev;
        if (e) {
            free(e->colHeights);
            free(e->deltaColHeights);
            free(e);
        }
    }
    return best_action;
}

std::vector<std::tuple<array, bestc>> possibleBoardsEachCol(mat m, block s, evars* previousEvars) {

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
/*
std::vector<tetrisState> possibleStatesWithNextBlock(mat m, block s, block nexts, evars* previousEvars) {
    // number of cleared ; configuration
    std::vector<tetrisState> statesMoveOne;

    std::vector<tetrisState> finalStates;

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
                statesMoveOne.push_back(std::tuple<evars *, bestc, int>(ev, config, numCleared));
                for (int ip = 0; ip < m.cols; ip++) {
                    for (int rp = 0; rp < s.numberOfShapes; rp++) {
                        nexts.currentShape = rp;
                        int numCleared2 = 0;
                        nexts.position[1] = ip;

                        mat *preview2 = previewMatIfPushDown(preview, nexts, &numCleared2);

                        if (preview2 && preview2 != NULL) {
                            evars* ev2 = retrieveEvars(*preview2, ev);
                            bestc config2 = {
                                .col = ip,
                                .shapeN = rp
                            };
                            finalStates.push_back(std::make_tuple(ev2, config, numCleared + numCleared2));
                            freeMat(preview2);
                        }
                    }
                }
                freeMat(preview);
            }
        }
    }

    return finalStates;
}
 */
bestc GeneticNN::actWithNextBlock(mat m, block s, block ns, evars *prev, int index) {
    bestc best_action = {
        .col = -1,
        .shapeN = -1
    };
    float max_rating = -std::numeric_limits<float>::infinity();
    float max_rating_middle = -std::numeric_limits<float>::infinity();

    int numChecked = 0;

    for (int i = 0; i < m.cols; i++) {
        for (int r = 0; r < s.numberOfShapes; r++) {
            s.currentShape = r;
            int numCleared = 0;
            s.position[1] = i;

            mat *preview = previewMatIfPushDown(&m, s, &numCleared);

            if (preview && preview != NULL) {
                evars* ev = retrieveEvars(*preview, prev);
                bestc config = {
                    .col = i,
                    .shapeN = r
                };

                tetrisState midState = {.ev = ev, .pos = config, .linesCleared = numCleared};


                array midOutput = generalizedForward(stateToArray(midState), this->population[index].mlp->params);

                float midRating = midOutput.item<float>();

                if (midRating < max_rating_middle && randomProba() > midRating/(max_rating_middle*max_rating_middle)) {
                    free(ev->colHeights);
                    free(ev->deltaColHeights);
                    free(ev);
                    freeMat(preview);
                    continue;
                }

                for (int ip = 0; ip < m.cols; ip++) {
                    for (int rp = 0; rp < s.numberOfShapes; rp++) {
                        ns.currentShape = rp;
                        int numCleared2 = 0;
                        ns.position[1] = ip;

                        mat *preview2 = previewMatIfPushDown(preview, ns, &numCleared2);

                        if (preview2 && preview2 != NULL) {
                            evars* ev2 = retrieveEvars(*preview2, ev);
                            bestc config2 = {
                                .col = ip,
                                .shapeN = rp
                            };

                            tetrisState input = { .ev = ev2, .pos = config, .linesCleared = numCleared + numCleared2 };

                            array output = generalizedForward(stateToArray(input), this->population[index].mlp->params);

                            float rating = output.item<float>();

                            numChecked += 1;

                            if (rating > max_rating) {
                                max_rating = rating;
                                best_action = config;
                                max_rating_middle = midRating;
                            }

                            free(ev2->colHeights);
                            free(ev2->deltaColHeights);
                            free(ev2);
                            freeMat(preview2);
                        }
                    }
                }
                numChecked -= 1; // We would have checked the middle state anyway
                free(ev->colHeights);
                free(ev->deltaColHeights);
                free(ev);
                freeMat(preview);
            }
        }
    }
    // printf("Checked %d (additional) moves before playing\n", numChecked);
    return best_action;
}

bestc GeneticNN::actWithNextBlock2(mat m, block s, block ns, evars *prev, int index) {
    std::vector<tetrisState> ps = possibleStates(m, s, prev);
    std::vector<array> ratings = batchForward(ps, index);

    std::vector<std::tuple<array, tetrisState>> possibleBoards;
    assert(ps.size() == ratings.size());
    for (int i = 0; i < ps.size(); i++) {
        possibleBoards.push_back(std::make_tuple(ratings[i], ps[i]));
    }

    // sort by rating
    std::sort(possibleBoards.begin(), possibleBoards.end(), [](std::tuple<array, tetrisState> a, std::tuple<array, tetrisState> b) {
        return std::get<0>(a).item<float>() > std::get<0>(b).item<float>();
    });

    unsigned int maxCheck = 5;
    unsigned int numberToCheck = ps.size() > maxCheck ? maxCheck : ps.size();

    float max_rating = -std::numeric_limits<float>::infinity();
    int max_index = -1;
    bestc best_action = {
        .col = -1,
        .shapeN = -1
    };

    for (int i = 0; i < numberToCheck; i++) {
        tetrisState state = std::get<1>(possibleBoards[i]);

        evars* ev = state.ev;
        bestc config = state.pos;
        int numCleared = state.linesCleared;

        for (int ip = 0; ip < m.cols; ip++) {
            for (int rp = 0; rp < ns.numberOfShapes; rp++) {
                ns.currentShape = rp;
                int numCleared2 = 0;
                ns.position[1] = ip;

                mat *preview2 = previewMatIfPushDown(&m, ns, &numCleared2);

                if (preview2 && preview2 != NULL) {
                    evars* ev2 = retrieveEvars(*preview2, ev);
                    bestc config2 = {
                        .col = ip,
                        .shapeN = rp
                    };

                    tetrisState input = {
                        .ev = ev2,
                        .pos = config,
                        .linesCleared = numCleared + numCleared2
                    };

                    array output = generalizedForward(stateToArray(input), this->population[index].mlp->params);

                    float rating = output.item<float>();

                    if (rating > max_rating) {
                        max_rating = rating;
                        best_action = config;
                        max_index = i;
                    }

                    free(ev2->colHeights);
                    free(ev2->deltaColHeights);
                    free(ev2);
                    freeMat(preview2);
                }
            }
        }
    }
    for (int i = 0; i < ps.size(); i++) {
        evars* e = ps[i].ev;
        free(e->colHeights);
        free(e->deltaColHeights);
        free(e);
    }
    if (max_index > 0) {
        printf(".");
    }
    return best_action;
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


bool GeneticNN::tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, unsigned int* linesCleared, unsigned int index, block** BASIC_BLOCKS, TetrisRandom& tetRand, bool instant) {
    int down = 0;
    if (!instant) {
        down = downShape(*m, s);
    } else {
        s->position[0] = s->downPos;
        down = -1;
    }

    // If the shape is at the bottom
    if (down == -1) {
        computeDownPos(*m, s);

        int numCleared = pushToMat(m, *s);
        *score += 15 * numCleared + 2;
        *linesCleared += numCleared;

        updateEvars(*m, e);
        changeBlock(s, nextBl);

        block* sA = tetRand.randomBlock();
        copyBlock(nextBl, sA);

        bestc compo;

        // if (e->hMax > 16) {
        //     compo = actWithNextBlock(*m, *s, *nextBl, e, index);
            //
            // compo = actWithNextBlock2(*m, *s, *nextBl, e, index);
        // } else {
        std::vector<tetrisState> maybeStates = possibleStates(*m, *s, e);

        compo = this->act(maybeStates, index);
        // }


        //std::vector<std::tuple<array, bestc>> maybeStates = possibleBoardsEachCol(*m, *s, e);

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

void GeneticNN::supafastindiv(block** BASIC_BLOCKS, unsigned int index) {
    unsigned int indiSeed = this->seed;
    // std::mt19937 indiGen(indiSeed);

    TetrisRandom tetRand = TetrisRandom(indiSeed, BASIC_BLOCKS);

    bool gameOver = false;

    unsigned int linesCleared = 0;
    unsigned int score = 0;

    mat* m = createMat(20, 10);
    block* s = emptyShape();
    block* sA = tetRand.randomBlock();
    copyBlock(s, sA);
    computeDownPos(*m, s);

    block* nextBlock = emptyShape();
    sA = tetRand.randomBlock();
    copyBlock(nextBlock, sA);
    evars* envVars = initVars(*m);

    while (!gameOver) {
        gameOver = !tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, BASIC_BLOCKS, tetRand, true);
    }
    setResult(index, score, linesCleared);
    printf("%s (#%02d) - lines cleared = %d\n", this->population[index].name.c_str(), this->population[index].id, linesCleared);

    if (envVars) {
        free(envVars->colHeights);
        free(envVars->deltaColHeights);
        free(envVars);
    }

    freeMat(m);
    freeBlock(s);
    freeBlock(nextBlock);
}

void GeneticNN::batchSupafast(block** BASIC_BLOCKS, unsigned int first, unsigned int last) {

    for (int i = first; i < last; i++) {
        supafastindiv(BASIC_BLOCKS, i);
    }
}

void GeneticNN::supafast(block** BASIC_BLOCKS) {
    mlx::core::set_default_device(Device::DeviceType::gpu);
    // std::thread threads[THREADS_COUNT];

    int step = count / THREADS_COUNT;

    // Avoids one thread to have all of the best players, which would slow down the process
    std::shuffle(population.begin(), population.end(), std::default_random_engine(this->seed));

    for (int i = 0; i < THREADS_COUNT; i++) {
        // threads[i] = std::thread(&GeneticNN::batchSupafast, this, BASIC_BLOCKS, i * step, (i + 1) * step);
        batchSupafast(BASIC_BLOCKS, i*step, (i+1)*step);
    }

    // for (int i = 0; i < THREADS_COUNT; i++) {
    //     if (threads[i].joinable()) {
    //         threads[i].join();
    //     }
    // }
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    printf("Population #%d\n", this->populationID + 1);
    printf("Updating weights & biases\n");
    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    this->udpatePopulation();
    supafast(BASIC_BLOCKS);
}
