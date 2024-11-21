#include "game.h"
#include <_stdlib.h>
#include <assert.h>
#include "mlx/array.h"
#include <cstdlib>
#include <vector>
#include "GenNN.hpp"

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
