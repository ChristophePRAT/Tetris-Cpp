#include "heuristic.hpp"
#include "NN.h"
#include "game.h"
#include "tetrisrandom.hpp"

double heuristic(int linesCleared, evars* e) {
    double meanColHeights = 0;
    double meanDeltaColHeights = 0;

    for (int i = 0; i < 10; i++) {
        meanColHeights += double(e->colHeights[i]);
        if (i != 9) {
            meanDeltaColHeights += double(e->deltaColHeights[i]);
        }
    }
    // meanColHeights /= 10;
    // meanDeltaColHeights /= 9;

    return meanColHeights * (-0.510066) + double(linesCleared) * 0.760666 + double(e->numHoles) * (-0.35663) + meanDeltaColHeights * (-0.184483);
}

bestc bestFromHeuristic(mat *m, block s, evars* e) {
    double max = std::numeric_limits<double>::lowest();
    std::vector<tetrisState> st = possibleStates(*m, s, e);

    bestc best = {
        .col = -1,
        .shapeN = -1
    };

    for (const auto& state: st) {
        evars *ef = state.ev;
        int linesCleared = state.linesCleared;
        double score = heuristic(linesCleared, ef);
        if (score > max) {
            max = score;
            best = state.pos;
        }
        free(ef->colHeights);
        free(ef->deltaColHeights);
        free(ef);
    }
    return best;
}
bool heuristicTickCallBack(mat *m, block *s, block *nextBl, unsigned int *score, unsigned int* linesCleared, block **BASIC_BLOCKS, evars *e, TetrisRandom& tetrisRand) {
    int down = downShape(*m, s);

    // If the shape is at the bottom
    if (down == -1) {
        computeDownPos(*m, s);

        int numCleared = pushToMat(m, *s);
        *score += 150 * numCleared + 50;
        *linesCleared += numCleared;
        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, tetrisRand.randomBlock());

        bestc compo = bestFromHeuristic(m, *s,e);
        if (compo.shapeN == -1) {
            return false;
        }
        s->position[1] = compo.col;
        s->currentShape = compo.shapeN;

        computeDownPos(*m, s);

        bool canInsert = canInsertShape(*m, *s);

        return canInsert;
    }
    return true;
}

void supafastOneHeuristic(block** BASIC_BLOCKS, TetrisRandom& tetrisRand) {
    mat* m = createMat(20, 10);
    block* s = emptyShape();
    block* sA = tetrisRand.randomBlock();
    copyBlock(s, sA);
    computeDownPos(*m, s);

    block* nextBlock = emptyShape();
    sA = tetrisRand.randomBlock();
    copyBlock(nextBlock, sA);

    evars* envVars = initVars(*m);

    bool gameOn = true;

    unsigned int score = 0;
    unsigned int linesCleared = 0;

    while (gameOn) {
        gameOn = heuristicTickCallBack(m, s, nextBlock, &score, &linesCleared, BASIC_BLOCKS, envVars, tetrisRand);
    }
    printf("Lines Cleared: %u\n", linesCleared);
}
