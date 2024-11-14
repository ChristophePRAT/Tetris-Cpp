//
//  agent.cpp
//  tutorial
//
//  Created by Christophe Prat on 26/04/2024.
//

#include "agent.h"
#include "game.h"
#include <cstdio>
#import <stdlib.h>
#include "cmath"
// #include <stdlib.h>
// #include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cassert>
// #include <iostream>


const int NUM_WEIGHTS = 6;

population* initializePopulation(int numIndi) {
    population* g = (population*)malloc(sizeof(population));
    g->id = 0;
    g->numIndividuals = numIndi;
//    g->learningRate = 0.2;
    g->individuals = (indi*)malloc(numIndi * sizeof(indi));

    for (int i = 0; i < g->numIndividuals; i++) {
        g->individuals[i].weights = (double*)malloc(NUM_WEIGHTS * sizeof(double));
        g->individuals[i].id = i;
        for (int j = 0; j < NUM_WEIGHTS; j++) {
            g->individuals[i].weights[j] = 2*randomProba() - 1;
        }
    }
    return g;
}

void freepopulation(population* g) {
    for (int i = 0; i < g->numIndividuals; i++) {
        free(g->individuals[i].weights);
    }
    free(g->individuals);
    free(g);
}

void printWeights(double* w) {
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        printf(" %f ", w[i]);
    }
    printf("\n");
}

void printpopulation(population* g) {
    printf("Model weights: ");
    printf("\n");
    for (int i = 0; i < g->numIndividuals; i++) {
        printf("Individual %d: ", g->individuals[i].id);
        printWeights(g->individuals[i].weights);
    }
}
// Sort the individuals by score
void sort(population* g, int* scores) {
    for (int i = 0; i < g->numIndividuals; i++) {
        for (int j = i + 1; j < g->numIndividuals; j++) {
            if (scores[i] > scores[j]) {
                indi temp = g->individuals[i];
                g->individuals[i] = g->individuals[j];
                g->individuals[j] = temp;
                int tempScore = scores[i];
                scores[i] = scores[j];
                scores[j] = tempScore;
            }
        }
    }
    assert(scores[0] <= scores[1]);
}

void pregnancy(indi parentOne, indi parentTwo, double* childWeights, double alpha, double learningRate) {
//    child.weights = (double*)malloc(NUM_WEIGHTS * sizeof(double));
    for (int i = 0; i < NUM_WEIGHTS; i++) {
        if (randomProba() < learningRate) {
            childWeights[i] = 2 * alpha * (randomProba() - 0.5);
        } else {
            childWeights[i] = randomProba() > 0.5 ? parentOne.weights[i] : parentTwo.weights[i];
        }
    }
//    return child;
}

indi randomIndiIn(int a, int b, population* g) {
    int i = randomIntBetween(a, b);
    assert(i < g->numIndividuals);
    assert(i >= g->numIndividuals / 2);
    return g->individuals[i];
}


unsigned int mutatepopulation(population* g, int* scores) {
    g->id = g->id + 1;
    sort(g, scores);

    // Ensure that the sort is correct
    assert(scores[0] <= scores[1]);

    int upper = g->numIndividuals;
    int lower = g->numIndividuals / 2;

    printf("\n\nMUTATING POPULATION\n");
    printf("Population ID: %d\n", g->id);
    printf("Previous best score: %d\n", scores[g->numIndividuals - 1]);
    printWeights(g->individuals[g->numIndividuals - 1].weights);
    printf("\n");


    for (int i = 0; i < g->numIndividuals/2; i++) {
//        assert(g->individuals[i].id != i);
        pregnancy(randomIndiIn(lower, upper, g), randomIndiIn(lower, upper, g), g->individuals[i].weights, 2, 0.05);
    }
    return scores[g->numIndividuals - 1];
}



double* getColFromBotDecision(mat m, block s, double* preferences, evars* previousEvars) {
//    int lastValidIndexForBlock = m.cols - blockWidth(s);

    double meanedColHeights = meaned(previousEvars->colHeights, m.cols);
    double meanedDeltaColHeights = meaned(previousEvars->deltaColHeights, m.cols);

    int bestCol = 0;

    double bestScore = previewScore(m, s, preferences, previousEvars, 0, meanedColHeights, meanedDeltaColHeights);
    double* cs = (double*) malloc(2 * sizeof(double));
    for (int i = 1; i < m.cols; i++) {
//        if (i > lastValidIndexForBlock + 2) {
//            break;
//        }
//        s.position[1] = i;
//        mat* preview = previewMatIfPushDown(&m, s);
//        if (preview == NULL) {
//            continue;
//        }
//        evars* ev = retrieveEvars(*preview, previousEvars);
//        double score = preferences[0] * ev->hMax +
//        preferences[1] * ev->numHoles +
//        preferences[2] * meanedColHeights +
//        preferences[3] * meanedDeltaColHeights;
        double score = previewScore(m, s, preferences, previousEvars, i, meanedColHeights, meanedDeltaColHeights);
        if (score > bestScore) {
            bestScore = score;
            bestCol = i;
        }
//        freeMat(preview);
//        free(ev->colHeights);
//        free(ev->deltaColHeights);
//        free(ev);
    }

    cs[0] = bestCol;
    cs[1] = bestScore;

    return cs;
}

bestc theFinestDecision(mat m, block s, double* preferences, evars* previousEvars) {
    int bestCol = 0;
    int bestScore = -100000;
    int bestShape = -1;
    for (int i = 0; i < s.numberOfShapes; i++) {
        s.currentShape = i;
        double* cs = getColFromBotDecision(m, s, preferences, previousEvars);
        if (cs[1] > bestScore) {
            bestScore = cs[1];
            bestCol = cs[0];
            bestShape = i;
        }
        free(cs);
    }
    return {
        .col = bestCol,
        .shapeN = bestShape
    };
}


bool tickCallback(mat* m, block* s, block* nextBl, evars* e, population* g, unsigned int* score, unsigned int* linesCleared, unsigned int index, bool userMode, block** BASIC_BLOCKS) {
    int down = downShape(*m, s);

    *score += 1;

    if (down == -1) {
        int numCleared = pushToMat(m, *s);
        *score += 200 * pow(numCleared, 2);
        *linesCleared = *linesCleared + numCleared;

        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, randomBlock(BASIC_BLOCKS));
        if (userMode) {
            computeDownPos(*m, s);
        }

        if (!userMode) {
            bestc compo = theFinestDecision(*m, *s, g->individuals[index].weights, e);
            if (compo.shapeN == -1) { return false; }
            int nextPosX = compo.col;
            assert(nextPosX >= 0 && nextPosX < m->cols + 2);
            s->position[1] = nextPosX;
            s->currentShape = compo.shapeN;
            computeDownPos(*m, s);
        }
        return canInsertShape(*m, *s);
    }
    return true;
}
