//
//  agent.hpp
//  tutorial
//
//  Created by Christophe Prat on 26/04/2024.
//

#ifndef agent_hpp
#define agent_hpp

#include "game.h"

typedef struct Individual {
  int id;
  double* weights;
} indi;
typedef struct population {
  indi* individuals;
  int numIndividuals;
  int numWeights;
    int id;
} population;

unsigned int mutatepopulation(population* g, int* scores);
void printpopulation(population* g);
population* initializePopulation(int numIndi);
void freepopulation(population* g);
bool tickCallback(mat* m, block* s, block* nextBl, evars* e, population* g, unsigned int* score, unsigned int index, bool userMode, block** BASIC_BLOCKS);
block* randomBlock(block** BASIC_BLOCKS);
void reset(unsigned int* score, mat* m, block* s, block* nextBlock, block** BASIC_BLOCKS, evars* envVars);
#endif /* agent_hpp */
