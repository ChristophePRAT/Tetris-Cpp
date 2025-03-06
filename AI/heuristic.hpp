#ifndef HEURISTIC_HPP
#define HEURISTIC_HPP

#include "tetrisrandom.hpp"
#include "game.h"

bool heuristicTickCallBack(mat *m, block *s, block *nextBl, unsigned int *score, unsigned int* linesCleared, block **BASIC_BLOCKS, evars *e, TetrisRandom& tetrisRand);

#endif
