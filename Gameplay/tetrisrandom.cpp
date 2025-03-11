// #include <random>
#include <algorithm>
#include "tetrisrandom.hpp"
#include "tetrisrandom.hpp"
#include <algorithm>
#include <stdlib.h>
#include <cassert>

block* TetrisRandom::randomBlock() {
    assert(BASIC_BLOCKS != NULL);
    assert(BASIC_BLOCKS != nullptr);

    if (currentPos >= currentBag.size()) {
        currentBag = nextBag;
        currentPos = 0;

        // Prepare next bag
        std::shuffle(nextBag.begin(), nextBag.end(), gen);
    }

    int index = currentBag[currentPos++];
    return BASIC_BLOCKS[index];
}
