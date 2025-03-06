// #include <random>
#include <algorithm>
#include "tetrisrandom.hpp"

#include "tetrisrandom.hpp"
#include <algorithm>

block* TetrisRandom::randomBlock() {
    if (currentPos >= currentBag.size()) {
        currentBag = nextBag;
        currentPos = 0;

        // Prepare next bag
        std::shuffle(nextBag.begin(), nextBag.end(), gen);
    }

    int index = currentBag[currentPos++];
    return BASIC_BLOCKS[index];
}
