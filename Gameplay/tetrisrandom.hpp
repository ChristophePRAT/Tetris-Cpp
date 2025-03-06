#ifndef tetrisrandom_h
#define tetrisrandom_h

#include <random>
#include <game.h>
#include <vector>
#include <algorithm>

class TetrisRandom {
public:
    TetrisRandom(int seed = std::random_device{}(), block** basicBlocks = nullptr)
        : gen(seed)
        , BASIC_BLOCKS(basicBlocks)
        , currentBag(7)
        , nextBag(7)
    {
        // Initialize both bags
        for (int i = 0; i < 7; ++i) {
            currentBag[i] = i;
            nextBag[i] = i;
        }

        // Shuffle both bags initially
        std::shuffle(currentBag.begin(), currentBag.end(), gen);
        std::shuffle(nextBag.begin(), nextBag.end(), gen);
        currentPos = 0;
    }

    block* randomBlock();

private:
    std::mt19937 gen;
    std::vector<int> currentBag;
    std::vector<int> nextBag;
    size_t currentPos;
    block** BASIC_BLOCKS;
};

#endif
