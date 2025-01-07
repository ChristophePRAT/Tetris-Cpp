//
//  game.h
//  tutorial
//
//  Created by Christophe Prat on 30/03/2024.
//

#ifndef game_h
#define game_h

// #include <stdio.h>
// #include <stdlib.h>
//struct Matrix;

struct Matrix {
    int rows;
    int cols;
    int** data;
};

typedef struct Matrix mat;

struct Block {
    int*** shape; // constante
    int currentShape;
    int numberOfShapes; // constante
    int position[2];
    int downPos;
};
typedef struct Block block;


typedef struct EnvVars {
    int* colHeights;
    int* deltaColHeights; // |h(p+1) - h(p)
    int hMax; // max hp
    int numHoles; // number of holes
    int minMax; // Difference of height between the tallest and the smallest col
} evars;

typedef struct BestConfig {
    int col;
    int shapeN;
} bestc;

extern "C" block* emptyShape(void);
void initBlock(block* b, int shape[][4][4], int numberOfShapes);
mat* createMat(int rows, int cols);
block* createShape(int data[4][4], int position[2]);

void moveRLShape(mat* m, block* s, int direction);
int downShape(mat m, block* s);
int pushToMat(mat* m, block s);
void copyBlock(block* dest, block* src);
void freeBlock(block* b);
void rotateShape(mat m, block* s);
bool canInsertShape(mat m, block s);
int fullDrop(mat m, block s, bool pre);
//void updateEVFromGame(mat m, block s, evars* previousEvars);
evars* initVars(mat m);
void updateEvars(mat m, evars* previousEvars);
//mat* previewMatIfPushDown(mat* m, block s);
int theFinestDecision(mat m, block s, int* preferences, evars* previousEvars);
void freeMat(mat* m);
bestc theFinestDecision(mat m, block s, double* preferences, evars* previousEvars);
double randomProba();
void computeDownPos(mat m, block *s);
int randomIntBetween(int a, int b);
void clearMat(mat* m);
void changeBlock(block* dest, block* src);
void resetVars(mat m, evars* ev);
block* randomBlock(block** BASIC_BLOCKS);
void reset(unsigned int* score, unsigned int* linesCleared, mat* m, block* s, block* nextBlock, block** BASIC_BLOCKS, evars* envVars);
double meaned(int* arr, int size);
double previewScore(mat m, block s, double* prefs, evars* previousEvars, int col, double mch, double mdch);
mat* previewMatIfPushDown(mat* m, block s, int* numCleared);
evars* retrieveEvars(mat m, evars* previousEvars);
void printMat(mat* m, block s);
void printMatrix(int** data, int rows, int cols);
int generateRandomNumber(int min, int max);
double generateRandomDouble(double min, double max);
block* randomBlockWithSeed(block** BASIC_BLOCKS, unsigned int* seed);
bool userTickCallBack(mat *m, block *s, block *nextBl, unsigned int *score, unsigned int* linesCleared, block **BASIC_BLOCKS);
bool heuristicTickCallBack(mat *m, block *s, block *nextBl, unsigned int *score, unsigned int* linesCleared, block **BASIC_BLOCKS, evars *e);
#endif /* game_h */
