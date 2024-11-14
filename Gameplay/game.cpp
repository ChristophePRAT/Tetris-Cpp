//
//  game.c
//  tutorial
//
//  Created by Christophe Prat on 30/03/2024.
//

#include "game.h"
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <random>
// using namespace std;

int min(int a, int b) {
    return a < b ? a : b;
}

int max(int a, int b) {
    return a > b ? a : b;
}



/// Returns the first non-0 index of the matrix at the given column
/// - Parameters:
///   - m: the given matrix
///   - col: the given column
int heightOfColumn(mat m, int col) {
  // Ensure the column index is within bounds
  if (col < 0 || col >= m.cols) {
    return m.rows;  // Return m.rows to indicate an invalid column
  }

  for (int i = 0; i < m.rows; i++) {
    if (m.data != NULL && m.data[i] != NULL && m.data[i][col] != 0) {
      return i;
    }
  }
  return m.rows;
}

int firstIndexNotZInColInShape(block s, int col) {
    for (int i = 3; i >= 0; i--) {
        if (s.shape[s.currentShape][i][col] != 0) {
            return i;
        }
    }
    return -1;
}

/// Returns the row when a block does a full drop (i.e. space bar)
int fullDrop(mat m, block s, bool preview) {
    int posMin = 20;

    for (int col = s.position[1]; col < 4 + s.position[1]; col++) {
        int index = firstIndexNotZInColInShape(s, col - s.position[1]);
        if (index == -1) {
            continue;
        }
        int h = heightOfColumn(m, col);
        if (h == 0) {
            return -1;
        }

        int pos = h - index;
        if (pos < posMin && pos > 0) {
            posMin = pos;
        }
    }
    return posMin - 1;
}

mat* createMat(int rows, int cols) {
    mat* m = (mat*)malloc(sizeof(mat));
    assert(m != NULL);
    m->rows = rows;
    m->cols = cols;
    m->data = (int**)malloc(rows * sizeof(int*));
    assert(m->data != NULL);
    for (int i = 0; i < rows; i++) {
        m->data[i] = (int*)malloc(cols * sizeof(int));
        assert(m->data[i] != NULL);
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = 0;
        }
    }
    return m;
}

void clearMat(mat* m) {
    assert(m->data != NULL);
    for (int i = 0; i < m->rows; i++) {
        assert(m->data[i] != NULL);
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = 0;
        }
    }
}

void initBlock(block* b, int shape[4][4][4], int numberOfShapes) {
    assert(b != NULL);
    assert(shape != NULL);
    b->shape = (int***) malloc(numberOfShapes * sizeof(int**));
    assert(b->shape != NULL);

    if (numberOfShapes > 4) {
        // Handle out-of-bounds error
        return;
    }
    for (int a = 0; a < numberOfShapes; a++) {
        b->shape[a] = (int**)malloc(4 * sizeof(int*));
        assert(shape[a]);

        // Copy the shape data to the block
        for (int i = 0; i < 4; i++) {
            b->shape[a][i] = (int*)malloc(4*sizeof(int));
            assert(b->shape[a][i]);
            for (int j = 0; j < 4; j++) {
                b->shape[a][i][j] = shape[a][i][j];
            }
        }
    }
    // Set the initial rotation and position of the block
    b->currentShape = 0;
    b->numberOfShapes = numberOfShapes;
    b->position[0] = 0;
    b->position[1] = 0;
}


block* createShape(int data[4][4], int position[2]) {
    block* s = (block*)malloc(sizeof(block));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            s->shape[0][i][j] = data[i][j];
        }
    }
    s->currentShape = 0;
    s->position[0] = position[0];
    s->position[1] = position[1];
    return s;
}

void freeMat(mat* m) {
    for (int i = 0; i < m->rows; i++) {
        if (m->data[i] != NULL) {
            free(m->data[i]);
        }
    }
    if (m->data != NULL) {
        free(m->data);
    }
    if (m != NULL) {
        free(m);
    }
}

void printMatrix(int** data, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", data[i][j]);
        }
        printf("\n");
    }
}

void printMat(mat* m, block s) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (i >= s.position[0] && i < s.position[0] + 4 && j >= s.position[1] &&
                j < s.position[1] + 4 &&
                s.shape[s.currentShape][i - s.position[0]][j - s.position[1]] > 0) {
                printf("%d ", s.shape[s.currentShape][i - s.position[0]][j - s.position[1]]);
            } else {
                printf("%d ", m->data[i][j]);
            }
        }
        printf("\n");
    }
}

bool canMoveDownShape(mat m, block s) {

    for (int i = 0; i < 4; i++) {    // for each row
        for (int j = 0; j < 4; j++) {  // for each col
            if (s.shape[s.currentShape][i][j] > 0) {      // if cell is in block
                if (
                    s.position[0] + i + 1 >= m.rows ||
                    m.data[s.position[0] + i + 1][s.position[1] + j] > 0
                    ) {
                        return false;
                    }
            }
        }
    }
    return true;
}
int downShape(mat m, block* s) {
//        if (canMoveDownShape(m, *s)) {
//            s->position[0]++;
//            return 0;
//        } else { return -1; }
    if (s->downPos > s->position[0]) {
        s->position[0]++;
        return 0;
    } else { return -1; }
}
bool canMoveRLShape(mat* m, block* s, int direction) {  // Can the block move to the right (direction = -1) or to the left (direction = 1)
    for (int i = 0; i < 4; i++) { // for each row
        for (int j = 0; j < 4; j++) { // for each colomn
            if (s->shape[s->currentShape][i][j] > 0) {
                int relativeRow = s->position[0] + i;
                int relativeCol = s->position[1] + j + direction;
                if (
                    relativeCol < 0 ||
                    relativeCol >= m->cols ||
                    relativeRow < 0 ||
                    relativeRow >= m->rows ||
                    m->data[relativeRow][relativeCol] > 0
                    ) {
                        return false;
                    }
            }
        }
    }
    return true;
}

void moveRLShape(mat* m, block* s, int direction) {
    if (canMoveRLShape(m, s, direction)) {
        s->position[1] += direction;
        s->downPos = fullDrop(*m, *s, false);
    }
}




block* emptyShape(void) {
    block* s = (block*) malloc(sizeof(block));
    assert(s != NULL);
    return s;
}

void pushDown(mat* m, int toRow) {
    for (int i = toRow; i > 0; i--) {
        for (int col = 0; col < m->cols; col++) {
            m->data[i][col] = m->data[i-1][col];
        }
    }
}


int pushToMat(mat* m, block s) {
    for (int i = 0; i < 4; i++) { // for each row
        for (int j = 0; j < 4; j++) { // for each column
            int d = s.shape[s.currentShape][i][j];
            if (d != 0) {
                //                assert(m->data[i + s.position[0]][j + s.position[1]] != d); // Make sure there are no conflicts
                assert (i + s.position[0] < 20);
                m->data[i + s.position[0]][j + s.position[1]] = d;
            }
        }
    }

    int numCleared = 0;
    int min = s.position[0] + 4 > m->rows ? m->rows : s.position[0] + 4;

    for (int row = s.position[0]; row < min; row++) {
//        printf("Checking position %d, %d \n", row);
        bool shouldPushDown = true;
        for (int col = 0; col < m->cols; col++) {
            if (m->data[row][col] == 0) {
                shouldPushDown = false;
                break;
            }
        }
        if (shouldPushDown) {
            // Push next rows 1 row down
            pushDown(m, row);
            numCleared += 1;
        }
    }
    return numCleared;
}

void freeBlock(block* b) {
    for (int i = 0; i < b->numberOfShapes; i++) {
        for (int j = 0; j < 4; j++) {
            if (b->shape[i][j] != NULL) {
                free(b->shape[i][j]);
            }
        }
        if (b->shape[i] != NULL) {
            free(b->shape[i]);
        }
    }
    if (b->shape != NULL) {
        free(b->shape);
    }
//    free(b);
}
void copyBlock(block* dest, block* src) {
    if (src == NULL) {
        printf("ERROR SRC IS NULL \n");
        return;
    }
//    if (dest != NULL) {
//        freeBlock(dest);
//    }
    dest->currentShape = src->currentShape;
    dest->numberOfShapes = src->numberOfShapes;
    dest->position[0] = src->position[0];
    dest->position[1] = src->position[1];
    dest->shape = (int***)malloc(4 * sizeof(int**));

    assert(dest->shape != NULL);
    assert(src->shape != NULL);
    for (int i = 0; i < 4; i++) {
        dest->shape[i] = (int**)malloc(4 * sizeof(int*));
        assert(dest->shape[i] != NULL);
//        assert(src->shape[i] != NULL);
        for (int j = 0; j < 4; j++) {
            dest->shape[i][j] = (int*)malloc(4 * sizeof(int));
            assert(dest->shape[i][j] != NULL);
//            assert(src->shape[i][j] != NULL);
            if (i < src->numberOfShapes) {
                for (int k = 0; k < 4; k++) {
                    dest->shape[i][j][k] = src->shape[i][j][k];
                }
            }
        }
    }
}

void changeBlock(block* dest, block* src) {
    dest->currentShape = src->currentShape;
    dest->numberOfShapes = src->numberOfShapes;
    dest->position[0] = src->position[0];
    dest->position[1] = src->position[1];
    assert(dest->shape != NULL);
    assert(src->shape != NULL);
    for (int i = 0; i < src->numberOfShapes; i++) {
//        dest->shape[i] = (int**)malloc(4 * sizeof(int*));
        assert(dest->shape[i] != NULL);
        assert(src->shape[i] != NULL);
        for (int j = 0; j < 4; j++) {
//            dest->shape[i][j] = (int*)malloc(4 * sizeof(int));
            assert(dest->shape[i][j] != NULL);
            assert(src->shape[i][j] != NULL);
            for (int k = 0; k < 4; k++) {
                dest->shape[i][j][k] = src->shape[i][j][k];
            }
        }
    }
}

bool canRotateBlock(mat m, block s) {
    if (s.shape == NULL) {
        return false;
    }
    int nextShape = (s.currentShape + 1) % s.numberOfShapes;
    for (int i = 0; i < 4; i++) { // for each row
        for (int j = 0; j < 4; j++) { // for each column
            if (s.shape[nextShape][i][j] > 0) {
                int relativeRow = s.position[0] + i;
                int relativeCol = s.position[1] + j;
                if (
                    relativeCol < 0 ||
                    relativeCol >= m.cols ||
                    relativeRow < 0 ||
                    relativeRow >= m.rows ||
                    m.data[relativeRow][relativeCol] > 0
                    ) {
                        return false;
                    }
            }
        }
    }
    return true;
}

void rotateShape(mat m, block* s) {
    if (canRotateBlock(m, *s)) {
        s->currentShape = (s->currentShape + 1) % s->numberOfShapes;
        s->downPos = fullDrop(m, *s, false);
    }
}


bool canInsertShape(mat m, block s) {
    // printMatrix(s.shape[s.currentShape], 4, 4);

    for (int i = 0; i < 4; i++) { // for each row
        for (int j = 0; j < 4; j++) { // for each column
            if (s.shape[s.currentShape][i][j] > 0) {
                if (s.position[0] + i >= m.rows || s.position[1] + j >= m.cols) {
                    return false;
                }
                if (m.data[s.position[0] + i][s.position[1] + j] > 0) {
                    return false;
                }
            }
        }
    }
    return true;
}
void computeDownPos(mat m, block *s) {
    s->downPos = fullDrop(m, *s, false);
}


evars* initVars(mat m) {
    int* ch = (int*)malloc(m.cols * sizeof( int));
    int* sl = (int*)malloc((m.cols-1)*sizeof( int));
    for (int i = 0; i < m.cols; i++) {
        ch[i] = 0;
        if (i != m.cols - 1) {
            sl[i] = 0;
        }
    }

    evars* ev = (evars*)malloc(sizeof (evars));

    *ev = {
        ch,
        sl,
        0,
        0,
        0
    };

    return ev;

}
void resetVars(mat m, evars* ev) {
    for (int i = 0; i < m.cols; i++) {
        ev->colHeights[i] = 0;
        if (i != m.cols - 1) {
            ev->deltaColHeights[i] = 0;
        }
    }
    ev->hMax = 0;
    ev->minMax = 0;
    ev->numHoles = 0;
}

int computeHolesAtCol(int col, mat m, int colHeight) {
    int holes = 0;

    for (int j = m.rows - colHeight; j < m.rows; j++) {
      holes += (m.data[j][col] == 0) ? 1 : 0;
    }

    return holes;
}

int computeHeightAtCol(int col, mat m) {
  for (int i = 0; i < m.rows; i++) {
    if (m.data[i][col] != 0) {
      return m.rows - i;
    }
  }
  return 0;
}

void updateEvars(mat m, evars* previousEvars) {
  int* colHeights = previousEvars->colHeights;
  int* deltaColHeights = previousEvars->deltaColHeights;
  int hMax = 0;
  int numHoles = 0;
  int minColHeight = 20;

  // Ensure colHeights and deltaColHeights are properly allocated
  if (colHeights == NULL || deltaColHeights == NULL) {
    colHeights = (int*)malloc(m.cols * sizeof(int));
    deltaColHeights = (int*)malloc(m.cols * sizeof(int));
    previousEvars->colHeights = colHeights;
    previousEvars->deltaColHeights = deltaColHeights;
  } else {
    colHeights = (int*)realloc(colHeights, m.cols * sizeof(int));
    deltaColHeights = (int*)realloc(deltaColHeights, m.cols * sizeof(int));
    previousEvars->colHeights = colHeights;
    previousEvars->deltaColHeights = deltaColHeights;
  }

  for (int i = 0; i < m.cols; i++) {
    colHeights[i] = computeHeightAtCol(i, m);
    if (i > 0) {
      deltaColHeights[i] = abs(colHeights[i - 1] - colHeights[i]);
    }
    if (colHeights[i] > hMax) {
      hMax = colHeights[i];
    } else if (colHeights[i] < minColHeight) {
      minColHeight = colHeights[i];
    }
    numHoles += computeHolesAtCol(i, m, colHeights[i]);
  }
  previousEvars->hMax = hMax;
  previousEvars->numHoles = numHoles;
  previousEvars->minMax = hMax - minColHeight;
}

evars* retrieveEvars(mat m, evars* previousEvars) {
    evars* ev = initVars(m);
    updateEvars(m, ev);
    return ev;
}


//void computeHoles(mat m, evars* previousEvars) {
//    int holes = 0;
//
//    for (int i = 0; i < m.cols; i++) {
//      for (int j = m.rows - previousEvars->colHeights[i]; j < m.rows; j++) {
//        holes += (m.data[j][i] == 0) ? 1 : 0;
//      }
//    }
//
//    previousEvars->numHoles = holes;
//}

mat *deepcopy(mat *m) {
    mat* newMat = createMat(m->rows, m->cols);
    newMat->rows = m->rows;
    newMat->cols = m->cols;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            newMat->data[i][j] = m->data[i][j];
        }
    }
    return newMat;
}

mat* previewMatIfPushDown(mat* m, block s, int* numCleared) {
    assert(s.position[0] >= 0);
    int row = fullDrop(*m, s, true);

    if (row == -1 || !canInsertShape(*m, s)) {
        return NULL;
    }

    assert(row >= 0);
    s.position[0] = row;
    // mat* newMat = createMat(m->rows, m->cols);
    // *newMat = *m;
    mat* newMat = deepcopy(m);
    assert(newMat != m);
    int numRowsCleared = pushToMat(newMat, s);
    *numCleared = numRowsCleared;
    return newMat;
}



int blockWidth(block s) {
    int w = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (s.shape[s.currentShape][i][j] > 0) {
                w++;
            }
        }
    }
    return w;
}
double meaned(int* arr, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum/float(size);
}

double previewScore(mat m, block s, double* prefs, evars* previousEvars, int col, double mch, double mdch) {
    s.position[1] = col;
    int numCleared = 0;

    mat* preview = previewMatIfPushDown(&m, s, &numCleared);;
    if (preview == NULL) {
        return -10000000000;
    }
    evars* ev = retrieveEvars(*preview, previousEvars);
    double score =
        prefs[0] * ev->hMax +
        prefs[1] * ev->numHoles +
    // les parametres suivants sont inutilisés par l'IA:
        prefs[2] * mch +
        prefs[3] * mdch +
        prefs[4] * numCleared +
        prefs[5] * ev->minMax;
    freeMat(preview);
    free(ev->colHeights);
    free(ev->deltaColHeights);
    free(ev);

    return score;
}

/// - Returns: Random number between 0 and 1
double randomProba() {
    double r = (double)random() / RAND_MAX;
    return r;
}

int randomIntBetween(int a, int b) {
    int r = a + (random() % (b - a));
    return r;
}

block* randomBlock(block** BASIC_BLOCKS) {
    int rdI = random() % 7;
    assert(rdI <= 6);
    return BASIC_BLOCKS[rdI];
}

void reset(unsigned int* score, unsigned int* linesCleared, mat* m, block* s, block* nextBlock, block** BASIC_BLOCKS, evars* envVars) {
    *score = 0;
    *linesCleared = 0;
    clearMat(m);
    changeBlock(s, randomBlock(BASIC_BLOCKS));
    computeDownPos(*m, s);
    changeBlock(nextBlock, randomBlock(BASIC_BLOCKS));
    resetVars(*m, envVars);
}

int generateRandomNumber(int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(min, max);

    return distrib(gen);
}
double generateRandomDouble(double min, double max) {
    // Use random_device to seed the random number generator
    std::random_device rd;  // Truly random seed source
    std::mt19937 gen(rd()); // Mersenne Twister RNG, seeded with rd
    std::uniform_real_distribution<> distrib(min, max);

    return distrib(gen);
}
