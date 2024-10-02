//
//  NN.cpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#include "NN.hpp"
#include "game.h"
#include <assert.h>
#include "mlx/mlx.h"
#include <cstdlib>
// #include "mlx/ops.h"
// #include "mlx/array.h"
// #include "mlx/transforms.h"
// using namespace mlx::core;
/*
array relu(const array& input) {
  return maximum(input, {0.0}); // Applies element-wise maximum [1]
}

array forward(mlp* ml, array input) {
  // Layer 1
  array z1 = add(matmul(input, *ml->w1), *ml->b1); // Linear transformation [2]
  array a1 = relu(z1);  // ReLU activation

  // Layer 2
  array z2 = add(matmul(a1, *ml->w2), *ml->b2);
  array a2 = relu(z2);

  // Layer 3
  array z3 = add(matmul(a2, *ml->w3), *ml->b3);
  array a3 = relu(z3);

  // Output Layer
  array output = add(matmul(a3, *ml->w4), *ml->b4);

  return output; // Assuming a single output value
}
void backprop(mlp* ml, const array& input, const array& target, double learning_rate) {
    // Forward pass

    auto loss_fn = [&](array w) {
        auto yhat = forward(ml, input);
        return 0.5f * sum(square(yhat - target));
      };

    auto grad_fn = grad(loss_fn);

    // Compute gradients
    auto gradw1 = grad_fn(*ml->w1);
    auto gradb1 = grad_fn(*ml->b1);
    auto gradw2 = grad_fn(*ml->w2);
    auto gradb2 = grad_fn(*ml->b2);
    auto gradw3 = grad_fn(*ml->w3);
    auto gradb3 = grad_fn(*ml->b3);
    auto gradw4 = grad_fn(*ml->w4);
    auto gradb4 = grad_fn(*ml->b4);

    // Update weights
    *ml->w1 = *ml->w1 - learning_rate * gradw1;
    *ml->b1 = *ml->b1 - learning_rate * gradb1;
    *ml->w2 = *ml->w2 - learning_rate * gradw2;
    *ml->b2 = *ml->b2 - learning_rate * gradb2;

    *ml->w3 = *ml->w3 - learning_rate * gradw3;
    *ml->b3 = *ml->b3 - learning_rate * gradb3;
    *ml->w4 = *ml->w4 - learning_rate * gradw4;
    *ml->b4 = *ml->b4 - learning_rate * gradb4;


    eval(*ml->w1);
    eval(*ml->b1);
    eval(*ml->w2);
    eval(*ml->b2);
    eval(*ml->w3);
    eval(*ml->b3);
    eval(*ml->w4);
    eval(*ml->b4);
}*/

MLP initMLP() {


    MLP ml = MLP(6, {64, 32, 1});

    return ml;
}
void backprop(MLP ml, Value loss) {
    ml.zero_grad();
    loss.backward();

    for (Value* p : ml.parameters()) {
        p->data -= p->grad * 0.01;
    }
}

double previewScore(mat m, block s, evars* previousEvars, int col, double mch, double mdch, MLP ml) {
    s.position[1] = col;
    int numCleared = 0;

    mat* preview = previewMatIfPushDown(&m, s, &numCleared);;
    if (preview == NULL) {
        return -10000000000;
    }
    evars* ev = retrieveEvars(*preview, previousEvars);

    double score = ml({Value(ev->hMax), Value(ev->numHoles), Value(mch), Value(mdch), Value(numCleared), Value(ev->minMax)})[0].data;
    // double score = forward(ml, array({
    //     float(ev->hMax),
    //     float(ev->numHoles),
    //     float(mch),
    //     float(mdch),
    //     float(numCleared),
    //     float(ev->minMax)
    // })).item<double>();
    freeMat(preview);
    free(ev->colHeights);
    free(ev->deltaColHeights);
    free(ev);

    return score;
}
double* getColFromBotDecision(mat m, block s, evars* previousEvars, MLP ml) {
//    int lastValidIndexForBlock = m.cols - blockWidth(s);

    double meanedColHeights = meaned(previousEvars->colHeights, m.cols);
    double meanedDeltaColHeights = meaned(previousEvars->deltaColHeights, m.cols);

    int bestCol = 0;

    double bestScore = previewScore(m, s, previousEvars, 0, meanedColHeights, meanedDeltaColHeights, ml);
    double* cs = (double*) malloc(2 * sizeof(double));
    for (int i = 1; i < m.cols; i++) {
        double score = previewScore(m, s, previousEvars, i, meanedColHeights, meanedDeltaColHeights, ml);
        if (score > bestScore) {
            bestScore = score;
            bestCol = i;
        }
    }

    cs[0] = bestCol;
    cs[1] = bestScore;

    return cs;
}
bestc theFinestDecision(mat m, block s, evars* previousEvars, MLP ml) {
    int bestCol = 0;
    int bestScore = -100000;
    int bestShape = -1;
    for (int i = 0; i < s.numberOfShapes; i++) {
        s.currentShape = i;
        double* cs = getColFromBotDecision(m, s, previousEvars, ml);
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

bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, MLP ml, unsigned int index, bool userMode, block** BASIC_BLOCKS) {
    int down = downShape(*m, s);

    *score += 1;

    if (down == -1) {
        int numCleared = pushToMat(m, *s);
        *score += 200 * pow(numCleared, 2);

        updateEvars(*m, e);
        changeBlock(s, nextBl);
        changeBlock(nextBl, randomBlock(BASIC_BLOCKS));
        if (userMode) {
            computeDownPos(*m, s);
        }

        if (!userMode) {

            bestc compo = theFinestDecision(*m, *s, e, ml);
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
