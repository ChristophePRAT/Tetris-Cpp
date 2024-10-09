//
//  NN.cpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#include "NN.hpp"
#include "game.h"
#include <assert.h>
#include "mlx/array.h"
#include "mlx/mlx.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

using namespace mlx::core;

// #include "mlx/ops.h"
// #include "mlx/array.h"
// #include "mlx/transforms.h"
// using namespace mlx::core;


void printArray(array a) {
    printf("SUM: %f\n", sum(a).item<float>());
    printf("Mean: %f\n", mean(a).item<float>());
    printf("Max: %f\n", max(a).item<float>());
    printf("Min: %f\n", min(a).item<float>());
    for (auto s: a) {
        for (auto j: s) {
            // printf("%zu\n", j.size());
            // printf("%f\n", j.dtype());
            // printf("data: %f\n", j.data<float>());
            printf(" %f ", j.item<float>());
        }
        printf("\n");
    }
}
void printVect(array x) {
    for (auto i: x) {
        printf("%f ", i.item<float>());
    }
    printf("\n");
}

array relu(const array& input) {
  return maximum(input, {0.0}); // Applies element-wise maximum [1]
}

array loss(array predictions, array targets) {
    auto loss = square(predictions - targets);
    return mean(loss);
}

array MultiLayer::forward(const array& x) {
    eval(x);

    printf("FORWARD PASS\n");
    // printf("%f", x.item<float>());
    printVect(x);
    array y = x;
    for (int i = 0; i < this->layers.size(); i++) {
        if (i == 0) {
            y = relu(this->layers[i].forward(y));
        } else {
            y = this->layers[i].forward(y);
        }

        printf("Bias: \n");
        printVect(*this->layers[i].bias);
        printf("y%d: \n", i);
        // printVect(y);
    }
    eval(y);
    printf("last y\n");
    printVect(y);
    printf("x:\n");
    // printVect(x);
    printf("END FORWARD PASS\n");
    return y;
}

void train() {
    MultiLayer ml = MultiLayer(2, {16,16,16,1});

    array x = linspace(0, 20, 100);
    eval(x);
    array y = 2 * x + 1;
    eval(y);
    array targetZ = 3 * y - 4*x -3;
    eval(targetZ);

    auto forw = [&](array w) {
        return ml.forward(w);
    };
    auto inputs = transpose(stack({x,y}));
    // auto inputs = x;

    eval(ml.params);

    // printf("fopisndfpis\n");
    printArray(ml.params[0]);


    auto loss_fn = [&ml, &targetZ](const std::vector<array>& input) {
            // ml.set_parameters(input);

            std::vector<array> ps = {input.begin(), input.end() - 1};
            // ml.update(ps);
            array predictions = ml.forward(input.back());
            return mean(square(predictions - targetZ));
        };

    // Create a value_and_grad function

    // calculate the grad of each parameter, i.e. from layers 0, 1, 2, each of them has 2 parameters (weights, bias)

    // create vector containing all integers from 0 to n
    std::vector<int> argnums = {};

    for (int i = 0; i < ml.params.size(); i++) {
        argnums.push_back(i);
    }

    auto value_and_grad_fn = value_and_grad(loss_fn, argnums);

    for (int epoch = 0; epoch < 10; epoch++) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> randI(0,inputs.size()/2 - 1);

        int r = randI(rng);

        assert(r<inputs.size()/2);
        printf("Random sample: %d\n", r);

        auto in = inputs.begin() + r;


        std::vector<array> input = ml.params;

        // input.push_back(in);
        array in2 = *in;
        // printf("\n\nNUmber: %f\n\n", in2.item<float>());
        // assert(in2.size() == 2);

        input.push_back(in2);
        auto [loss, grads] = value_and_grad_fn(input);
        for (int i = 0; i < grads.size(); i++) {
            eval(grads[i]);
            printf("Grads of %d\n", i);
            if (i % 2 == 0) {
                printArray(grads[i]);
            } else {
                printVect(grads[i]);
            }

        }
        printf("faoipsjdfiopasjdf\n");
        printf("faoipsjdfiopasjdf\n");
        eval(loss);
        ml.update_parameters(grads, 1);
        eval(ml.params);
        // print loss
        printf("Epoch %d, Loss: %f\n", epoch, loss.item<float>());
    }
    // for (auto i: inputs) {
        // std::vector<array> input = {*ml.layers[0].weights, *ml.layers[0].bias, *ml.layers[1].weights, *ml.layers[1].bias, *ml.layers[2].weights, *ml.layers[2].bias};
    //     input.push_back(i);
    //     auto [loss, grads] = value_and_grad_fn(input);
    //     ml.update_parameters(grads, 0.01);
    // }
}

// MLP initMLP() {


//     MLP ml = MLP(6, {64, 32, 1});

//     return ml;
// }
// void backprop(MLP ml, Value loss) {
//     ml.zero_grad();
//     loss.backward();

//     for (Value* p : ml.parameters()) {
//         p->data -= p->grad * 0.01;
//     }
// }

double previewScore(mat m, block s, evars* previousEvars, int col, double mch, double mdch, MLP ml) {
    s.position[1] = col;
    int numCleared = 0;

    mat* preview = previewMatIfPushDown(&m, s, &numCleared);;
    if (preview == NULL) {
        return -10000000000;
    }
    evars* ev = retrieveEvars(*preview, previousEvars);

    double score = 0;
    // double score = ml({Value(ev->hMax), Value(ev->numHoles), Value(mch), Value(mdch), Value(numCleared), Value(ev->minMax)})[0].data;
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
