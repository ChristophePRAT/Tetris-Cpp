//
//  NN.hpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#ifndef NN_hpp
#define NN_hpp

#include <stdio.h>
#include <mlx/mlx.h>
class MLP {
    public:
        MLP(int nInput, int nHidden, int nOutput);
        ~MLP();
        void forward(mlx::core::array input);
        void backward(mlx::core::array target);
    private:
        int nInput;
        int nHidden;
        int nOutput;
};

#endif /* NN_hpp */
