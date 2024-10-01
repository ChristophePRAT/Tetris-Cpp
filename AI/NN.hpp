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

typedef struct MLP {
    // Taille input = 6 pour le moment (hauteur max, hmax - hmin, delta h col, somme col, nb trous et nb de lignes effacées)
    unsigned int input_dim = 6;
    unsigned int couche_1 = 64;
    unsigned int couche_2 = 64;
    unsigned int couche_3 = 32;
    unsigned int output_dim = 1;

    mlx::core::array *w1;
    mlx::core::array *b1;

    mlx::core::array *w2;
    mlx::core::array *b2;

    mlx::core::array *w3;
    mlx::core::array *b3;

    mlx::core::array *w4;
    mlx::core::array *b4;


    // Taille output = 1 pour le moment (valeur de la fonction d'utilité)

} mlp;

#endif /* NN_hpp */
