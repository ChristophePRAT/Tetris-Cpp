#include "../AI/GenNN.hpp"
#include "loader.hpp"
// #include <iostream>

using namespace mlx::core;

void mkdir(std::string name) {
    if (mkdir(name.c_str(), 0777) == -1) {
        printf("ERROR: coulnd't create directory \n");
        printf("ERROR: %s\n", strerror(errno));
        printf("ERROR: dir name: %s\n", name.c_str());
        return;
    }
}

void saveGen(GeneticNN genNN) {
    std::string root = genNN.name + "/generation_" + std::to_string(genNN.populationID);
    mkdir(root);

    for (int i = 0; i < genNN.count; i++) {
        std::string folderName = root + "/individual_" + std::to_string(i);
        mkdir(folderName);

        for (int j = 0; j < genNN.population[i].mlp->params.size(); j++) {
            std::string fileName = folderName + "/layer_" + std::to_string(j);

            save(fileName, genNN.population[i].mlp->params[j]);
        }
    }
}

void loadGen(GeneticNN &genNN, int genID, std::string date) {
    std::string root = date + "/generation_" + std::to_string(genID);
    genNN.populationID = genID;

    for (int i = 0; i < genNN.count; i++) {
        std::string folderName = root + "/individual_" + std::to_string(i);

        for (int j = 0; j < genNN.population[i].mlp->params.size(); j++) {
            std::string fileName = folderName + "/layer_" + std::to_string(j) + ".npy";

            genNN.population[i].mlp->params[j] = load(fileName);
        }
    }
}
