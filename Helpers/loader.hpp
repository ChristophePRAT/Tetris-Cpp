
#ifndef loader_hpp
#define loader_hpp
#include "GenNN.hpp"
// #include <stdio.h>

void saveGen(GeneticNN genNN);
void loadGen(GeneticNN &genNN, int genID, std::string date);
void mkdir(std::string name);
#endif /* FileHelper_hpp */
