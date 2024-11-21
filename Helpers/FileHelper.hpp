//
//  FileHelper.hpp
//  tutorial
//
//  Created by Christophe Prat on 18/06/2024.
//

#ifndef FileHelper_hpp
#define FileHelper_hpp

#include <stdio.h>
void addGMEntry(unsigned int *fileNum, int score, double* weights, int populationId, bool firstEntry);
void addDQNEntry(unsigned int *fileNum, int score, int linesCleared, bool firstEntry, int step);
void addGenNNEntry(unsigned int *fileNum, int score, int linesCleared, bool firstEntry, int individual, int population);
#endif /* FileHelper_hpp */
