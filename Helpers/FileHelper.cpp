//
//  FileHelper.cpp
//  tutorial
//
//  Created by Christophe Prat on 18/06/2024.
//

#include "FileHelper.hpp"
#include "stdlib.h"
#include "string.h"
#include <cstdio>

char* join_doubles(double* arr, int len) {
  int i;
  int size = 0;
  char* str;
  char temp[20];

  // Calculate the size of the resulting string
  for (i = 0; i < len; i++) {
    size += snprintf(temp, sizeof(temp), "%lf", arr[i]);
    if (i < len - 1) {
      size++;  // for the comma
    }
  }

  // Allocate memory for the resulting string
  str = (char*)malloc(size + 1);
  if (str == NULL) {
    return NULL;
  }

  // Join the doubles into the string
  str[0] = '\0';
  for (i = 0; i < len; i++) {
      snprintf(temp, 20, "%lf", arr[i]);
//    sprintf(temp, "%lf", arr[i]);
    strcat(str, temp);
    if (i < len - 1) {
      strcat(str, ",");
    }
  }

  return str;
}

void addGMEntry(unsigned int *fileNum, int score, double* weights, int populationId, bool firstEntry) {
    FILE *fptr;
    char fileName[40];

    sprintf(fileName, "scores_%d.csv", *fileNum);

    // Open a file in read mode
    fptr = fopen(fileName, "a");
    if (fptr == nullptr) {
        fptr = fopen(fileName, "w");
        fprintf(fptr, "score,weight1,weight2,weight3,weight4,weight5,weight6,population");
    } else {
        fseek(fptr, 0, SEEK_END);
        if (ftell(fptr) == 0) {
            fprintf(fptr, "score,weight1,weight2,weight3,weight4,weight5,weight6,population");
        } else {
            if (firstEntry) {
                *fileNum += 1;
                addGMEntry(fileNum, score, weights, populationId, firstEntry);
            }
        }
    }

    // Stored at /Users/christopheprat/Library/Containers/com.christopheprat.tutorial/Data
    fprintf(fptr, "\n%d, %s, %d", score, join_doubles(weights, 6), populationId);


    fclose(fptr);
}

void addDQNEntry(unsigned int *fileNum, int score, int linesCleared, bool firstEntry, int step) {
    FILE *fptr;
    char fileName[40];

    sprintf(fileName, "scores_dqn_%d.csv", *fileNum);

    // Open a file in read mode
    fptr = fopen(fileName, "a");
    if (fptr == nullptr) {
        fptr = fopen(fileName, "w");
        fprintf(fptr, "score,linesCleared,step");
    } else {
        fseek(fptr, 0, SEEK_END);
        if (ftell(fptr) == 0) {
            fprintf(fptr, "score,linesCleared,step");
        } else {
            if (firstEntry) {
                *fileNum += 1;
                addDQNEntry(fileNum, score, linesCleared, firstEntry, step);
            }
        }
    }

    fprintf(fptr, "\n%d, %d, %d", score, linesCleared, step);

    fclose(fptr);
}

void addGenNNEntry(unsigned int *fileNum, int score, int linesCleared, bool firstEntry, int individual, int population) {
    FILE *fptr;
    char fileName[40];

    sprintf(fileName, "scores_gennn_%d.csv", *fileNum);

    // Open a file in read mode
    fptr = fopen(fileName, "a");
    if (fptr == nullptr) {
        fptr = fopen(fileName, "w");
        fprintf(fptr, "score,linesCleared,individual,population");
    } else {
        fseek(fptr, 0, SEEK_END);
        if (ftell(fptr) == 0) {
            fprintf(fptr, "score,linesCleared,individual,population");
        } else {
            if (firstEntry) {
                *fileNum += 1;
                addGenNNEntry(fileNum, score, linesCleared, firstEntry, individual, population);
            }
        }
    }

    fprintf(fptr, "\n%d, %d, %d, %d", score, linesCleared, individual, population);

    fclose(fptr);
}
void addGenWithName(const char* name, int score, int linesCleared, bool firstEntry, int individual, int population) {
    FILE *fptr;
    char fileName[40];

    sprintf(fileName, "scores_gennn_%s.csv", name);

    // Open a file in read mode
    fptr = fopen(fileName, "a");
    if (fptr == nullptr) {
        fptr = fopen(fileName, "w");
        fprintf(fptr, "score,linesCleared,individual,population");
    } else {
        fseek(fptr, 0, SEEK_END);
        if (ftell(fptr) == 0) {
            fprintf(fptr, "score,linesCleared,individual,population");
        }
    }

    fprintf(fptr, "\n%d, %d, %d, %d", score, linesCleared, individual, population);

    fclose(fptr);
}
