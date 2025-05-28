#ifndef GenNN_hpp
#define GenNN_hpp
#include "FileHelper.hpp"
#include "NN.h"
#include "assert.h"
#include "game.h"
#include "mlx/array.h"
#include "tetrisrandom.hpp"
#include <mlx/mlx.h>
#include <vector>
#include <random>
#include "tetrisrandom.hpp"

std::string getCurrentDateTime();

using namespace mlx::core;
// a tuple representing the env variables, the combination it came from and the
// lines cleared

class NNIndividual {
public:
  MultiLayer *mlp = nullptr;
  unsigned int id;
  unsigned int score = 0;
  unsigned int linesCleared = 0;
  std::string name;

  NNIndividual(int input_size, std::vector<int> hidden_sizes, unsigned int id,
               std::string name) {
    this->name = name;
    this->mlp = new MultiLayer(input_size, hidden_sizes);
    this->id = id;
  }

  void updateParams(const std::vector<array> &params) {
    this->mlp->update(params);
  }
};

class GeneticNN {
public:
  std::vector<NNIndividual> population;
  unsigned int count = 64;
  std::vector<int> hidden_sizes;
  std::string name;
  unsigned int populationID = 0;
  std::mt19937 seed_gen;
  int seed;

  GeneticNN(int input_size, std::vector<int> hidden_sizes, std::string name = "") {
    std::random_device rd;
    seed_gen = std::mt19937(rd());
    this->seed = seed_gen();

    this->hidden_sizes = hidden_sizes;

    if (name == "") {
      this->name = getCurrentDateTime();
    } else {
      this->name = name;
    }

    for (int i = 0; i < count; i++) {
      population.push_back(
          NNIndividual(input_size, hidden_sizes, i, NAMES[i]));
    }
  }
  void setResult(unsigned int popid, unsigned int score,
                 unsigned int linesCleared) {
    population[popid].score = score;
    population[popid].linesCleared = linesCleared;
    addGenWithName(name.c_str(), score, linesCleared,
                   popid == 0 && populationID == 0, population[popid].id,
                   populationID);
  }
  void udpatePopulation();

  void breed(int parent1, int parent2, int child);

  bool tickCallback(mat *m, block *s, block *nextBl, evars *e,
                    unsigned int *score, unsigned int *linesCleared,
                    unsigned int index, block **BASIC_BLOCKS,
                    TetrisRandom& tetRand, bool instant);

  std::vector<array> batchForward(std::vector<tetrisState> states, int index) {
    std::vector<array> inputs;
    inputs.reserve(states.size());
    for (const auto &state : states) {
        array input = stateToArray(state);
        mlx::core::eval(input);
        inputs.push_back(input);
    }

    return batchForward(inputs, index);
  }
  std::vector<array> batchForward(std::vector<array> batchInput, int index) {
    std::vector<array> batchOutput;
    batchOutput.reserve(batchInput.size());
    for (const array &input : batchInput) {
      batchOutput.push_back(
          generalizedForward(input, this->population[index].mlp->params)
      );
    }
    mlx::core::eval(batchOutput);
    return batchOutput;
  }

  void loadPrevious(int genID, std::string date);
  void supafast(block **BASIC_BLOCKS);
  void createDir() {
      if (mkdir(("./saved_gens/" + this->name).c_str(), 0777) == -1) {
        printf("ERROR: coulnd't create directory \n");
        printf("ERROR: %s\n", strerror(errno));
        printf("ERROR: dir name: %s\n", this->name.c_str());
        return;
      }
  }
private:
  void supafastindiv(block **BASIC_BLOCKS, unsigned int index);
  void batchSupafast(block **BASIC_BLOCKS, unsigned int first,
                     unsigned int last);
  bestc act(std::vector<tetrisState> &possibleStates, int index);
  bestc act2(std::vector<std::tuple<array, bestc>> possibleBoards, int index);
  bestc actWithNextBlock(mat m, block s, block ns, evars *prev, int index);
  bestc actWithNextBlock2(mat m, block s, block ns, evars *prev, int index);
  const std::string NAMES[64] = {
      "Kauli Vaast         ", // surf, homme
      "Althéa Laurin       ", // taekwondo, +67kg, femmes
      "Alban               ",
      "Thibault            ",
      "Dai-Khanh           ",
      "Sidonie             ",
      "Romain              ",
      "Irène               ",
      "Simon               ",
      "Aurélien            ",
      "Cédrick             ",
      "Zacharie            ",
      "Maxime              ",
      "Auxence             ",
      "Shrek               ",
      "Timothée            ",
      "Adrien              ",
      "Karl                ",
      "Jérémie             ",
      "William             ",
      "Benoît              ",
      "Christophe          ",
      "Hilaire             ",
      "Jules               ",
      "David               ",
      "Sami                ",
      "Eliot               ",
      "Lucas               ",
      "Hélène              ",
      "Émilie              ",
      "Armand              ",
      "Lauriane Nolot      ", // voile
      "Yannick Borel       ", // escrime, épée hommes
      "Victor Koretzky     ", // VTT, cross-country hommes
      "M. Dijkstra         ",
      "M. Lamport          ",
      "Henri Ford          ",
      "Didier              ",
      "M. Deschamps        ",
      "Sid                 ",
      "Sara Balzer         ", // escrime, sabre individuel femmes
      "Sylvain André       ", // BMX racing, hommes
      "Angele Hug          ", // canoë slalom, kayak cross femmes
      "Sofiane Oumiha      ", // boxe, -63,5kg, hommes
      "George Washington   ",
      "Jean de la Fontaine ",
      "Victor Hugo         ",
      "Abraham Lincoln     ",
      "Alice & Natacha     ",
      "Paul                ",
      "Léonard             ",
      "Matis               ",
      "Henri               ",
      "Léon Marchand       ",
      "Teddy Riner         ", // Médaillé de judo
      "Joan-Benjamin Gaba  ", // médaillé de Judo
      "Luka Mkheidze       ", // médaillé de Judo
      "Clarisse Agbegnenou ", // médaillée de Judo
      "Manon Apithy-Brunet ", // médaillée d'escrime
      "Félix Lebrun        ", // médaillé de tennis de table
      "Joris Daudet        ", // médaillé de BMX
      "Antoine Dupont      ", // médaillé de rugby
      "Nicolas Gestin      ", // médaillé de canoë-kayak
      "Benjamin Thomas     ", // médaillé de cyclisme
  };
};
#endif
