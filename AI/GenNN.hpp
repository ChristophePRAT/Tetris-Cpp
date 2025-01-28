#ifndef GenNN_hpp
#define GenNN_hpp
#include "FileHelper.hpp"
#include "NN.h"
#include "assert.h"
#include "game.h"
#include "mlx/array.h"
#include <mlx/mlx.h>
#include <vector>

std::string getCurrentDateTime();

using namespace mlx::core;
// a tuple representing the env variables, the combination it came from and the
// lines cleared
typedef std::tuple<evars *, bestc, int> tetrisState;

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
  unsigned int seed;

  GeneticNN(int input_size, std::vector<int> hidden_sizes,
            std::string name = "") {
    this->seed = time(NULL);
    this->hidden_sizes = hidden_sizes;
    // this->count = count;
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
                    unsigned int *see);

  std::vector<array> batchForward(std::vector<tetrisState> states, int index) {
    std::vector<array> inputs;
    for (const auto &state : states) {
      inputs.push_back(stateToArray(state));
    }
    return batchForward(inputs, index);
  }
  std::vector<array> batchForward(std::vector<array> batchInput, int index) {
    std::vector<array> batchOutput;
    for (const array &input : batchInput) {
      // batchOutput.push_back(this->ml->forward(input));
      batchOutput.push_back(
          // generalizedForward(input, this->ml->params)
          generalizedForward(input, this->population[index].mlp->params));
    }
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
      "M. Castel           ",
      "Mme Hémery          ",
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
      "M. Boisseleau       ",
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
      "M. Delamaire        ",
      "M. André            ",
      "Mme D'Halluin       ",
      "M. Guillemaud       ",
      "M. Lamport          ",
      "Henri Ford          ",
      "Didier              ",
      "M. Deschamps        ",
      "Sid                 ",
      "M. Corbineau        ",
      "Mme Chalmain        ",
      "M. Legaie           ",
      "M. Obadia           ",
      "M. Washington       ",
      "M. de la Fontaine   ",
      "M. Hugo             ",
      "M. Lincoln          ",
      "Armand              ",
      "Paul                ",
      "Léonard             ",
      "Matis               ",
      "Henri               ",
      "Léon Marchand       ",
      "Teddy Riner         ",
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
