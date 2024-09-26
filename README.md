# Tetris C++ Game: Testing different AIs
## How to run
1. Clone the repository
2. Install [MLX](https://ml-explore.github.io/mlx/build/html/install.html#c-api) library
3. Run the following commands:
```bash
mkdir build
cd build
cmake ..
make
./tetris-ai
```
## Game
Tetris is an arcade game created by Alexey Pajitnov in 1984. The game consists of a matrix of 10x20 cells, where the player has to place the falling pieces in a way that they form a line. When a line is formed, it disappears and the player scores points. The game ends when the pieces reach the top of the matrix.

## AIs

You can check the scores in the `build/scores.csv` file.

### Genetic Algorithm
Our genetic algorithm is based on the following steps:
1. Create an initial population of random individuals, with random weights
2. Evaluate the fitness of each individual, by making them play 1 game of tetris
3. Sort the individuals by fitness
4. Eliminate the worst individuals (1/2 of the population)
5. Create new individuals by crossing over the best individuals (choosing random weights from the best individuals to create the new ones)
6. Repeat step 2 to 6
### DQN

Yet to come...
