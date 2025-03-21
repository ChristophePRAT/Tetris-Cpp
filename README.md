# Tetris C++ Game: Testing different AIs

<img width="812" alt="tetris-ai" src="https://github.com/user-attachments/assets/6afceb35-c506-462d-a4ad-f059674fbb27" />

## Motivation 

For our TIPE (personal project in CPGE), I built a tetris game in C++ and different AIs to learn to play the game. In the end, the goal is to compare the results of the different AIs.

## How to run
1. Clone the repository
2. Install [MLX](https://ml-explore.github.io/mlx/build/html/install.html#c-api) library. You can use `brew install mlx` on macOS.
3. Run the following commands:
```bash
cmake -B build .
cmake --build build
./build/tetris-ai -m 3
```

Quick note about the parameters:
- `-m 1` is for training a simple genetic mutation without neural networks
- `-m 2` is for training a DQN
- `-m 3` is for training a Genetic Neural Network

## Game
Tetris is an arcade game created by Alexey Pajitnov in 1984. The game consists of a matrix of 10x20 cells, where the player has to place the falling pieces in a way that they form a line. When a line is formed, it disappears and the player scores points. The game ends when the pieces reach the top of the matrix.

## AIs

You can check the scores in the `scores/` folder.

### Genetic Algorithm
Our genetic algorithm is based on the following steps:
1. Create an initial population of random individuals, with random weights
2. Evaluate the fitness of each individual, by making them play 1 game of tetris
3. Sort the individuals by fitness
4. Eliminate the worst individuals (1/2 of the population)
5. Create new individuals by crossing over the best individuals (choosing random weights from the best individuals to create the new ones)
6. Repeat step 2 to 6
### DQN
Our DQN algorithm is based on the following steps:
1. Create a NN
2. Define a heuristic function that evaluates the state of the game
3. Play the game and randomly store states in a replay buffer
4. Once each game ends, train the NN with the states stored in the replay buffer for 10 epochs

### Genetic Neural Network
Same as the basic genetic algorithm but the individuals are neural networks.
Currently this is the best we have. Managed to get a score of 289 lines cleared within 7 minutes of training in boost mode. For reference, I scored 180 and gamers can score around 220.

Our max score was around 2500 (in roughly 10 minute of training).
