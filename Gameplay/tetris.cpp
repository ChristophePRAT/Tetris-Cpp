#include <SDL2/SDL.h>
#include <SDL2_ttf/SDL_ttf.h>
#include <SDL2_image/SDL_image.h>
#include <random>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "SDL2/SDL_events.h"
#include "SDL2/SDL_render.h"
#include "SDL2/SDL_timer.h"
#include "SDL2/SDL_video.h"
#include "SDL2/begin_code.h"
#include "SDL_scancode.h"
#include "game.h"
#include "blocksNshapes.hpp"
#include "simple-gm.h"
#include <cstdlib>
#include <time.h>
#include "FileHelper.hpp"
#include "rl.h"
#include "GenNN.hpp"
#include "display.hpp"
#include "heuristic.hpp"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "tetrisrandom.hpp"

// AI_MODE  = -1: User mode | 0: Genetic | 1: RL | 2: Genetic Neural Network | 3: Pre-defined heuristic
int AI_MODE = 2;

int loadGen = -1;
std::string loadName = "";

void loop(void);
int init(void);
void kill(void);

// USER MODE
bool userMode;

bool instantMode = false;
bool supafast = false;

int TIMER_INTERVAL;

//The window we'll be rendering to
block** BASIC_BLOCKS = NULL;

SDL_Window* window = NULL;


SDL_Surface* screenSurface = NULL;

SDL_Renderer* renderer = NULL;

//Globally used font
TTF_Font* gFont = NULL;

// Refresh timer
Uint32 timerID = 0;

Uint64 start, end, time_taken;

// Random piece generator following the original Tetris (i.e. bags of pieces)
TetrisRandom tetrisRand;

int main(int argc, char* args[] ) {
    for (int i = 0; i < argc; i++) {
        if (strcmp(args[i], "--ai_mode") == 0 || strcmp(args[i], "-m") == 0) {
            if (i != argc - 1) {
                AI_MODE = atoi(args[i + 1]);
            } else {
                printf("Please provide a value for the AI mode\n");
                return 1;
            }
        } else if (strcmp(args[i], "--load") == 0 || strcmp(args[i], "-l") == 0) {
            if (i < argc - 1) {
                loadGen = atoi(args[i + 1]);
            } else {
                printf("Please provide a value for the file to load\n");
                return 1;
            }
        } else if (strcmp(args[i], "--name") == 0 || strcmp(args[i], "-n") == 0) {
            if (i < argc - 1) {
                loadName = args[i + 1];
            } else {
                printf("Please provide a value for the file to load\n");
                return 1;
            }
        } else if (strcmp(args[i], "--supafast") == 0 || strcmp(args[i], "-sp") == 0) {
            supafast = true;
        }
    }

    userMode = AI_MODE == -1;

    TIMER_INTERVAL = userMode ? 400 : 100;

    BASIC_BLOCKS = createBlocks();
    assert(BASIC_BLOCKS != NULL);

    tetrisRand = TetrisRandom(std::random_device{}(), BASIC_BLOCKS);

    if (AI_MODE == -1) {
        printf("Game loaded with user mode 🕹️\n");
    } else if (AI_MODE == 0) {
        printf("Game loaded with genetic mutation mode 🦁\n");
    } else if (AI_MODE == 1) {
        printf("Game loaded with RL mode 🧠\n");
    } else if (AI_MODE == 2) {
        printf("Game loaded with neural network genetic mutation mode ⚛️\n");
    } else if (AI_MODE == 3) {
        printf("Game loaded with pre-defined heuristic mode 🤡\n");
    }

    if (supafast) {
        if (AI_MODE == 2) {
            GeneticNN genNN = GeneticNN(4, { 3, 1 }, loadName);
            if (loadName != "" && loadGen != -1) {
                genNN.loadPrevious(loadGen, loadName);
            } else {
                genNN.createDir();
            }
            genNN.supafast(BASIC_BLOCKS);
        } else if (AI_MODE == 3) {
            while (true) {
                supafastOneHeuristic(BASIC_BLOCKS, tetrisRand);
            }
        }
    }

    if (!init()) {
        loop();
    } else { kill(); return 0; }
    kill();
    return 0;
}

void receiveEvent(SDL_Event e, SDL_Renderer* renderer, TTF_Font *latexFont, bool *quit, mat* m, block* s) {
    // Track keyboard input
    // if (SDL_PollEvent(&e)) {
        switch (e.type) {
            case SDL_QUIT:
                *quit = true;
                break;
            case SDL_KEYDOWN:
                if (e.key.keysym.scancode == SDL_SCANCODE_B) {
                    instantMode = !instantMode;
                    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
                    SDL_RenderClear(renderer);
                    char s[] = "Boost Mode: ";
                    renderText(renderer, latexFont, 1, s, 0);
                    SDL_RenderPresent(renderer);
                } else if (e.key.keysym.scancode == SDL_SCANCODE_9) {
                    TIMER_INTERVAL = 20;
                } else if (e.key.keysym.scancode == SDL_SCANCODE_8) {
                    TIMER_INTERVAL = userMode ? 400 : 100;
                } else if (e.key.keysym.scancode == SDL_SCANCODE_F) {
                    if (TIMER_INTERVAL > 0) {
                        TIMER_INTERVAL -= 1;
                    }
                } else if (e.key.keysym.scancode == SDL_SCANCODE_S) {
                    TIMER_INTERVAL += 10;
                } else if (userMode) {
                    if (e.key.keysym.scancode == SDL_SCANCODE_H) {
                        moveRLShape(m, s, -1);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_L) {
                        moveRLShape(m, s, 1);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_R) {
                        rotateShape(*m, s);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_SPACE) {
                        s->position[0] = s->downPos;
                    }
                }
                break;
            default:
                break;
        }
    // }

}

void loop() {
    TTF_Font* latexFont = TTF_OpenFont("resources/lmroman17-regular.otf", 24);

    mat* m = createMat(20, 10);
    block* s = emptyShape();
    block* sA = tetrisRand.randomBlock();
    copyBlock(s, sA);
    computeDownPos(*m, s);

    block* nextBlock = emptyShape();
    sA = tetrisRand.randomBlock();
    copyBlock(nextBlock, sA);

    evars* envVars = initVars(*m);

    Uint32 next_time = SDL_GetTicks() + TIMER_INTERVAL;

    bool gameOver = false;
    bool quit = false;

    population* g;
    RL rl = RL(4, 100);
    GeneticNN genNN = GeneticNN(4, {3, 1 }, loadName);

    std::random_device rd;
    std::mt19937 gen(rd());

    if (AI_MODE == 0) {
        g = initializePopulation(20);
        printpopulation(g);
    } else if (AI_MODE == 2) {
        srand(genNN.seed);
    }

    if (loadGen != -1) {
        genNN.loadPrevious(loadGen, loadName);
    }

    unsigned int index = 0;
    unsigned int linesCleared = 0;
    unsigned int score = 0;
    unsigned int previousBest = 0;

    int* scores;
    if (AI_MODE == 0) {
        scores = (int*)malloc(g->numIndividuals * sizeof(int));
    }

    unsigned int fileNum = 0;

    while (!quit) {
        SDL_Event e;

        while(SDL_PollEvent(&e)) {
            receiveEvent(e, renderer, latexFont, &quit, m, s);
        }

        Uint32 current_time = SDL_GetTicks();
        if (current_time >= next_time || instantMode) {
            if (AI_MODE == 0) {
                gameOver = !tickCallback(m, s, nextBlock, envVars, g, &score, &linesCleared, index, BASIC_BLOCKS);
            } else if (AI_MODE == 1) {
                gameOver = !rl.tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, BASIC_BLOCKS, tetrisRand, TIMER_INTERVAL == 0 || instantMode);
            } else if (AI_MODE == 2) {
                gameOver = !genNN.tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, BASIC_BLOCKS, tetrisRand, TIMER_INTERVAL == 0 || instantMode);
            } else if (AI_MODE == -1) {
                gameOver = !userTickCallBack(m, s, nextBlock, &score, &linesCleared, BASIC_BLOCKS);
            } else if (AI_MODE == 3) {
                gameOver = !heuristicTickCallBack(m, s, nextBlock, &score, &linesCleared, BASIC_BLOCKS, envVars, tetrisRand);
            }

            if (gameOver) {
                printf("Lines cleared: %d \n", linesCleared);
                printf("------------------\n");
                if (userMode) {
                    TTF_CloseFont(latexFont);
                    quit = true;
                    return;
                } else if (AI_MODE == 0) {
                    addGMEntry(&fileNum, score, g->individuals[index].weights, g->id, index == 0 && g->id == 0);
                } else if (AI_MODE == 1) {
                    addRLEntry(&fileNum, score, linesCleared, index == 0 && rl.step == 0, rl.step);
                }

                if (AI_MODE == 0) {
                    scores[index] = score;
                    index++;
                    if (index >= g->numIndividuals) {
                        previousBest = mutatepopulation(g, scores);
                        index = 0;
                    }
                } else if (AI_MODE == 1) {
                    rl.trainNN(linesCleared);
                } else if (AI_MODE == 2) {
                    genNN.setResult(index, score, linesCleared);
                    srand(genNN.seed);
                    index++;
                    if (index >= genNN.count) {
                        genNN.udpatePopulation();
                        index = 0;
                    }
                }
                reset(&score, &linesCleared, m, s, nextBlock, BASIC_BLOCKS, envVars);
            }

            next_time = current_time + TIMER_INTERVAL;
        }

        if (!instantMode) {
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
            SDL_RenderClear(renderer);
            char s1[] = "Score: ";
            char s2[] = "Individual: #";
            char s3[] = "Population: ";
            char s4[] = "Previous best: ";
            char s5[] = "Interval: ";
            char s6[] = "Name: ";
            renderText(renderer, latexFont, score, s1, 5);
            if (AI_MODE == 0) {
                renderText(renderer, latexFont, g->id, s3, 9);
                renderText(renderer, latexFont, index, s2, 8);
                renderText(renderer, latexFont, previousBest, s4, 7);
            } else if (AI_MODE == 2) {
                renderText(renderer, latexFont, genNN.populationID, s3, 10);
                renderText(renderer, latexFont, index, s2, 8);
                renderString(renderer, latexFont, s6, genNN.population[index].name.c_str(), 9);
            }
            char linesClearedText[] = "Lines Cleared: ";
            renderText(renderer, latexFont, linesCleared, linesClearedText, 6);
            renderText(renderer, latexFont, TIMER_INTERVAL, s5, 11);

            drawBlock(*nextBlock, 12, 1, renderer);
            drawMat(*m, *s, renderer, TIMER_INTERVAL, 0, 0);
            SDL_RenderPresent(renderer);
        }
    }
}

int init() {
    srandom(time(NULL));
    srand(time(NULL));
    if( SDL_Init( SDL_INIT_VIDEO | SDL_INIT_TIMER | TTF_Init() ) < 0 ) {
        printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
        return 1;
    }
    //Create window
    window = SDL_CreateWindow( "Tetris AI", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );

    if (window == NULL) {
        printf("Error with window: %s\n", SDL_GetError());
        return 1;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);

    if (renderer == NULL) {
        printf("Error with renderer: %s\n", SDL_GetError());
        return 1;
    }

    screenSurface = SDL_GetWindowSurface(window);

    if (screenSurface == NULL) {
        printf("Error with screen surface: %s\n", SDL_GetError());
        return 1;
    }

    return 0;
}

void kill(void) {
    //Destroy window
    SDL_RemoveTimer(timerID);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    //Quit SDL subsystems
    TTF_Quit();
    SDL_Quit();
}
