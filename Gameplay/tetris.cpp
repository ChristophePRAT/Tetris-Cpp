#include <SDL2/SDL.h>
#include <SDL2_ttf/SDL_ttf.h>
#include <SDL2_image/SDL_image.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "SDL2/SDL_pixels.h"
#include "SDL2/SDL_render.h"
#include "SDL2/SDL_video.h"
#include "game.h"
#include "../Helpers/blocksNshapes.hpp"
#include "agent.h"
#include <cstdlib>
#include <time.h>
#include "FileHelper.hpp"
#include "NN.hpp"
#include "GenNN.hpp"

// AI_MODE  = -1: User mode | 0: Genetic | 1: DQN | 2: Genetic Neural Network
unsigned int AI_MODE = 2;

int loadGen = -1;
std::string loadDate;

void loop(void);
int init(void);
void kill(void);

// USER MODE
const bool userMode = AI_MODE == -1;

bool instantMode = false;

//Screen dimension constants
const int SCREEN_WIDTH = 700;
const int SCREEN_HEIGHT = 700;
//const int TIMER_INTERVAL = 1000;
const int SQUARE_WIDTH = 30;

int TIMER_INTERVAL = userMode ? 400 : 100;

//The window we'll be rendering to
block** BASIC_BLOCKS = NULL;

SDL_Window* window = NULL;

//The surface contained by the window
SDL_Surface* screenSurface = NULL;
//The window renderer
SDL_Renderer* renderer = NULL;

//Globally used font
TTF_Font* gFont = NULL;

// Refresh timer
Uint32 timerID = 0;

Uint64 start, end, time_taken;


void renderText(SDL_Renderer *renderer, TTF_Font* Sans, int s, char* str, int y) {
    char message[25];
    SDL_Color black = {0, 0, 0};
    SDL_snprintf(message, sizeof(message), "%s%d", str, s);

    SDL_Surface* surfaceMessage = TTF_RenderText_Shaded(Sans, message, black, {255, 255, 255});
    if (!surfaceMessage) {
        SDL_Log("Failed to render text: %s", TTF_GetError());
        return;
    }

    SDL_Texture* Message = SDL_CreateTextureFromSurface(renderer, surfaceMessage);
    if (!Message) {
        SDL_Log("Failed to create texture from surface: %s", SDL_GetError());
        SDL_FreeSurface(surfaceMessage);
        return;
    }

    SDL_Rect Message_rect = { 13 * SQUARE_WIDTH, y * SQUARE_WIDTH, surfaceMessage->w, surfaceMessage->h };

    SDL_RenderCopy(renderer, Message, NULL, &Message_rect);

    SDL_FreeSurface(surfaceMessage);
    SDL_DestroyTexture(Message);
}




void drawSquare(SDL_Rect* squareRect, SDL_Color* color, SDL_Renderer* renderer) {
    // Set renderer color to draw the square
    SDL_SetRenderDrawColor(renderer, color->r, color->g, color->b, color->a);

    // Draw filled square
    SDL_RenderFillRect(renderer, squareRect);
}


void drawBlock(block s, int x, int y, SDL_Renderer* renderer) {
    SDL_Rect squareRect;
    SDL_Color color;
    for (int i = 0; i < 4; i++) {
        squareRect.y = (y + i) * SQUARE_WIDTH;
        for (int j = 0; j < 4; j++) {
            squareRect.x = (x + j) * SQUARE_WIDTH;
            squareRect.w = SQUARE_WIDTH;
            squareRect.h = SQUARE_WIDTH;
            if (s.shape[s.currentShape][i][j] != 0) {
                color = blockColors[s.shape[s.currentShape][i][j]];
                drawSquare(&squareRect, &color, renderer);
            }
        }
    }
}


void drawMat(mat m, block s, SDL_Renderer* renderer) {
    SDL_Rect squareRect;
    SDL_Color color;
    int i_offset, j_offset;
    for (int i = 0; i < m.rows; i++) {
        squareRect.y = (i + 1) * SQUARE_WIDTH;
        for (int j = 0; j < m.cols; j++) {
            squareRect.x = (j + 1) * SQUARE_WIDTH;
            squareRect.w = SQUARE_WIDTH;
            squareRect.h = SQUARE_WIDTH;
            if (m.data[i][j] != 0) {
                color = blockColors[m.data[i][j]];
                drawSquare(&squareRect, &color, renderer);
            } else {
                i_offset = i - s.position[0];
                j_offset = j - s.position[1];
                if (TIMER_INTERVAL >= 20 && i_offset >= 0 && j_offset >= 0 && i_offset < 4 && j_offset < 4 &&
                    s.shape[s.currentShape][i_offset][j_offset] != 0) {
                    color = blockColors[s.shape[s.currentShape][i_offset][j_offset]];
                    drawSquare(&squareRect, &color, renderer);
                } else {
                    color = blockColors[0];
                    drawSquare(&squareRect, &color, renderer);
                }
            }
        }
    }
}

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
            if (i < argc - 2) {
                loadGen = atoi(args[i + 1]);
                loadDate = args[i+2];
            } else {
                printf("Please provide a value for the file to load\n");
                return 1;
            }
        }
    }

    if (AI_MODE == -1) {
        printf("Game loaded with user mode 🕹️\n");
    } else if (AI_MODE == 0) {
        printf("Game loaded with genetic mutation mode 🦁\n");
    } else if (AI_MODE == 1) {
        printf("Game loaded with DQN mode 🧠\n");
    } else if (AI_MODE == 2) {
        printf("Game loaded with neural network genetic mutation mode ⚛️\n");
    }

    if (!init()) {
        loop();
    } else { kill(); return 0; }
    kill();
    return 0;
}


// create a standard c class

void loop() {
//    srand(time(NULL));

    TTF_Font* latexFont = TTF_OpenFont("resources/lmroman17-regular.otf", 24);


    BASIC_BLOCKS = createBlocks();
    assert(BASIC_BLOCKS != NULL);
    mat* m = createMat(20, 10);
    block* s = emptyShape();
//    block* sA = BASIC_BLOCKS[rand() % 7];

//    block* sA = BASIC_BLOCKS[(int)floor(randomProba() * 8)];
    block* sA = randomBlock(BASIC_BLOCKS);
    copyBlock(s, sA);
    computeDownPos(*m, s);

    block* nextBlock = emptyShape();
//    sA = BASIC_BLOCKS[(int)floor(randomProba() * 8)];
    sA = randomBlock(BASIC_BLOCKS);
    copyBlock(nextBlock, sA);

    evars* envVars = initVars(*m);

    Uint32 next_time = SDL_GetTicks() + TIMER_INTERVAL;

    bool gameOver = false;
    // Event loop exit flag
    bool quit = false;

    // AIs
    population* g;
    DQN dqn = DQN(6, 0,0,0,0,0.01, 50);
    GeneticNN genNN = GeneticNN(20, 6, { 16,8,1 });

    // -------------
    // AIs
    // Genetic mutation
    if (AI_MODE == 0) {
        g = initializePopulation(20);
        printpopulation(g);
    }

    if (loadGen != -1) {
        genNN.loadPrevious(loadGen, loadDate);
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

    // Event loop
    while(!quit) {
        SDL_Event e;

        if (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT:
                    quit = true;
                    break;
                case SDL_USEREVENT:
                    printf("User event \n");
                    break;
                case SDL_KEYDOWN:
                    if (e.key.keysym.scancode == SDL_SCANCODE_RIGHT) {
                        moveRLShape(m, s, 1);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_LEFT) {
                        moveRLShape(m, s, -1);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_DOWN) {
                        //                        downShape(*m, s);
                        if (AI_MODE == 0) {
                            tickCallback(m, s, nextBlock, envVars, g, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
                        }
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_UP) {
                        rotateShape(*m, s);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_B) {
                        instantMode = !instantMode;
                        // Initialize renderer color white for the background
                        SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);

                        // Clear screen
                        SDL_RenderClear(renderer);
                        char s[] = "Boost Mode: ";
                        renderText(renderer, latexFont, 1, s, 0);
                        SDL_RenderPresent(renderer);
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_9) {
                        TIMER_INTERVAL = 20;
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_8) {
                        TIMER_INTERVAL = userMode ? 400 : 100;
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_F) {
                        //                        TIMER_INTERVAL *= 0.9;
                        if (TIMER_INTERVAL > 1) {
                            TIMER_INTERVAL -= 1;
                        }
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_S) {
                        TIMER_INTERVAL += 1;
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_SPACE) {
                        int i = fullDrop(*m, *s, false);
                        s->position[0] = i;
                        if (AI_MODE == 0) {
                            tickCallback(m, s, nextBlock, envVars, g, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
                        } else if (AI_MODE == 1) {
                            dqn.tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
                        } else if (AI_MODE == 2) {
                            genNN.tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
                        }
                    } else if (e.key.keysym.scancode == SDL_SCANCODE_0) {
                        int i = fullDrop(*m, *s, false);
                        s->position[0] = i;
                    }
                    break;
                default:
                    break;
            }
        }

        // Check if it's time to call the TimerCallback function
        Uint32 current_time = SDL_GetTicks();
        if (current_time >= next_time || instantMode) {
            if (AI_MODE == 0) {
                gameOver = !tickCallback(m, s, nextBlock, envVars, g, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
            } else if (AI_MODE == 1) {
                gameOver = !dqn.tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
            } else {
                gameOver = !genNN.tickCallback(m, s, nextBlock, envVars, &score, &linesCleared, index, userMode, BASIC_BLOCKS);
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
                    addDQNEntry(&fileNum, score, linesCleared, index == 0 && dqn.step == 0, dqn.step);
                } else if (AI_MODE == 2) {
                    addGenNNEntry(&fileNum, score, linesCleared, index == 0 && genNN.populationID == 0, index, genNN.populationID);
                }

                if (AI_MODE == 0) {
                    scores[index] = score;
                    index++;
                    if (index >= g->numIndividuals) {
                        previousBest = mutatepopulation(g, scores);
                        index = 0;
                    }
                } else if (AI_MODE == 1) {
                    dqn.trainNN();
                } else if (AI_MODE == 2) {
                    scores[index] = score;
                    genNN.setResult(index, score);
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
            // Initialize renderer color white for the background
            SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);

            // Clear screen
            SDL_RenderClear(renderer);
            char s1[] = "Score: ";
            char s2[] = "Individual: ";
            char s3[] = "Population: ";
            char s4[] = "Previous best: ";
            char s5[] = "Interval: ";
            renderText(renderer, latexFont, score, s1, 5);
            if (AI_MODE == 0) {
                renderText(renderer, latexFont, g->id, s3, 9);
                renderText(renderer, latexFont, index, s2, 8);
                renderText(renderer, latexFont, previousBest, s4, 7);
            } else if (AI_MODE == 2) {
                renderText(renderer, latexFont, genNN.populationID, s3, 9);
                renderText(renderer, latexFont, index, s2, 8);
            }
            char linesClearedText[] = "Lines Cleared: ";
            renderText(renderer, latexFont, linesCleared, linesClearedText, 6);
            renderText(renderer, latexFont, TIMER_INTERVAL, s5, 11);

            // Display next shape by the side of the screen
            drawBlock(*nextBlock, 12, 1, renderer);

            drawMat(*m, *s, renderer);
            // }
            //         Update screen
            SDL_RenderPresent(renderer);
        }
    }
}

int init() {
    srandom(time(NULL));
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

    //Quit SDL subsystems
    TTF_Quit();
//    SDL_Quit();
}
