#ifndef display_hpp
#define display_hpp
#include <SDL2/SDL.h>
#include <SDL2_ttf/SDL_ttf.h>
#include <SDL2_image/SDL_image.h>
#include "SDL2/SDL_pixels.h"
#include "SDL2/SDL_render.h"
#include "SDL2/SDL_video.h"
#include "game.h"

//Screen dimension constants
const int SCREEN_WIDTH = 700;
const int SCREEN_HEIGHT = 700;
const int SQUARE_WIDTH = 30;

void renderText(SDL_Renderer *renderer, TTF_Font* Sans, int s, char* str, int y);
void drawSquare(SDL_Rect* squareRect, SDL_Color* color, SDL_Renderer* renderer);
void drawBlock(block s, int x, int y, SDL_Renderer* renderer);
void renderString(SDL_Renderer *renderer, TTF_Font* Sans, char* str, const char* str2, int y);
void drawMat(mat m, block s, SDL_Renderer* renderer, int TIMER_INTERVAL, int x, int y);
#endif
