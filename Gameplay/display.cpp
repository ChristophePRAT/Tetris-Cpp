#include "display.hpp"
#include <SDL2/SDL.h>
#include <SDL2_ttf/SDL_ttf.h>
#include <SDL2_image/SDL_image.h>
#include "SDL2/SDL_pixels.h"
#include "SDL2/SDL_render.h"
// #include "SDL2/SDL_video.h"
#include "../Helpers/blocksNshapes.hpp"

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

void renderString(SDL_Renderer *renderer, TTF_Font* Sans, char* str, const char* str2, int y) {
    char message[25];
    SDL_Color black = {0, 0, 0};
    SDL_snprintf(message, sizeof(message), "%s%s", str, str2);


    SDL_Surface* surfaceMessage = TTF_RenderUTF8_LCD(Sans, message, black, {255, 255, 255});
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

void drawMat(mat m, block s, SDL_Renderer* renderer, int TIMER_INTERVAL, int x, int y) {
    SDL_Rect squareRect;
    SDL_Color color;
    int i_offset, j_offset;

    // Draw the black grid
    color = blockColors[0];
    squareRect.w = SQUARE_WIDTH * (m.cols + x);
    squareRect.h = SQUARE_WIDTH * (m.rows + y);
    squareRect.x = SQUARE_WIDTH;
    squareRect.y = SQUARE_WIDTH;
    drawSquare(&squareRect, &color, renderer);

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
                }
                // else {
                //     color = blockColors[0];
                //     drawSquare(&squareRect, &color, renderer);
                // }
            }
        }
    }
}
