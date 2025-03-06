//
//  blocksNshapes.cpp
//  tutorial
//
//  Created by Christophe Prat on 30/03/2024.
//

#include "blocksNshapes.hpp"
#include <stdlib.h>
#include "game.h"
#include <stdbool.h>
#include <assert.h>

block** createBlocks() {
  block** standardBlocks =
      (block**)malloc(7 * sizeof(block*));  // Change 6 to 7
  assert(standardBlocks != NULL);

  // square shape
  int square[1][4][4] = {{
      {0, 0, 0, 0},
      {1, 1, 0, 0},
      {1, 1, 0, 0},
      {0, 0, 0, 0},
  }};

  block* b = (block*)malloc(sizeof(block));
  initBlock(b, square, 1);
  standardBlocks[0] = b;

  // L shape
  int Lshape[4][4][4] = {{
                             {2, 0, 0, 0},
                             {2, 0, 0, 0},
                             {2, 2, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {0, 0, 0, 0},
                             {2, 2, 2, 0},
                             {2, 0, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {2, 2, 0, 0},
                             {0, 2, 0, 0},
                             {0, 2, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {0, 0, 2, 0},
                             {2, 2, 2, 0},
                             {0, 0, 0, 0},
                             {0, 0, 0, 0},
                         }};
  block* l = (block*)malloc(sizeof(block));
  initBlock(l, Lshape, 4);
  standardBlocks[1] = l;

  // T shape
  int Tshape[4][4][4] = {{
                             {0, 0, 0, 0},
                             {3, 3, 3, 0},
                             {0, 3, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {0, 3, 0, 0},
                             {3, 3, 0, 0},
                             {0, 3, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {0, 3, 0, 0},
                             {3, 3, 3, 0},
                             {0, 0, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {3, 0, 0, 0},
                             {3, 3, 0, 0},
                             {3, 0, 0, 0},
                             {0, 0, 0, 0},
                         }};
  block* t = (block*)malloc(sizeof(block));
  initBlock(t, Tshape, 4);
  standardBlocks[2] = t;

  // I shape
  int Ishape[2][4][4] = {{
                             {4, 4, 4, 4},
                             {0, 0, 0, 0},
                             {0, 0, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {4, 0, 0, 0},
                             {4, 0, 0, 0},
                             {4, 0, 0, 0},
                             {4, 0, 0, 0},
                         }};
  block* i = (block*)malloc(sizeof(block));
  initBlock(i, Ishape, 2);
  standardBlocks[3] = i;

  // reversed L shape
  int revLshape[4][4][4] = {{
                                {0, 5, 0, 0},
                                {0, 5, 0, 0},
                                {5, 5, 0, 0},
                                {0, 0, 0, 0},
                            },
                            {
                                {5, 0, 0, 0},
                                {5, 5, 5, 0},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                            },
                            {
                                {0, 5, 5, 0},
                                {0, 5, 0, 0},
                                {0, 5, 0, 0},
                                {0, 0, 0, 0},
                            },
                            {
                                {0, 0, 0, 0},
                                {5, 5, 5, 0},
                                {0, 0, 5, 0},
                                {0, 0, 0, 0},
                            }};
  block* revl = (block*)malloc(sizeof(block));
  initBlock(revl, revLshape, 4);
  standardBlocks[4] = revl;

  // S shape
  int Sshape[2][4][4] = {{
                             {0, 0, 0, 0},
                             {0, 6, 6, 0},
                             {6, 6, 0, 0},
                             {0, 0, 0, 0},
                         },
                         {
                             {6, 0, 0, 0},
                             {6, 6, 0, 0},
                             {0, 6, 0, 0},
                             {0, 0, 0, 0},
                         }};
  block* s = (block*)malloc(sizeof(block));
  initBlock(s, Sshape, 2);
  standardBlocks[5] = s;

  // reversed S shape
  int revSshape[2][4][4] = {{
                                {0, 0, 0, 0},
                                {7, 7, 0, 0},
                                {0, 7, 7, 0},
                                {0, 0, 0, 0},
                            },
                            {
                                {0, 7, 0, 0},
                                {7, 7, 0, 0},
                                {7, 0, 0, 0},
                                {0, 0, 0, 0},
                            }};
  block* revs = (block*)malloc(sizeof(block));
  initBlock(revs, revSshape, 2);
  standardBlocks[6] = revs;

  return standardBlocks;
}

SDL_Color blockColors[8] = {
    {0x00, 0x00, 0x00, 0xFF},
    {0x00, 0x00, 0xFF, 0xFF},
    {0xFF, 0xFF, 0x00, 0xFF},
    {0x00, 0xFF, 0x00, 0xFF},
    {0xFF, 0x00, 0x00, 0xFF},
    {0xCB, 0x00, 0xCB, 0xFF},
    {0x00, 0xFF, 0xFF, 0xFF},
    {0xFF, 0x00, 0xFF, 0xFF}
};
