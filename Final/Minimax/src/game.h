#ifndef GAME_H
#define GAME_H

#include "board.h"

// 游戏状态
#define PLAYING 0
#define BLACK_WINS 1
#define WHITE_WINS 2
#define DRAW 3

// 函数原型
int check_win(Board *board, int x, int y, int piece);
int check_draw(Board *board);
int parse_input(char *input, int *x, int *y);
int get_game_state(Board *board);

#endif
