#ifndef GAME_H
#define GAME_H

#include "board.h"

// 游戏状态
#define PLAYING 0
#define BLACK_WINS 1
#define WHITE_WINS 2
#define DRAW 3

// 函数原型
int check_win(Board *board, int row, int col, int piece);
int check_draw(Board *board);
int parse_input(char *input, int *row, int *col);
int get_game_state(Board *board);
void save_game(Board *board, const char *filename);

#endif
