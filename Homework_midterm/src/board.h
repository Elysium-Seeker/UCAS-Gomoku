#ifndef BOARD_H
#define BOARD_H

#define BOARD_SIZE 15

// 棋子类型
#define EMPTY 0
#define BLACK 1
#define WHITE 2

// 棋盘结构体
typedef struct {
    int grid[BOARD_SIZE][BOARD_SIZE];
    int last_move_x;
    int last_move_y;
    int move_count;
} Board;

// 函数原型
void init_board(Board *board);
void print_board(Board *board);
int is_valid_pos(int x, int y);
int is_empty(Board *board, int x, int y);
void place_piece(Board *board, int x, int y, int piece);

int is_empty(Board *board, int x, int y);

#endif
