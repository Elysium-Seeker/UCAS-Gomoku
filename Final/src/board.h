#ifndef BOARD_H
#define BOARD_H

#define BOARD_SIZE 15

// 棋子类型
#define EMPTY 0
#define BLACK 1
#define WHITE 2

typedef struct {
    int row;
    int col;
    int player;
} MoveRecord;

// 棋盘结构体
typedef struct {
    int grid[BOARD_SIZE][BOARD_SIZE];
    int last_move_row;
    int last_move_col;
    MoveRecord history[BOARD_SIZE * BOARD_SIZE];
    int move_count;
} Board;

// 函数原型
void init_board(Board *board);
void print_board(Board *board);
int is_valid_pos(int row, int col);
int is_empty(Board *board, int row, int col);
void place_piece(Board *board, int row, int col, int piece);
int undo_move(Board *board); // Returns 1 if successful, 0 if empty

int is_empty(Board *board, int row, int col);

#endif
