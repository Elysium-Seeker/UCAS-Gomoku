#ifndef AI_H
#define AI_H

#include "board.h"
#include <time.h>

// AI Stats
typedef struct {
    double last_move_time;
    double total_time;
    int move_count;
} AIStats;

void init_ai_stats(AIStats *stats);
void ai_get_move(Board *board, int role, int *x, int *y, AIStats *stats);

#endif
