#ifndef RULES_H
#define RULES_H

#include "board.h"

// check_recursive: 1 = enable deep check for False Forbidden (3-3), 0 = shallow check
int is_forbidden(Board *board, int row, int col);

#endif
