#include <stdio.h>
#include "rules.h"
#include "game.h"

//主要是判断禁手，最麻烦的部分
// 方向：水平, 垂直, 对角线 \, 对角线 /
static int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

// 辅助函数：获取指定位置的棋子，边界返回 -1
static int get_stone(Board *board, int r, int c) {
    if (!is_valid_pos(r, c)) return -1; // 边界
    return board->grid[r][c];
}

// 前向声明
int is_forbidden_recursive(Board *board, int x, int y, int check_recursive);

// 提取以 x,y 为中心的线 (范围：-5 到 +5)
static void get_line_info(Board *board, int x, int y, int dx, int dy, 
                          int *line, int *r_pos, int *c_pos) {
    for (int i = -5; i <= 5; i++) {
        int r = x + i * dx;
        int c = y + i * dy;
        int s = get_stone(board, r, c);
        if (i == 0) s = BLACK; // 假设当前落子为黑
        line[i + 5] = s;
        if (r_pos) r_pos[i + 5] = r;
        if (c_pos) c_pos[i + 5] = c;
    }
}

// 1. 检查长连和五连
static void check_five_long(int *line, int *is_five, int *is_long) {
    int center = 5;
    int count = 1;
    int l = center - 1;
    while (l >= 0 && line[l] == BLACK) { count++; l--; }
    int r = center + 1;
    while (r <= 10 && line[r] == BLACK) { count++; r++; }
    if (count == 5) *is_five = 1;
    if (count > 5) *is_long = 1;
}

// 2. 检查四 (返回数量)
static int count_fours(int *line) {
    int cnt = 0;
    int last_four_mask = 0;
    for (int s = 1; s <= 5; s++) {
        int b_cnt = 0;
        int e_cnt = 0;
        int current_mask = 0;
        for (int k = 0; k < 5; k++) {
            if (line[s+k] == BLACK) {
                b_cnt++;
                current_mask |= (1 << (s+k));
            }
            else if (line[s+k] == EMPTY) { e_cnt++; }
            else { b_cnt = -99; } 
        }

        if (b_cnt == 4 && e_cnt == 1) {
            int valid_five = 1;
            if (s > 0 && line[s-1] == BLACK) valid_five = 0;
            if (s+5 <= 10 && line[s+5] == BLACK) valid_five = 0;

            if (valid_five) {
                // 检查这是否与上一个四连是同一组棋子
                if (current_mask != last_four_mask) {
                    cnt++;
                    last_four_mask = current_mask;
                }
            }
        }
    }
    return cnt;
}

// 3. 检查活三 (返回数量) 感谢gemini 帮我理清思路，复杂禁手的判断属实困难。
static int count_threes(Board *board, int x, int y, int *line, int *r_pos, int *c_pos, int check_recursive) {
    int cnt = 0, center = 5;
    for (int s = 1; s <= 4; s++) { 
        if (line[s] != EMPTY || line[s+5] != EMPTY) continue;
        int b_cnt = 0, e_cnt = 0, has_center = 0, key_spot_idx = -1;
        for (int k = 1; k <= 4; k++) {
            int idx = s + k;
            if (line[idx] == BLACK) {
                b_cnt++;
                if (idx == center) has_center = 1;
            } 
                else if (line[idx] == EMPTY) e_cnt++,key_spot_idx = idx;
                    else b_cnt = -99;
        }
        if (b_cnt == 3 && e_cnt == 1 && has_center) {
            // 发现活三候选。
            int is_valid_three = 1;
            if (check_recursive) {
                board->grid[x][y] = BLACK; // 放置假棋子用来检验
                // 检查 key_spot 是否禁手 (递归调用，但 check_recursive=0 防止无限递归)
                if (is_forbidden_recursive(board, r_pos[key_spot_idx], c_pos[key_spot_idx], 0)) {
                    is_valid_three = 0;
                }
                board->grid[x][y] = EMPTY; // 恢复
            }
            if (is_valid_three) cnt++;
        }
    }
    return cnt;
}

// 分析线条
// 检查指定方向上的连子情况，识别长连、五连、四连、活三等
static void analyze_line(Board *board, int x, int y, int dx, int dy, int check_recursive,
                         int *is_five, int *is_long, int *four_cnt, int *three_cnt) {
    *is_five = 0;*is_long = 0;*four_cnt = 0;*three_cnt = 0;
    int line[11], r_pos[11], c_pos[11]; // 存储列坐标
    get_line_info(board, x, y, dx, dy, line, r_pos, c_pos);
    check_five_long(line, is_five, is_long);
    if (*is_five || *is_long) return;
    *four_cnt = count_fours(line);
    *three_cnt = count_threes(board, x, y, line, r_pos, c_pos, check_recursive);
}

// 递归检查禁手
// 核心逻辑：检查某个位置是否构成禁手（长连、双四、双三）
// check_recursive 参数用于控制是否进行深层递归检查（防止假活三）
int is_forbidden_recursive(Board *board, int x, int y, int check_recursive) {
    if (board->grid[x][y] != EMPTY) return 0;
    int total_five = 0, total_long = 0, total_fours = 0, total_threes = 0;
    for (int i = 0; i < 4; i++) {
        int f = 0, l = 0, f4 = 0, t3 = 0;
        analyze_line(board, x, y, directions[i][0], directions[i][1], check_recursive, &f, &l, &f4, &t3);
        
        if (f) total_five = 1;
        if (l) total_long = 1;
        total_fours += f4;
        total_threes += t3;
    }
    if (total_five) return 0;
    if (total_long) return 1;
    if (total_fours >= 2) return 1;
    if (total_threes >= 2) return 1;
    return 0;
}

// 检查禁手（对外接口）
// 判断黑棋在 (x, y) 落子是否违规
int is_forbidden(Board *board, int x, int y) {
    return is_forbidden_recursive(board, x, y, 1);
}
