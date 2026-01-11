#include <stdio.h>
#include "rules.h"
#include "game.h"

// 禁手逻辑判断核心模块
// 主要禁手类型：长连 (>5子), 双四 (Double Four), 双三 (Double Three)

// 四个检查方向：水平, 垂直, 主对角线(\), 副对角线(/)
static int directions[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

// 辅助函数：安全获取棋盘位置状态
static int get_stone(Board *board, int r, int c) {
    if (!is_valid_pos(r, c)) return -1; // 越界
    return board->grid[r][c];
}

// 前向声明：递归禁手检查函数
int is_forbidden_recursive(Board *board, int x, int y, int check_recursive);

// --- 提取线条信息 ---
// 获取以 (x,y) 为中心，半径为 5 的线段上的棋子分布
static void get_line_info(Board *board, int x, int y, int dx, int dy, 
                          int *line, int *r_pos, int *c_pos) {
    for (int i = -5; i <= 5; i++) {
        int r = x + i * dx;
        int c = y + i * dy;
        int s = get_stone(board, r, c);
        if (i == 0) s = BLACK; // 假设当前点落子为黑
        line[i + 5] = s;
        if (r_pos) r_pos[i + 5] = r;
        if (c_pos) c_pos[i + 5] = c;
    }
}

// --- 1. 检查长连和五连 ---
static void check_five_long(int *line, int *is_five, int *is_long) {
    int center = 5;
    int count = 1;
    // 向左统计
    int l = center - 1;
    while (l >= 0 && line[l] == BLACK) { count++; l--; }
    // 向右统计
    int r = center + 1;
    while (r <= 10 && line[r] == BLACK) { count++; r++; }
    
    if (count == 5) *is_five = 1; // 构成五连
    if (count > 5) *is_long = 1;  // 构成长连 (禁手)
}

// --- 2. 统计"四"的数量 ---
// 包括活四 (Live Four) 和冲四 (Dead Four)
static int count_fours(int *line) {
    int cnt = 0;
    int last_four_mask = 0; // 用于避免重复统计同一组棋子
    
    // 扫描长度为 5 的窗口
    for (int s = 1; s <= 5; s++) {
        int b_cnt = 0; // 黑子数
        int e_cnt = 0; // 空位数
        int current_mask = 0;
        
        for (int k = 0; k < 5; k++) {
            if (line[s+k] == BLACK) {
                b_cnt++;
                current_mask |= (1 << (s+k));
            }
            else if (line[s+k] == EMPTY) { e_cnt++; }
            else { b_cnt = -99; } // 被白子阻断
        }

        // 4黑 + 1空
        if (b_cnt == 4 && e_cnt == 1) {
            int valid_five = 1;
            // 确保不是长连的一部分
            if (s > 0 && line[s-1] == BLACK) valid_five = 0; 
            if (s+5 <= 10 && line[s+5] == BLACK) valid_five = 0;

            if (valid_five) {
                // 如果是新的一组四
                if (current_mask != last_four_mask) {
                    cnt++;
                    last_four_mask = current_mask;
                }
            }
        }
    }
    return cnt;
}

// --- 3. 统计"活三"的数量 ---
// 活三定义：能够在下一步形成活四的三
// 需要排除因禁手而无法落子的"假活三"
// 注意：同一条线上最多只能有一个活三，避免重复计算
static int count_threes(Board *board, int x, int y, int *line, int *r_pos, int *c_pos, int check_recursive) {
    int cnt = 0, center = 5;
    int found_three_mask = 0; // 用于避免重复统计同一组棋子
    
    // 扫描长度为 5 的窗口，寻找 "黑黑黑空" 或 "黑空黑黑" 等模式
    // 简化为：窗口内 3 黑 + 1~2 空 (标准定义参考 Renju 规则)
    
    for (int s = 1; s <= 4; s++) { 
        // 边界必须未被阻断
        if (line[s] != EMPTY || line[s+5] != EMPTY) continue;
        
        int b_cnt = 0, e_cnt = 0, has_center = 0, key_spot_idx = -1;
        int current_mask = 0;
        
        for (int k = 1; k <= 4; k++) {
            int idx = s + k;
            if (line[idx] == BLACK) {
                b_cnt++;
                current_mask |= (1 << idx); // 记录黑子位置
                if (idx == center) has_center = 1; // 必须包含当前落子点
            } 
                else if (line[idx] == EMPTY) e_cnt++,key_spot_idx = idx; // 记录空位位置
                    else b_cnt = -99;
        }
        
        if (b_cnt == 3 && e_cnt == 1 && has_center) {
            // 检查是否已经统计过这组棋子（避免重复计算）
            if ((current_mask & found_three_mask) == current_mask) continue;
            
            // 检查关键空位是否为禁手点 (若是禁手，则无法形成活四，即为假活三)
            int is_valid_three = 1;
            if (check_recursive) {
                board->grid[x][y] = BLACK; // 模拟落子
                
                // 递归调用：检查关键点是否禁手
                // check_recursive=0 避免无限递归
                if (is_forbidden_recursive(board, r_pos[key_spot_idx], c_pos[key_spot_idx], 0)) {
                    is_valid_three = 0; 
                }
                board->grid[x][y] = EMPTY; // 恢复
            }
            if (is_valid_three) {
                cnt++;
                found_three_mask |= current_mask; // 标记已统计
            }
        }
    }
    return cnt;
}

// --- 综合分析单条线 ---
static void analyze_line(Board *board, int x, int y, int dx, int dy, int check_recursive,
                         int *is_five, int *is_long, int *four_cnt, int *three_cnt) {
    *is_five = 0;*is_long = 0;*four_cnt = 0;*three_cnt = 0;
    int line[11], r_pos[11], c_pos[11]; 
    
    get_line_info(board, x, y, dx, dy, line, r_pos, c_pos);
    
    check_five_long(line, is_five, is_long);
    if (*is_five || *is_long) return; // 已成五或长连，无需继续统计
    
    *four_cnt = count_fours(line);
    *three_cnt = count_threes(board, x, y, line, r_pos, c_pos, check_recursive);
}

// --- 禁手递归判定入口 ---
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
    
    // 判定优先级：五连 > 长连 > 双四/双三
    if (total_five) return 0; // 只要成五，即获胜 (包括长连带五连的情况，在此简化处理，以五连为准)
    if (total_long) return 1; // 长连禁手
    if (total_fours >= 2) return 1; // 双四禁手
    if (total_threes >= 2) return 1; // 双三禁手
    
    return 0; // 合法
}

// --- 外部接口 ---
// 判断点 (x, y) 是否为黑方禁手
int is_forbidden(Board *board, int x, int y) {
    return is_forbidden_recursive(board, x, y, 1); 
}
