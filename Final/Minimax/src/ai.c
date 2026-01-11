#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include "ai.h"
#include "rules.h"
#include "game.h"

// --- 评分常量 ---
#define SCORE_FIVE       10000000
#define SCORE_LIVE_FOUR  1000000
#define SCORE_DEAD_FOUR  100000
#define SCORE_LIVE_THREE 50000
#define SCORE_DEAD_THREE 5000
#define SCORE_LIVE_TWO   500
#define SCORE_DEAD_TWO   50
#define SCORE_LIVE_ONE   10

// 搜索参数
#define MAX_SEARCH_DEPTH 12
#define TIME_LIMIT_MS 15000  // 15秒时间限制
#define CANDIDATE_LIMIT 15   // 候选着法数量限制

// ==================== 置换表 ====================
#define TT_SIZE (1 << 20)  // 约100万条目
#define TT_MASK (TT_SIZE - 1)

typedef enum { TT_EXACT, TT_ALPHA, TT_BETA } TTFlag;

typedef struct {
    uint64_t hash;
    int depth;
    int score;
    TTFlag flag;
    int best_x, best_y;
} TTEntry;

static TTEntry transposition_table[TT_SIZE];
static uint64_t zobrist_table[BOARD_SIZE][BOARD_SIZE][3]; // 0=empty, 1=black, 2=white
static uint64_t current_hash = 0;
static int zobrist_initialized = 0;

// 时间控制
static clock_t search_start_time;
static int time_up = 0;

// 历史启发表
static int history_table[2][BOARD_SIZE][BOARD_SIZE]; // [side][x][y]

// 方向数组
static int dirs[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

void init_ai_stats(AIStats *stats) {
    stats->last_move_time = 0.0;
    stats->total_time = 0.0;
    stats->move_count = 0;
}

// 初始化Zobrist哈希表
static void init_zobrist(void) {
    if (zobrist_initialized) return;
    srand(12345); // 固定种子以保证一致性
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int k = 0; k < 3; k++) {
                zobrist_table[i][j][k] = ((uint64_t)rand() << 32) | rand();
            }
        }
    }
    zobrist_initialized = 1;
}

// 计算棋盘哈希值
static uint64_t compute_hash(Board *board) {
    uint64_t hash = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            int piece = board->grid[i][j];
            hash ^= zobrist_table[i][j][piece];
        }
    }
    return hash;
}

// 增量更新哈希
static void update_hash(int x, int y, int old_piece, int new_piece) {
    current_hash ^= zobrist_table[x][y][old_piece];
    current_hash ^= zobrist_table[x][y][new_piece];
}

// 检查时间
static int check_time(void) {
    clock_t now = clock();
    double elapsed = (double)(now - search_start_time) * 1000.0 / CLOCKS_PER_SEC;
    return elapsed >= TIME_LIMIT_MS;
}

// 查询置换表
static TTEntry* tt_probe(uint64_t hash) {
    TTEntry *entry = &transposition_table[hash & TT_MASK];
    if (entry->hash == hash) {
        return entry;
    }
    return NULL;
}

// 存储置换表
static void tt_store(uint64_t hash, int depth, int score, TTFlag flag, int best_x, int best_y) {
    TTEntry *entry = &transposition_table[hash & TT_MASK];
    // 深度替换策略：更深的搜索结果优先
    if (entry->hash != hash || entry->depth <= depth) {
        entry->hash = hash;
        entry->depth = depth;
        entry->score = score;
        entry->flag = flag;
        entry->best_x = best_x;
        entry->best_y = best_y;
    }
}

// 读取某方向的棋型（返回5格内的棋子序列）
// pattern[0..4] = 正向5格, pattern[5..9] = 反向5格 (不含中心)
static void get_line_pattern(Board *board, int x, int y, int dx, int dy, 
                              int role, int pattern[10]) {
    int opponent = (role == BLACK) ? WHITE : BLACK;
    // 正向扫描
    for (int i = 0; i < 5; i++) {
        int nx = x + (i+1) * dx, ny = y + (i+1) * dy;
        if (!is_valid_pos(nx, ny)) pattern[i] = -1;  // 边界
        else if (board->grid[nx][ny] == role) pattern[i] = 1;  // 己方
        else if (board->grid[nx][ny] == opponent) pattern[i] = -1;  // 对方
        else pattern[i] = 0;  // 空
    }
    // 反向扫描
    for (int i = 0; i < 5; i++) {
        int nx = x - (i+1) * dx, ny = y - (i+1) * dy;
        if (!is_valid_pos(nx, ny)) pattern[5+i] = -1;
        else if (board->grid[nx][ny] == role) pattern[5+i] = 1;
        else if (board->grid[nx][ny] == opponent) pattern[5+i] = -1;
        else pattern[5+i] = 0;
    }
}

// 评估某点在某方向上的棋型（增强版：识别跳棋型）
static int evaluate_line(Board *board, int x, int y, int dx, int dy, int role) {
    int pattern[10];
    get_line_pattern(board, x, y, dx, dy, role, pattern);
    
    // 连续棋子计数（双向）
    int count_f = 0, count_b = 0;
    int space_f = -1, space_b = -1;  // 跳空位置
    int blocked_f = 0, blocked_b = 0;
    int open_f = 0, open_b = 0;
    
    // 正向：统计连续+跳过一个空格后的连续
    for (int i = 0; i < 5; i++) {
        if (pattern[i] == 1) {
            count_f++;
        } else if (pattern[i] == 0) {
            if (space_f < 0 && i < 4) {
                space_f = i;  // 记录第一个空格
                // 检查空格后是否还有己方棋子
                int extra = 0;
                for (int j = i+1; j < 5 && j < i+4; j++) {
                    if (pattern[j] == 1) extra++;
                    else break;
                }
                if (extra > 0) count_f += extra;
            }
            open_f = 1;
            break;
        } else {
            blocked_f = 1;
            break;
        }
    }
    
    // 反向：同样处理
    for (int i = 0; i < 5; i++) {
        if (pattern[5+i] == 1) {
            count_b++;
        } else if (pattern[5+i] == 0) {
            if (space_b < 0 && i < 4) {
                space_b = i;
                int extra = 0;
                for (int j = i+1; j < 5 && j < i+4; j++) {
                    if (pattern[5+j] == 1) extra++;
                    else break;
                }
                if (extra > 0) count_b += extra;
            }
            open_b = 1;
            break;
        } else {
            blocked_b = 1;
            break;
        }
    }
    
    int total = 1 + count_f + count_b;  // 包含中心点
    int open_ends = open_f + open_b;
    int has_jump = (space_f >= 0 && space_f < count_f) || (space_b >= 0 && space_b < count_b);
    
    // 评分
    if (total >= 5) return SCORE_FIVE;
    
    if (total == 4) {
        if (has_jump) {
            // 跳四（如X_XXX）相当于活四威胁
            return open_ends >= 1 ? SCORE_DEAD_FOUR : SCORE_DEAD_FOUR / 2;
        }
        if (open_ends == 2) return SCORE_LIVE_FOUR;
        if (open_ends == 1) return SCORE_DEAD_FOUR;
    }
    
    if (total == 3) {
        if (has_jump) {
            // 跳三（如X_XX）比普通活三稍弱但仍是威胁
            return open_ends == 2 ? SCORE_LIVE_THREE * 8 / 10 : SCORE_DEAD_THREE;
        }
        if (open_ends == 2) return SCORE_LIVE_THREE;
        if (open_ends == 1) return SCORE_DEAD_THREE;
    }
    
    if (total == 2) {
        if (open_ends == 2) return SCORE_LIVE_TWO;
        if (open_ends == 1) return SCORE_DEAD_TWO;
    }
    
    if (total == 1 && open_ends == 2) return SCORE_LIVE_ONE;
    
    return 0;
}

// 评估某点对于某方的分数（增强版：识别组合威胁）
static int evaluate_point_score(Board *board, int x, int y, int role) {
    int line_scores[4];
    int live_three_count = 0;
    int dead_four_count = 0;
    int live_four = 0;
    int total = 0;
    
    for (int d = 0; d < 4; d++) {
        line_scores[d] = evaluate_line(board, x, y, dirs[d][0], dirs[d][1], role);
        total += line_scores[d];
        
        // 统计威胁类型
        if (line_scores[d] >= SCORE_LIVE_FOUR) live_four = 1;
        else if (line_scores[d] >= SCORE_DEAD_FOUR) dead_four_count++;
        else if (line_scores[d] >= SCORE_LIVE_THREE * 7 / 10) live_three_count++;  // 包括跳三
    }
    
    // 组合威胁加成
    if (live_four) {
        return total;  // 活四已经足够强
    }
    
    // 双冲四 = 必胜
    if (dead_four_count >= 2) {
        total += SCORE_LIVE_FOUR;  // 双冲四相当于活四
    }
    // 冲四 + 活三 = 必胜
    else if (dead_four_count >= 1 && live_three_count >= 1) {
        total += SCORE_LIVE_FOUR / 2;  // 冲四活三非常强
    }
    // 双活三 = 很强的威胁
    else if (live_three_count >= 2) {
        total += SCORE_DEAD_FOUR;  // 双活三相当于冲四
    }
    
    return total;
}

// 全局评估函数
static int evaluate_board(Board *board, int role) {
    int opponent = (role == BLACK) ? WHITE : BLACK;
    int my_score = 0, opp_score = 0;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board->grid[i][j] == role) {
                my_score += evaluate_point_score(board, i, j, role);
            } else if (board->grid[i][j] == opponent) {
                opp_score += evaluate_point_score(board, i, j, opponent);
            }
        }
    }
    
    return my_score - opp_score;
}

// 候选着法结构
typedef struct { 
    int x, y, score; 
} Move;

// 快速排序候选着法
static void sort_moves(Move *moves, int n) {
    // 使用插入排序（对于小数组效率高）
    for (int i = 1; i < n; i++) {
        Move key = moves[i];
        int j = i - 1;
        while (j >= 0 && moves[j].score < key.score) {
            moves[j + 1] = moves[j];
            j--;
        }
        moves[j + 1] = key;
    }
}

// 生成候选着法
static int generate_moves(Board *board, Move *candidates, int role, int is_maximizing, 
                          int tt_best_x, int tt_best_y) {
    int opponent = (role == BLACK) ? WHITE : BLACK;
    int current_player = is_maximizing ? role : opponent;
    int cand_count = 0;
    
    // 首先检查是否有立即获胜或必须防守的点
    int urgent_attack_x = -1, urgent_attack_y = -1;
    int urgent_defend_x = -1, urgent_defend_y = -1;
    int best_attack_score = 0, best_defend_score = 0;
    
    // 追踪对手的活三威胁点
    int live_three_threats[20][2];
    int live_three_count = 0;
    
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (board->grid[i][j] != EMPTY) continue;
            
            // 启发式剪枝：只考虑周围2格内有子的位置
            int has_neighbor = 0;
            for (int dx = -2; dx <= 2 && !has_neighbor; dx++) {
                for (int dy = -2; dy <= 2 && !has_neighbor; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = i + dx, ny = j + dy;
                    if (is_valid_pos(nx, ny) && board->grid[nx][ny] != EMPTY) {
                        has_neighbor = 1;
                    }
                }
            }
            if (!has_neighbor) continue;
            
            // 禁手检查
            if (current_player == BLACK && is_forbidden(board, i, j)) continue;
            
            // 评估此点
            int my_score = evaluate_point_score(board, i, j, current_player);
            int opp_score = evaluate_point_score(board, i, j, 
                            current_player == BLACK ? WHITE : BLACK);
            
            // 检查紧急着法
            if (my_score >= SCORE_FIVE) {
                // 立即获胜
                candidates[0] = (Move){i, j, my_score * 2};
                return 1;
            }
            if (my_score >= SCORE_LIVE_FOUR && my_score > best_attack_score) {
                best_attack_score = my_score;
                urgent_attack_x = i; urgent_attack_y = j;
            }
            if (opp_score >= SCORE_LIVE_FOUR && opp_score > best_defend_score) {
                best_defend_score = opp_score;
                urgent_defend_x = i; urgent_defend_y = j;
            }
            
            // 追踪对手的活三威胁（防守重要！）
            if (opp_score >= SCORE_LIVE_THREE && opp_score < SCORE_LIVE_FOUR) {
                if (live_three_count < 20) {
                    live_three_threats[live_three_count][0] = i;
                    live_three_threats[live_three_count][1] = j;
                    live_three_count++;
                }
            }
            
            // 综合评分 = 我方得分 + 阻止对手得分 + 历史启发
            // 提高防守权重：对手威胁越大，防守越重要
            double defend_weight = 1.1;  // 防守权重提升到1.1（略高于进攻）
            if (opp_score >= SCORE_DEAD_FOUR) defend_weight = 1.5;  // 冲四必须防
            else if (opp_score >= SCORE_LIVE_THREE) defend_weight = 1.3;  // 活三要重视
            
            int combined = my_score + (int)(opp_score * defend_weight) + 
                          history_table[current_player == BLACK ? 0 : 1][i][j];
            
            // 置换表最佳着法加分
            if (i == tt_best_x && j == tt_best_y) {
                combined += 10000000;
            }
            
            candidates[cand_count++] = (Move){i, j, combined};
        }
    }
    
    // 如果有紧急进攻点，优先返回
    if (urgent_attack_x >= 0) {
        candidates[0] = (Move){urgent_attack_x, urgent_attack_y, best_attack_score * 2};
        return 1;
    }
    
    // 如果对手有威胁，必须防守
    if (urgent_defend_x >= 0) {
        for (int i = 0; i < cand_count; i++) {
            if (candidates[i].x == urgent_defend_x && candidates[i].y == urgent_defend_y) {
                candidates[i].score += 5000000; // 大幅提升防守优先级
            }
        }
    }
    
    // 如果对手有多个活三威胁点，大幅提升这些点的防守优先级
    // 因为活三如果不堵，下一步就变活四必胜
    if (live_three_count > 0) {
        for (int t = 0; t < live_three_count; t++) {
            for (int i = 0; i < cand_count; i++) {
                if (candidates[i].x == live_three_threats[t][0] && 
                    candidates[i].y == live_three_threats[t][1]) {
                    candidates[i].score += 500000; // 提升活三防守优先级
                }
            }
        }
    }
    
    // 排序
    sort_moves(candidates, cand_count);
    
    return cand_count;
}

// 带Alpha-Beta剪枝的Minimax搜索
static int minimax(Board *board, int depth, int alpha, int beta, 
                   int is_maximizing, int role, int *move_x, int *move_y) {
    // 时间检查
    if (time_up || (depth < MAX_SEARCH_DEPTH - 2 && check_time())) {
        time_up = 1;
        return 0;
    }
    
    int opponent = (role == BLACK) ? WHITE : BLACK;
    
    // 空盘下中心
    if (board->move_count == 0) {
        if (move_x) { *move_x = 7; *move_y = 7; }
        return 0;
    }
    
    // 查询置换表
    int tt_best_x = -1, tt_best_y = -1;
    TTEntry *tt_entry = tt_probe(current_hash);
    if (tt_entry) {
        tt_best_x = tt_entry->best_x;
        tt_best_y = tt_entry->best_y;
        if (tt_entry->depth >= depth) {
            if (tt_entry->flag == TT_EXACT) {
                if (move_x) { *move_x = tt_best_x; *move_y = tt_best_y; }
                return tt_entry->score;
            }
            if (tt_entry->flag == TT_ALPHA && tt_entry->score <= alpha) {
                return alpha;
            }
            if (tt_entry->flag == TT_BETA && tt_entry->score >= beta) {
                return beta;
            }
        }
    }
    
    // 生成候选着法
    Move candidates[225];
    int cand_count = generate_moves(board, candidates, role, is_maximizing, tt_best_x, tt_best_y);
    
    if (cand_count == 0) return 0;
    
    // 限制候选数量
    int limit = (cand_count > CANDIDATE_LIMIT) ? CANDIDATE_LIMIT : cand_count;
    
    // 叶子节点评估
    if (depth == 0) {
        return evaluate_board(board, role);
    }
    
    int best_x = candidates[0].x;
    int best_y = candidates[0].y;
    TTFlag tt_flag = TT_ALPHA;
    
    if (is_maximizing) {
        int max_eval = -2000000000;
        
        for (int i = 0; i < limit; i++) {
            int cx = candidates[i].x;
            int cy = candidates[i].y;
            
            // 落子
            board->grid[cx][cy] = role;
            update_hash(cx, cy, EMPTY, role);
            board->move_count++;
            
            // 检查获胜
            if (check_win(board, cx, cy, role)) {
                board->grid[cx][cy] = EMPTY;
                update_hash(cx, cy, role, EMPTY);
                board->move_count--;
                if (move_x) { *move_x = cx; *move_y = cy; }
                return SCORE_FIVE + depth * 1000; // 越快获胜越好
            }
            
            int eval = minimax(board, depth - 1, alpha, beta, 0, role, NULL, NULL);
            
            // 撤销
            board->grid[cx][cy] = EMPTY;
            update_hash(cx, cy, role, EMPTY);
            board->move_count--;
            
            if (time_up) return 0;
            
            if (eval > max_eval) {
                max_eval = eval;
                best_x = cx;
                best_y = cy;
            }
            
            if (eval > alpha) {
                alpha = eval;
                tt_flag = TT_EXACT;
            }
            if (beta <= alpha) {
                // 更新历史启发表
                history_table[role == BLACK ? 0 : 1][cx][cy] += depth * depth;
                tt_flag = TT_BETA;
                break;
            }
        }
        
        if (move_x) { *move_x = best_x; *move_y = best_y; }
        tt_store(current_hash, depth, max_eval, tt_flag, best_x, best_y);
        return max_eval;
        
    } else {
        int min_eval = 2000000000;
        
        for (int i = 0; i < limit; i++) {
            int cx = candidates[i].x;
            int cy = candidates[i].y;
            
            board->grid[cx][cy] = opponent;
            update_hash(cx, cy, EMPTY, opponent);
            board->move_count++;
            
            if (check_win(board, cx, cy, opponent)) {
                board->grid[cx][cy] = EMPTY;
                update_hash(cx, cy, opponent, EMPTY);
                board->move_count--;
                return -(SCORE_FIVE + depth * 1000);
            }
            
            int eval = minimax(board, depth - 1, alpha, beta, 1, role, NULL, NULL);
            
            board->grid[cx][cy] = EMPTY;
            update_hash(cx, cy, opponent, EMPTY);
            board->move_count--;
            
            if (time_up) return 0;
            
            if (eval < min_eval) {
                min_eval = eval;
                best_x = cx;
                best_y = cy;
            }
            
            if (eval < beta) {
                beta = eval;
                tt_flag = TT_EXACT;
            }
            if (beta <= alpha) {
                history_table[opponent == BLACK ? 0 : 1][cx][cy] += depth * depth;
                tt_flag = TT_ALPHA;
                break;
            }
        }
        
        tt_store(current_hash, depth, min_eval, tt_flag, best_x, best_y);
        return min_eval;
    }
}

// 迭代加深搜索
void ai_get_move(Board *board, int role, int *out_x, int *out_y, AIStats *stats) {
    search_start_time = clock();
    time_up = 0;
    
    // 初始化
    init_zobrist();
    current_hash = compute_hash(board);
    memset(history_table, 0, sizeof(history_table));
    
    // 清空置换表，避免不同轮次之间的分数混淆
    // 因为置换表的 hash 没有包含"当前轮到谁下"的信息
    memset(transposition_table, 0, sizeof(transposition_table));
    
    int best_x = 7, best_y = 7;
    int best_score = 0;
    int final_depth = 0;
    
    // 迭代加深
    for (int depth = 2; depth <= MAX_SEARCH_DEPTH; depth += 2) {
        int move_x = -1, move_y = -1;
        int score = minimax(board, depth, -2000000000, 2000000000, 1, role, &move_x, &move_y);
        
        if (time_up) {
            printf("  Depth %d: time limit reached\n", depth);
            break;
        }
        
        if (move_x >= 0) {
            best_x = move_x;
            best_y = move_y;
            best_score = score;
            final_depth = depth;
        }
        
        printf("  Depth %d: (%c%d) score=%d\n", depth, 'A' + move_y, move_x + 1, score);
        
        // 如果找到必胜，提前退出
        if (score >= SCORE_FIVE) {
            printf("  Found winning move!\n");
            break;
        }
        
        // 检查时间
        if (check_time()) break;
    }
    
    // 异常兜底
    if (best_x == -1) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board->grid[i][j] == EMPTY) {
                    best_x = i; best_y = j;
                    goto done;
                }
            }
        }
    }
    
done:
    clock_t end_time = clock();
    double time_spent = (double)(end_time - search_start_time) / CLOCKS_PER_SEC;
    
    stats->last_move_time = time_spent;
    stats->total_time += time_spent;
    stats->move_count++;

    *out_x = best_x;
    *out_y = best_y;

    printf("AI (Minimax D=%d) placed at %c%d\n", final_depth, 'A' + best_y, best_x + 1);
    printf("Time: %.2fs (Avg: %.2fs)\n", time_spent, stats->total_time / stats->move_count);
}
