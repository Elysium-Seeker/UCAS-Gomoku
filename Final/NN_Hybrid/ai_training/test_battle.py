"""
测试优化后的 Minimax vs 混合AI
"""
import torch
import numpy as np
import time
import os
import sys

BOARD_SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2

# ============ 棋盘显示 ============
def print_board(board, last_move=None):
    """打印棋盘"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("   ", end="")
    for c in range(BOARD_SIZE):
        print(f" {chr(ord('A') + c)}", end="")
    print()
    
    for r in range(BOARD_SIZE):
        print(f"{r+1:2d} ", end="")
        for c in range(BOARD_SIZE):
            if board[r, c] == BLACK:
                if last_move and last_move == (r, c):
                    print(" ◆", end="")
                else:
                    print(" ●", end="")
            elif board[r, c] == WHITE:
                if last_move and last_move == (r, c):
                    print(" ◇", end="")
                else:
                    print(" ○", end="")
            else:
                print(" ·", end="")
        print(f" {r+1}")
    
    print("   ", end="")
    for c in range(BOARD_SIZE):
        print(f" {chr(ord('A') + c)}", end="")
    print("\n")

# ============ 模型定义 ============
class GomokuNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = torch.nn.Linear(256, 1)  # value head
        self.fc_policy = torch.nn.Linear(256, BOARD_SIZE * BOARD_SIZE)  # policy head
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc2(x))
        return policy, value

# ============ 规则检查 ============
def is_valid_pos(x, y):
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

def check_win(board, x, y):
    if board[x, y] == EMPTY:
        return False
    role = board[x, y]
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in dirs:
        count = 1
        for i in range(1, 5):
            nx, ny = x + i*dx, y + i*dy
            if is_valid_pos(nx, ny) and board[nx, ny] == role:
                count += 1
            else:
                break
        for i in range(1, 5):
            nx, ny = x - i*dx, y - i*dy
            if is_valid_pos(nx, ny) and board[nx, ny] == role:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False

def count_line(board, x, y, dx, dy, role):
    """统计某方向的连子数"""
    count = 0
    for i in range(1, 6):
        nx, ny = x + i*dx, y + i*dy
        if is_valid_pos(nx, ny) and board[nx, ny] == role:
            count += 1
        else:
            break
    return count

def is_overline(board, x, y):
    """检查长连"""
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in dirs:
        count = 1 + count_line(board, x, y, dx, dy, BLACK) + count_line(board, x, y, -dx, -dy, BLACK)
        if count > 5:
            return True
    return False

def count_live_three(board, x, y):
    """统计活三数量"""
    count = 0
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in dirs:
        # 简化检测
        line_count = 1 + count_line(board, x, y, dx, dy, BLACK) + count_line(board, x, y, -dx, -dy, BLACK)
        if line_count == 3:
            # 检查两端
            nx1, ny1 = x + (count_line(board, x, y, dx, dy, BLACK) + 1) * dx, y + (count_line(board, x, y, dx, dy, BLACK) + 1) * dy
            nx2, ny2 = x - (count_line(board, x, y, -dx, -dy, BLACK) + 1) * dx, y - (count_line(board, x, y, -dx, -dy, BLACK) + 1) * dy
            open1 = is_valid_pos(nx1, ny1) and board[nx1, ny1] == EMPTY
            open2 = is_valid_pos(nx2, ny2) and board[nx2, ny2] == EMPTY
            if open1 and open2:
                count += 1
    return count

def count_four(board, x, y):
    """统计四的数量"""
    count = 0
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in dirs:
        line_count = 1 + count_line(board, x, y, dx, dy, BLACK) + count_line(board, x, y, -dx, -dy, BLACK)
        if line_count == 4:
            count += 1
    return count

def is_forbidden(board, x, y):
    """检查禁手（仅黑方）"""
    if board[x, y] != EMPTY:
        return False
    
    board[x, y] = BLACK
    
    # 长连禁手
    if is_overline(board, x, y):
        board[x, y] = EMPTY
        return True
    
    # 检查是否形成五连
    if check_win(board, x, y):
        board[x, y] = EMPTY
        return False
    
    # 双四禁手
    if count_four(board, x, y) >= 2:
        board[x, y] = EMPTY
        return True
    
    # 双三禁手
    if count_live_three(board, x, y) >= 2:
        board[x, y] = EMPTY
        return True
    
    board[x, y] = EMPTY
    return False

# ============ 评估函数 ============
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

SCORE_FIVE = 10000000
SCORE_LIVE_FOUR = 1000000
SCORE_DEAD_FOUR = 100000
SCORE_LIVE_THREE = 50000
SCORE_DEAD_THREE = 5000
SCORE_LIVE_TWO = 500
SCORE_DEAD_TWO = 50

def evaluate_point(board, x, y, role):
    """评估某点对于某方的分数"""
    total = 0
    for dx, dy in DIRS:
        count = 1
        open_ends = 0
        
        # 正向
        for i in range(1, 5):
            nx, ny = x + i*dx, y + i*dy
            if not is_valid_pos(nx, ny):
                break
            if board[nx, ny] == role:
                count += 1
            elif board[nx, ny] == EMPTY:
                open_ends += 1
                break
            else:
                break
        
        # 反向
        for i in range(1, 5):
            nx, ny = x - i*dx, y - i*dy
            if not is_valid_pos(nx, ny):
                break
            if board[nx, ny] == role:
                count += 1
            elif board[nx, ny] == EMPTY:
                open_ends += 1
                break
            else:
                break
        
        if count >= 5:
            total += SCORE_FIVE
        elif count == 4:
            if open_ends == 2:
                total += SCORE_LIVE_FOUR
            elif open_ends == 1:
                total += SCORE_DEAD_FOUR
        elif count == 3:
            if open_ends == 2:
                total += SCORE_LIVE_THREE
            elif open_ends == 1:
                total += SCORE_DEAD_THREE
        elif count == 2:
            if open_ends == 2:
                total += SCORE_LIVE_TWO
            elif open_ends == 1:
                total += SCORE_DEAD_TWO
    
    return total

# ============ 混合AI (NN + Minimax) ============
class HybridAgent:
    def __init__(self, model_path, max_depth=4, time_limit=15.0):
        self.model = GomokuNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = 0
        self.time_up = False
        
    def get_nn_scores(self, board, role):
        """获取NN评分"""
        # 标准化棋盘
        if role == WHITE:
            norm_board = np.where(board == BLACK, WHITE, np.where(board == WHITE, BLACK, board))
        else:
            norm_board = board.copy()
        
        with torch.no_grad():
            x = torch.FloatTensor(norm_board).unsqueeze(0).unsqueeze(0)
            policy, _ = self.model(x)
            probs = torch.softmax(policy, dim=1).squeeze().numpy()
        
        return probs.reshape(BOARD_SIZE, BOARD_SIZE)
    
    def evaluate_board(self, board, role):
        """全局评估"""
        opponent = WHITE if role == BLACK else BLACK
        my_score = 0
        opp_score = 0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == role:
                    my_score += evaluate_point(board, i, j, role)
                elif board[i, j] == opponent:
                    opp_score += evaluate_point(board, i, j, opponent)
        
        return my_score - opp_score
    
    def get_candidates(self, board, role, nn_probs):
        """生成候选着法"""
        opponent = WHITE if role == BLACK else BLACK
        candidates = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] != EMPTY:
                    continue
                
                # 邻居检查
                has_neighbor = False
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = i + dx, j + dy
                        if is_valid_pos(nx, ny) and board[nx, ny] != EMPTY:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                
                if not has_neighbor:
                    continue
                
                # 禁手检查
                if role == BLACK and is_forbidden(board, i, j):
                    continue
                
                # 评分
                my_score = evaluate_point(board, i, j, role)
                opp_score = evaluate_point(board, i, j, opponent)
                nn_score = nn_probs[i, j] * 1000
                
                combined = my_score + opp_score * 0.9 + nn_score
                candidates.append((i, j, combined))
        
        candidates.sort(key=lambda x: -x[2])
        return candidates[:15]
    
    def minimax(self, board, depth, alpha, beta, is_max, role, nn_probs):
        """Minimax搜索"""
        if time.time() - self.start_time > self.time_limit:
            self.time_up = True
            return 0, None
        
        opponent = WHITE if role == BLACK else BLACK
        current = role if is_max else opponent
        
        candidates = self.get_candidates(board, current, nn_probs)
        
        if not candidates:
            return 0, None
        
        if depth == 0:
            return self.evaluate_board(board, role), None
        
        best_move = candidates[0][:2]
        
        if is_max:
            max_eval = float('-inf')
            for x, y, _ in candidates:
                board[x, y] = role
                
                if check_win(board, x, y):
                    board[x, y] = EMPTY
                    return SCORE_FIVE + depth * 1000, (x, y)
                
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False, role, nn_probs)
                board[x, y] = EMPTY
                
                if self.time_up:
                    return 0, None
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (x, y)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for x, y, _ in candidates:
                board[x, y] = opponent
                
                if check_win(board, x, y):
                    board[x, y] = EMPTY
                    return -(SCORE_FIVE + depth * 1000), (x, y)
                
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True, role, nn_probs)
                board[x, y] = EMPTY
                
                if self.time_up:
                    return 0, None
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (x, y)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_move(self, board, role):
        """迭代加深搜索"""
        self.start_time = time.time()
        self.time_up = False
        
        nn_probs = self.get_nn_scores(board, role)
        
        best_move = None
        best_score = 0
        final_depth = 0
        
        for depth in range(2, self.max_depth + 1, 2):
            score, move = self.minimax(board.copy(), depth, float('-inf'), float('inf'), True, role, nn_probs)
            
            if self.time_up:
                break
            
            if move:
                best_move = move
                best_score = score
                final_depth = depth
            
            if score >= SCORE_FIVE:
                break
            
            if time.time() - self.start_time > self.time_limit * 0.8:
                break
        
        elapsed = time.time() - self.start_time
        return best_move, elapsed, final_depth

# ============ 纯Minimax ============
class MinimaxAgent:
    def __init__(self, max_depth=6, time_limit=15.0):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = 0
        self.time_up = False
        
    def evaluate_board(self, board, role):
        opponent = WHITE if role == BLACK else BLACK
        my_score = 0
        opp_score = 0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == role:
                    my_score += evaluate_point(board, i, j, role)
                elif board[i, j] == opponent:
                    opp_score += evaluate_point(board, i, j, opponent)
        
        return my_score - opp_score
    
    def get_candidates(self, board, role):
        opponent = WHITE if role == BLACK else BLACK
        candidates = []
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] != EMPTY:
                    continue
                
                has_neighbor = False
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = i + dx, j + dy
                        if is_valid_pos(nx, ny) and board[nx, ny] != EMPTY:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                
                if not has_neighbor:
                    continue
                
                if role == BLACK and is_forbidden(board, i, j):
                    continue
                
                my_score = evaluate_point(board, i, j, role)
                opp_score = evaluate_point(board, i, j, opponent)
                combined = my_score + opp_score * 0.9
                candidates.append((i, j, combined))
        
        candidates.sort(key=lambda x: -x[2])
        return candidates[:15]
    
    def minimax(self, board, depth, alpha, beta, is_max, role):
        if time.time() - self.start_time > self.time_limit:
            self.time_up = True
            return 0, None
        
        opponent = WHITE if role == BLACK else BLACK
        current = role if is_max else opponent
        
        candidates = self.get_candidates(board, current)
        
        if not candidates:
            return 0, None
        
        if depth == 0:
            return self.evaluate_board(board, role), None
        
        best_move = candidates[0][:2]
        
        if is_max:
            max_eval = float('-inf')
            for x, y, _ in candidates:
                board[x, y] = role
                
                if check_win(board, x, y):
                    board[x, y] = EMPTY
                    return SCORE_FIVE + depth * 1000, (x, y)
                
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False, role)
                board[x, y] = EMPTY
                
                if self.time_up:
                    return 0, None
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (x, y)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for x, y, _ in candidates:
                board[x, y] = opponent
                
                if check_win(board, x, y):
                    board[x, y] = EMPTY
                    return -(SCORE_FIVE + depth * 1000), (x, y)
                
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True, role)
                board[x, y] = EMPTY
                
                if self.time_up:
                    return 0, None
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (x, y)
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_move(self, board, role):
        self.start_time = time.time()
        self.time_up = False
        
        best_move = None
        best_score = 0
        final_depth = 0
        
        for depth in range(2, self.max_depth + 1, 2):
            score, move = self.minimax(board.copy(), depth, float('-inf'), float('inf'), True, role)
            
            if self.time_up:
                break
            
            if move:
                best_move = move
                best_score = score
                final_depth = depth
            
            if score >= SCORE_FIVE:
                break
            
            if time.time() - self.start_time > self.time_limit * 0.8:
                break
        
        elapsed = time.time() - self.start_time
        return best_move, elapsed, final_depth

# ============ 对战 ============
def battle(agent1, name1, agent2, name2, show_board=True):
    """两个AI对战"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    
    agents = [(agent1, name1, BLACK), (agent2, name2, WHITE)]
    move_count = 0
    last_move = None
    
    print(f"\n🎮 对战: {name1} (黑) vs {name2} (白)\n")
    
    while True:
        agent, name, role = agents[move_count % 2]
        
        if show_board:
            print_board(board, last_move)
            print(f"第 {move_count + 1} 步: {name} 思考中...")
        
        # 空盘下中心
        if move_count == 0:
            move = (7, 7)
            elapsed = 0
            depth = 0
        else:
            move, elapsed, depth = agent.get_move(board, role)
        
        if move is None:
            print(f"⚠️ {name} 无法落子")
            break
        
        x, y = move
        board[x, y] = role
        last_move = (x, y)
        
        if show_board:
            print_board(board, last_move)
            print(f"✅ {name} 落子: {chr(ord('A') + y)}{x + 1} (深度{depth}, {elapsed:.1f}s)")
        
        if check_win(board, x, y):
            print(f"\n🎉 {name} ({'黑' if role == BLACK else '白'}) 获胜! 共 {move_count + 1} 步")
            return name, move_count + 1
        
        move_count += 1
        
        if move_count >= 225:
            print("平局!")
            return "平局", move_count
        
        time.sleep(0.1)
    
    return None, move_count

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "gomoku_model.pth")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建AI - 使用较低深度加快测试
    # 混合AI: 深度4, 10秒限时
    hybrid = HybridAgent(model_path, max_depth=4, time_limit=10.0)
    
    # 纯Minimax: 深度4, 10秒限时
    minimax = MinimaxAgent(max_depth=4, time_limit=10.0)
    
    print("=" * 50)
    print("对战测试: 混合AI(深度4) vs Minimax(深度4)")
    print("时间限制: 10秒")
    print("=" * 50)
    
    # Hybrid 先手
    winner1, moves1 = battle(hybrid, "Hybrid AI(深度4)", minimax, "Minimax(深度4)")
    
    input("\n按回车继续下一局...")
    
    # Minimax 先手
    winner2, moves2 = battle(minimax, "Minimax(深度4)", hybrid, "Hybrid AI(深度4)")
    
    print("\n" + "=" * 50)
    print("对战结果汇总:")
    print(f"  第1局: {winner1} 胜 ({moves1}步)")
    print(f"  第2局: {winner2} 胜 ({moves2}步)")
    print("=" * 50)

if __name__ == "__main__":
    main()
