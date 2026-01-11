"""
训练脚本：使用限时混合AI生成数据训练神经网络
每步限时15秒，使用迭代加深Minimax
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import copy

BOARD_SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2
TIME_LIMIT = 15.0  # 每步限时15秒

device = torch.device("cpu")
print(f"Using device: {device}")

# ============ 模型定义 ============
class GomokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 1)
        self.fc_policy = nn.Linear(256, 225)

    def forward(self, x):
        x = x.view(-1, 1, 15, 15)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 15 * 15)
        x = torch.relu(self.fc1(x))
        return self.fc_policy(x), torch.tanh(self.fc2(x))

# ============ 棋型分数 ============
SCORE_FIVE = 100000
SCORE_OPEN_FOUR = 10000
SCORE_BROKEN_FOUR = 5000
SCORE_OPEN_THREE = 1000
SCORE_BROKEN_THREE = 100
SCORE_OPEN_TWO = 50

def get_rel_piece(piece, player):
    if piece == EMPTY: return 0
    return 1 if piece == player else -1

def match_pattern_score(window):
    count = sum(1 for x in window if x == 1)
    opponent = sum(1 for x in window if x == -1)
    if opponent > 0: return 0
    if count == 5: return SCORE_FIVE
    if count == 4:
        return SCORE_OPEN_FOUR if window[0] == 0 and window[-1] == 0 else SCORE_BROKEN_FOUR
    if count == 3:
        return SCORE_OPEN_THREE if window[0] == 0 and window[-1] == 0 else SCORE_BROKEN_THREE
    if count == 2 and window[0] == 0 and window[-1] == 0:
        return SCORE_OPEN_TWO
    return 0

def evaluate_line(line, player):
    rel_line = [get_rel_piece(p, player) for p in line]
    score = 0
    for i in range(len(rel_line) - 5):
        score += match_pattern_score(rel_line[i:i+6])
    for i in range(len(rel_line) - 4):
        if sum(1 for x in rel_line[i:i+5] if x == 1) == 5:
            score += SCORE_FIVE
    return score

def heuristic_score(board, row, col, piece):
    score = 0
    opponent = WHITE if piece == BLACK else BLACK
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    original = board[row, col]
    board[row, col] = piece
    for dr, dc in directions:
        line = []
        r, c = row, col
        while 0 <= r - dr < BOARD_SIZE and 0 <= c - dc < BOARD_SIZE:
            r -= dr; c -= dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(board[r, c])
            r += dr; c += dc
        score += evaluate_line(line, piece)
    board[row, col] = original
    
    board[row, col] = opponent
    for dr, dc in directions:
        line = []
        r, c = row, col
        while 0 <= r - dr < BOARD_SIZE and 0 <= c - dc < BOARD_SIZE:
            r -= dr; c -= dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(board[r, c])
            r += dr; c += dc
        threat = evaluate_line(line, opponent)
        if threat >= SCORE_FIVE: score += SCORE_FIVE * 10
        elif threat >= SCORE_OPEN_FOUR: score += SCORE_OPEN_FOUR * 5
        elif threat >= SCORE_BROKEN_FOUR: score += SCORE_BROKEN_FOUR * 5
        elif threat >= SCORE_OPEN_THREE: score += SCORE_OPEN_THREE * 5
    board[row, col] = original
    
    center = BOARD_SIZE // 2
    score += (30 - abs(row - center) - abs(col - center))
    return score

def get_candidates(board):
    moves = []
    has_neighbor = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    stones = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                stones += 1
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == EMPTY:
                            has_neighbor[nr, nc] = True
    if stones == 0:
        return [(BOARD_SIZE // 2, BOARD_SIZE // 2, 0)]
    return [(r, c, 0) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if has_neighbor[r, c]]

def evaluate_full_board(board, ai_color):
    score = 0
    opp = WHITE if ai_color == BLACK else BLACK
    for r in range(BOARD_SIZE):
        line = list(board[r, :])
        score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
    for c in range(BOARD_SIZE):
        line = list(board[:, c])
        score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
    for k in range(-BOARD_SIZE + 1, BOARD_SIZE):
        line = list(np.diag(board, k))
        if len(line) >= 5:
            score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
        line = list(np.diag(np.fliplr(board), k))
        if len(line) >= 5:
            score += evaluate_line(line, ai_color) - int(evaluate_line(line, opp) * 1.5)
    return score

# ============ 限时迭代加深 Minimax ============
class TimeLimitExceeded(Exception):
    pass

def minimax_timed(board, depth, alpha, beta, is_max, ai_color, deadline):
    if time.time() > deadline:
        raise TimeLimitExceeded()
    
    if depth == 0:
        return evaluate_full_board(board, ai_color)
    
    moves = get_candidates(board)
    curr = ai_color if is_max else (WHITE if ai_color == BLACK else BLACK)
    scored = [(m[0], m[1], heuristic_score(board, m[0], m[1], curr)) for m in moves]
    scored.sort(key=lambda x: -x[2])
    scored = scored[:10 if depth >= 2 else 6]  # 更激进的剪枝
    
    if is_max:
        max_eval = float('-inf')
        for r, c, _ in scored:
            board[r, c] = curr
            val = minimax_timed(board, depth - 1, alpha, beta, False, ai_color, deadline)
            board[r, c] = EMPTY
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for r, c, _ in scored:
            board[r, c] = curr
            val = minimax_timed(board, depth - 1, alpha, beta, True, ai_color, deadline)
            board[r, c] = EMPTY
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha: break
        return min_eval

class HybridAgentTimed:
    """限时混合AI: NN筛选 + 迭代加深Minimax"""
    def __init__(self, model, time_limit=TIME_LIMIT):
        self.model = model
        self.time_limit = time_limit
    
    def nn_score(self, board, row, col, player):
        original = board[row, col]
        board[row, col] = player
        rel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rel[r, c] = get_rel_piece(board[r, c], player)
        with torch.no_grad():
            _, val = self.model(torch.from_numpy(rel.flatten()).unsqueeze(0).to(device))
        board[row, col] = original
        return val.item()
    
    def get_move_with_policy(self, board, player):
        """返回 (move, policy_distribution)"""
        board = board.copy()
        start_time = time.time()
        deadline = start_time + self.time_limit
        
        moves = get_candidates(board)
        
        # 用 NN + 启发式给候选点打分
        scored = []
        for r, c, _ in moves:
            h = heuristic_score(board, r, c, player)
            n = self.nn_score(board, r, c, player) * 1000
            scored.append((r, c, h + int(n)))
        scored.sort(key=lambda x: -x[2])
        candidates = scored[:12]  # 取 top 12
        
        # 迭代加深搜索
        best_move = candidates[0][:2]
        move_scores = {(r, c): 0 for r, c, _ in candidates}
        
        for depth in range(1, 6):  # 最大深度 5
            if time.time() > deadline - 0.5:  # 留 0.5s 余量
                break
            
            try:
                for r, c, _ in candidates:
                    board[r, c] = player
                    val = minimax_timed(board, depth, float('-inf'), float('inf'), False, player, deadline)
                    board[r, c] = EMPTY
                    move_scores[(r, c)] = val
                
                # 更新最佳走法
                best_move = max(move_scores.keys(), key=lambda m: move_scores[m])
            except TimeLimitExceeded:
                break
        
        # 生成 policy 分布 (基于搜索得分的 softmax)
        policy = np.zeros(225)
        scores = np.array([move_scores.get((r, c), -1e9) for r, c, _ in candidates])
        
        # 归一化到合理范围再 softmax
        scores = scores - scores.max()
        scores = np.clip(scores / 1000, -10, 0)  # 缩放
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        
        for i, (r, c, _) in enumerate(candidates):
            policy[r * BOARD_SIZE + c] = probs[i]
        
        return best_move, policy

# ============ 游戏逻辑 ============
def check_winner(board):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == EMPTY: continue
            p = board[r, c]
            for dr, dc in directions:
                count = 1
                for i in range(1, 5):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == p:
                        count += 1
                    else: break
                if count >= 5: return p
    return None

def get_symmetries(board, pi):
    board_2d = board.reshape(15, 15)
    pi_2d = pi.reshape(15, 15)
    syms = []
    for i in range(4):
        b_rot = np.rot90(board_2d, i)
        p_rot = np.rot90(pi_2d, i)
        syms.append((b_rot.flatten(), p_rot.flatten()))
        syms.append((np.fliplr(b_rot).flatten(), np.fliplr(p_rot).flatten()))
    return syms

# ============ 自对弈 ============
def self_play(agent):
    """混合AI自对弈，返回训练数据"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    current = BLACK
    moves_count = 0
    
    states = []
    policies = []
    players = []
    
    while moves_count < 225:
        # 获取走法和策略
        move, policy = agent.get_move_with_policy(board, current)
        
        # 存储数据 (规范化棋盘: 当前玩家视角)
        canonical = np.zeros(225, dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                canonical[r * 15 + c] = get_rel_piece(board[r, c], current)
        
        states.append(canonical)
        policies.append(policy)
        players.append(current)
        
        # 落子
        r, c = move
        board[r, c] = current
        moves_count += 1
        
        # 检查胜负
        winner = check_winner(board)
        if winner:
            return states, policies, players, winner, moves_count
        
        current = WHITE if current == BLACK else BLACK
    
    return states, policies, players, 0, moves_count  # 平局

# ============ 训练 ============
def train():
    model = GomokuNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 加载已有模型
    model_path = os.path.join(os.path.dirname(__file__), "gomoku_model.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Loaded model from {model_path}")
        except:
            print("Model mismatch, starting fresh.")
    
    agent = HybridAgentTimed(model, time_limit=TIME_LIMIT)
    
    buffer_s = []
    buffer_p = []
    buffer_v = []
    
    total_games = 500
    batch_size = 64
    epochs = 5
    
    print(f"\n限时训练 (每步{TIME_LIMIT}s)")
    print(f"目标: {total_games} 局")
    print("=" * 50)
    
    for i in range(total_games):
        model.eval()
        
        game_start = time.time()
        states, policies, players, winner, length = self_play(agent)
        game_time = time.time() - game_start
        
        winner_str = {BLACK: "黑胜", WHITE: "白胜", 0: "平局"}[winner]
        print(f"Game {i+1}/{total_games}: {length}步, {winner_str}, 用时{game_time/60:.1f}分钟")
        
        # 处理数据
        for j in range(len(states)):
            s = states[j]
            p = policies[j]
            cp = players[j]
            
            if winner == 0:
                v = 0.0
            elif winner == cp:
                v = 1.0
            else:
                v = -1.0
            
            # 数据增强
            for s_sym, p_sym in get_symmetries(s, p):
                buffer_s.append(s_sym)
                buffer_p.append(p_sym)
                buffer_v.append(v)
        
        # 限制 buffer 大小
        if len(buffer_s) > 10000:
            buffer_s = buffer_s[-10000:]
            buffer_p = buffer_p[-10000:]
            buffer_v = buffer_v[-10000:]
        
        # 训练
        if len(buffer_s) >= batch_size:
            model.train()
            total_loss = 0
            for _ in range(epochs):
                indices = np.random.choice(len(buffer_s), batch_size, replace=False)
                b_s = torch.FloatTensor(np.array([buffer_s[idx] for idx in indices])).to(device)
                b_p = torch.FloatTensor(np.array([buffer_p[idx] for idx in indices])).to(device)
                b_v = torch.FloatTensor(np.array([buffer_v[idx] for idx in indices])).to(device).unsqueeze(1)
                
                out_p, out_v = model(b_s)
                
                loss_v = nn.MSELoss()(out_v, b_v)
                log_probs = nn.LogSoftmax(dim=1)(out_p)
                loss_p = -(b_p * log_probs).sum(dim=1).mean()
                
                loss = loss_v + loss_p
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / epochs
            print(f"  -> Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
            
            # 更新 agent 的模型
            agent.model = model
        
        # 保存
        if (i + 1) % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print("  [Model saved]")

if __name__ == "__main__":
    train()
