import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
import math
import copy
import time

# 检查 GPU
# RTX 5070 (sm_120) 暂不支持，强制使用 CPU
device = torch.device("cpu")
print(f"Using device: {device} (RTX 5070 sm_120 not yet supported by PyTorch)")

# ==========================================
# 1. 定义模型结构 (Dual Head: Policy + Value)
# ==========================================
class GomokuNet(nn.Module):
    def __init__(self):
        super(GomokuNet, self).__init__()
        # Shared Backbone
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 15 * 15, 256)
        
        # Value Head (用于 C 引擎评估)
        self.fc2 = nn.Linear(256, 1)
        
        # Policy Head (用于 Python MCTS 训练)
        self.fc_policy = nn.Linear(256, 225)

    def forward(self, x):
        # x: [batch, 225] -> [batch, 1, 15, 15]
        x = x.view(-1, 1, 15, 15)
        
        # Backbone
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 15 * 15)
        x = self.relu(self.fc1(x))
        
        # Value Output [-1, 1]
        val = torch.tanh(self.fc2(x))
        
        # Policy Output (Logits)
        pol = self.fc_policy(x)
        
        return pol, val

# ==========================================
# 2. MCTS 实现 (AlphaZero Style)
# ==========================================
class MCTSNode:
    def __init__(self, parent=None, prior_prob=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob
        
    @property
    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, c_puct=5, n_playout=100):
        self.model = model
        self.c_puct = c_puct
        self.n_playout = n_playout

    def get_move_probs(self, game, temp=1.0):
        root = MCTSNode(prior_prob=1.0)
        self._expand(root, game) # Expand root

        for _ in range(self.n_playout):
            node = root
            scratch_game = copy.deepcopy(game)
            
            # 1. Selection
            while not node.is_leaf():
                action, node = self._select_child(node)
                scratch_game.make_move(action // 15, action % 15)

            # 2. Evaluation & Expansion
            winner = scratch_game.check_winner()
            if winner != 0:
                # Game over
                # If winner is current player (who just moved to get here), value is 1?
                # No, scratch_game.current_player is the NEXT player.
                # If winner == scratch_game.current_player, then it's a win for next player (impossible if game just ended).
                # If winner != 0, the PREVIOUS player won.
                # So for the current player (node), it is a LOSS (-1).
                leaf_value = -1.0
            else:
                leaf_value = self._expand(node, scratch_game)

            # 3. Backpropagation
            self._backpropagate(node, -leaf_value)

        acts = list(root.children.keys())
        visits = [node.visit_count for node in root.children.values()]
        
        if temp == 0:
            best_act = acts[np.argmax(visits)]
            probs = [0.0] * 225
            probs[best_act] = 1.0
            return probs
            
        visits = np.array(visits)
        probs_visits = visits ** (1.0 / temp)
        probs_visits = probs_visits / np.sum(probs_visits)
        
        full_probs = [0.0] * 225
        for act, p in zip(acts, probs_visits):
            full_probs[act] = p
        return full_probs

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand(self, node, game):
        # Prepare input
        board_tensor = torch.FloatTensor(game.get_canonical_board()).to(device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
            
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        value = value.item()
        
        legal_moves = game.get_legal_moves()
        
        # Mask & Renormalize
        valid_probs = []
        for move in legal_moves:
            valid_probs.append(policy_probs[move])
            
        sum_probs = sum(valid_probs)
        if sum_probs > 0:
            valid_probs = [p / sum_probs for p in valid_probs]
        else:
            valid_probs = [1.0 / len(legal_moves)] * len(legal_moves)

        for move, prob in zip(legal_moves, valid_probs):
            node.children[move] = MCTSNode(parent=node, prior_prob=prob)
            
        return value

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            value = -value

# ==========================================
# 3. 游戏逻辑
# ==========================================
class GomokuGame:
    def __init__(self):
        self.size = 15
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1 # 1: Black, -1: White
        self.moves_count = 0

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.moves_count = 0

    def get_canonical_board(self):
        # Return board from perspective of current player
        # Self = 1, Opponent = -1
        return self.board.flatten() * self.current_player

    def get_legal_moves(self):
        return [i * 15 + j for i in range(15) for j in range(15) if self.board[i][j] == 0]

    def make_move(self, r, c):
        if self.board[r][c] != 0: return False
        self.board[r][c] = self.current_player
        self.current_player *= -1
        self.moves_count += 1
        return True

    def check_winner(self):
        # Check lines for the PREVIOUS player (who just moved)
        # Previous player is -self.current_player
        player = -self.current_player
        
        # Simple check (can be optimized)
        for r in range(15):
            for c in range(15):
                if self.board[r][c] == player:
                    if self.check_line(r, c, 1, 0, player) or \
                       self.check_line(r, c, 0, 1, player) or \
                       self.check_line(r, c, 1, 1, player) or \
                       self.check_line(r, c, 1, -1, player):
                        return player
        if self.moves_count == 225: return 2 # Draw
        return 0

    def check_line(self, r, c, dr, dc, player):
        count = 0
        for i in range(5):
            nr, nc = r + dr*i, c + dc*i
            if 0 <= nr < 15 and 0 <= nc < 15 and self.board[nr][nc] == player:
                count += 1
            else:
                break
        return count == 5

# ==========================================
# 4. 训练流程
# ==========================================
def get_symmetries(board, pi):
    # board: 225, pi: 225
    board_2d = board.reshape(15, 15)
    pi_2d = pi.reshape(15, 15)
    syms = []
    
    for i in range(4):
        b_rot = np.rot90(board_2d, i)
        p_rot = np.rot90(pi_2d, i)
        syms.append((b_rot.flatten(), p_rot.flatten()))
        
        b_flip = np.fliplr(b_rot)
        p_flip = np.fliplr(p_rot)
        syms.append((b_flip.flatten(), p_flip.flatten()))
    return syms

def self_play(model, mcts_sims=100):
    game = GomokuGame()
    mcts = MCTS(model, n_playout=mcts_sims)
    
    states = []
    probs = []
    current_players = []
    
    while True:
        # Get MCTS probs
        # Temp: 1.0 for first 10 moves, then 0.1
        temp = 1.0 if game.moves_count < 10 else 0.1
        pi = mcts.get_move_probs(game, temp=temp)
        
        # Store data
        states.append(game.get_canonical_board())
        probs.append(pi)
        current_players.append(game.current_player)
        
        # Pick move
        action = np.random.choice(len(pi), p=pi)
        game.make_move(action // 15, action % 15)
        
        winner = game.check_winner()
        if winner != 0:
            return states, probs, current_players, winner

def train():
    model = GomokuNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # verbose parameter is deprecated/removed in newer PyTorch versions or defaults differently
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Load if exists
    if os.path.exists("ai_training/gomoku_model.pth"):
        try:
            model.load_state_dict(torch.load("ai_training/gomoku_model.pth", map_location=device), strict=False)
            print("Loaded existing model.")
        except:
            print("Model mismatch, starting fresh.")

    buffer_s = []
    buffer_p = []
    buffer_v = []
    
    # 8 Hours target: Let's just run a huge loop
    # Assuming 1 game takes 30s (CPU) or 5s (GPU)
    # 8 hours = 28800 seconds.
    # Let's aim for 1000 games.
    total_games = 1000
    batch_size = 128 # Increased batch size for stability
    epochs = 10 # Train more on each batch
    
    print(f"优化训练 (Optimized Training). 目标: {total_games} 局")
    print("升级点: 动态学习率, 更大的 Batch Size, 更深的 MCTS 搜索")
    
    for i in range(total_games):
        model.eval()
        # Increase simulation count for better quality games
        # Early games can be fast, later games need depth
        sims = 200 if i < 500 else 400
        states, pis, players, winner = self_play(model, mcts_sims=sims)
        
        # Process game result
        # Winner: 1 (Black), -1 (White), 2 (Draw)
        # If winner == 1: Black won.
        # For a state where current_player was 1: Value = 1
        # For a state where current_player was -1: Value = -1
        
        game_len = len(states)
        print(f"Game {i+1}/{total_games} finished. Length: {game_len}. Winner: {winner}")
        
        for j in range(game_len):
            s = states[j]
            p = np.array(pis[j])
            cp = players[j]
            
            if winner == 2:
                v = 0.0
            elif winner == cp:
                v = 1.0
            else:
                v = -1.0
                
            # Data Augmentation
            syms = get_symmetries(s, p)
            for s_sym, p_sym in syms:
                buffer_s.append(s_sym)
                buffer_p.append(p_sym)
                buffer_v.append(v)
                
        # Keep buffer size reasonable
        if len(buffer_s) > 20000:
            buffer_s = buffer_s[-20000:]
            buffer_p = buffer_p[-20000:]
            buffer_v = buffer_v[-20000:]
            
        # Train
        if len(buffer_s) >= batch_size:
            model.train()
            total_loss = 0
            for _ in range(epochs):
                # Sample batch
                indices = np.random.choice(len(buffer_s), batch_size, replace=False)
                b_s = torch.FloatTensor(np.array([buffer_s[i] for i in indices])).to(device)
                b_p = torch.FloatTensor(np.array([buffer_p[i] for i in indices])).to(device)
                b_v = torch.FloatTensor(np.array([buffer_v[i] for i in indices])).to(device).unsqueeze(1)
                
                out_p, out_v = model(b_s)
                
                # Loss: Value MSE + Policy CrossEntropy
                loss_v = nn.MSELoss()(out_v, b_v)
                
                # Policy Loss: -sum(target * log(pred))
                # out_p is logits. Use log_softmax
                log_probs = nn.LogSoftmax(dim=1)(out_p)
                loss_p = -(b_p * log_probs).sum(dim=1).mean()
                
                loss = loss_v + loss_p
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/epochs
            print(f"  -> Train Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
            
        # Save frequently
        if (i+1) % 10 == 0:
            torch.save(model.state_dict(), "ai_training/gomoku_model.pth")
            print("Model saved.")

if __name__ == "__main__":
    train()
