# coding=utf-8
import os
import random
import torch
import torch.nn.functional as F
from go import Go
from net import PolicyNetwork, ValueNetwork
from genMove import getPolicyNetResult, toPosition
from features import getAllFeatures
from sgfmill import sgf
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def self_play_game(black_model_dir, white_model_dir, sgf_path=None):
    board = Go()
    sgf_game = sgf.Sgf_game(size=19)
    node = sgf_game.get_root()

    models = {1: black_model_dir, -1: white_model_dir}
    records = {1: [], -1: []}

    while not board.game_over():
        color = board.current_color
        model = models[color]

        logits = getPolicyNetResult(board, model)
        sorted_indices = list(reversed(torch.argsort(logits)))

        for idx in sorted_indices:
            x, y = toPosition(idx.item())

            if (x, y) == (None, None):
                # board.move(x,y)
                board.passcount += 1
                node = node.new_child()
                node.set_move('b' if color == 1 else 'w', None)
                records[color].append((getAllFeatures(board), idx.item()))
                break

            if board.move(x, y):
                node = node.new_child()
                node.set_move('b' if color == 1 else 'w', (x, y))
                records[color].append((getAllFeatures(board), idx.item()))
                break

    winner = board.get_winner()
    sgf_game.get_root().set("RE", "B+" if winner == 1 else "W+")

    if sgf_path:
        os.makedirs(os.path.dirname(sgf_path), exist_ok=True)
        with open(sgf_path, "wb") as f:
            f.write(sgf_game.serialise())

    return records, winner


def train_policy_gradient(policy_model, value_model, data, old_policy_model=None,
                          epochs=5, lr=5e-4, entropy_coef=1e-3):
    policy_model.train()
    value_model.eval()

    # 准备训练数据
    states, actions, rewards = zip(*data)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    # 基线（来自值网络）
    with torch.no_grad():
        baseline = value_model(states).squeeze()
    advantages = rewards - baseline

    # 标准化 advantage
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(states), 64):
            batch_states = states[i:i+64]
            batch_actions = actions[i:i+64]
            batch_advantages = advantages[i:i+64]

            logits = policy_model(batch_states)
            log_probs = F.log_softmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)
            selected_log_probs = log_probs[range(len(batch_actions)), batch_actions]

            # 熵正则项（鼓励探索）
            entropy = -torch.sum(probs * log_probs, dim=1).mean()

            # 若提供旧策略，计算重要性采样比率（PPO 的思想）
            if old_policy_model is not None:
                with torch.no_grad():
                    old_logits = old_policy_model(batch_states)
                    old_log_probs = F.log_softmax(old_logits, dim=1)
                    old_selected_log_probs = old_log_probs[range(len(batch_actions)), batch_actions]
                ratios = torch.exp(selected_log_probs - old_selected_log_probs)
                policy_loss = -torch.mean(ratios * batch_advantages)
            else:
                policy_loss = -torch.mean(selected_log_probs * batch_advantages)

            # 总损失 = 策略损失 - 熵奖励
            loss = policy_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(states) // 64 + 1)
        print(f"[PolicyNet] Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return loss_history


def train_value_network(model, data, epochs=5, lr=1e-4):
    model.train()
    states, rewards = zip(*data)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(epochs):
        for i in range(0, len(states), 64):
            batch_states = states[i:i+64]
            batch_rewards = rewards[i:i+64]

            preds = model(batch_states).squeeze()
            loss = F.mse_loss(preds, batch_rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[ValueNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return loss_history

def evaluate_win_rate(model_a, model_b, n_games=10):
    wins = 0
    for i in range(n_games):
        black, white = (model_a, model_b) if i % 2 == 0 else (model_b, model_a)
        _, winner = self_play_game(black, white)
        if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
            wins += 1
    return wins / n_games



def plot_training_losses(policy_losses, value_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(policy_losses, label='Policy Network Loss', color='blue')
    plt.plot(value_losses, label='Value Network Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.show()

def selfplay_train_pipeline(best_model_path, model_pool_dir, sgf_dir, total_cycles=10, games_per_cycle=10, train_epochs=5):
    os.makedirs(model_pool_dir, exist_ok=True)
    os.makedirs(sgf_dir, exist_ok=True)

    best_model = load_model(best_model_path, PolicyNetwork)
    value_model = ValueNetwork().to(device)

    policy_loss_all = []
    value_loss_all = []
    for cycle in range(total_cycles):
        print(f"\n=== Cycle {cycle + 1} ===")
        records_all = []

        for game_idx in range(games_per_cycle):
            opponent_path = random.choice([
                os.path.join(model_pool_dir, f) for f in os.listdir(model_pool_dir) if f.endswith(".pt")
            ]) if cycle > 0 else best_model_path
            opponent = load_model(opponent_path, PolicyNetwork)

            black, white = (best_model, opponent) if game_idx % 2 == 0 else (opponent, best_model)
            sgf_path = os.path.join(sgf_dir, f"game_{cycle:03d}_{game_idx}.sgf")
            records, winner = self_play_game(black, white, sgf_path)

            for color in [1, -1]:
                reward = 1.0 if color == winner else -1.0
                records_all.extend([(s, a, reward) for s, a in records[color]])

        # Train networks
        value_losses=train_value_network(value_model, [(s, r) for s, a, r in records_all], epochs=train_epochs)
        policy_losses=train_policy_gradient(best_model, value_model, records_all, epochs=train_epochs)

        value_loss_all.extend(value_losses)
        policy_loss_all.extend(policy_losses)
        # Save candidate model
        candidate_path = os.path.join(model_pool_dir, f"candidate_cycle_{cycle:03d}.pt")
        torch.save(best_model.state_dict(), candidate_path)

        # Evaluate vs previous best
        old_model = load_model(best_model_path, PolicyNetwork)
        win_rate = evaluate_win_rate(best_model, old_model, n_games=10)
        print(f"[评估] 当前胜率 vs 最优模型: {win_rate*100:.1f}%")

        if win_rate > 0.55:
            torch.save(best_model.state_dict(), best_model_path)
            print("[更新] 替换最优模型 ")
        else:
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("[回退] 当前模型未达标，保持原模型 ")

        plot_training_losses(policy_loss_all, value_loss_all)

def selfplay_train_pipeline_v2(
    best_model_path,
    model_pool_dir,
    sgf_dir,
    total_cycles=20,
    games_per_cycle=10,
    train_epochs=10,
    policy_lr=1e-5,
    value_lr=1e-5,
    update_threshold=0.52,
    eval_games=20
):
    os.makedirs(model_pool_dir, exist_ok=True)
    os.makedirs(sgf_dir, exist_ok=True)

    best_model = load_model(best_model_path, PolicyNetwork)
    value_model = ValueNetwork().to(device)
    policy_loss_all = []
    value_loss_all = []
    for cycle in range(total_cycles):
        print(f"\n=== Cycle {cycle + 1}/{total_cycles} ===")
        records_all = []

        for game_idx in range(games_per_cycle):
            opponent_path = random.choice([
                os.path.join(model_pool_dir, f)
                for f in os.listdir(model_pool_dir) if f.endswith(".pt")
            ]) if cycle > 0 else best_model_path
            opponent = load_model(opponent_path, PolicyNetwork)

            black, white = (best_model, opponent) if game_idx % 2 == 0 else (opponent, best_model)
            sgf_path = os.path.join(sgf_dir, f"game_{cycle:03d}_{game_idx}.sgf")
            records, winner = self_play_game(black, white, sgf_path)

            for color in [1, -1]:
                reward = 1.0 if color == winner else -1.0
                records_all.extend([(s, a, reward) for s, a in records[color]])

        print(f"[训练数据] 总样本数: {len(records_all)}")

        # 训练 Value Network
        value_losses =train_value_network(value_model,
                            [(s, r) for s, a, r in records_all],
                            epochs=train_epochs,
                            lr=value_lr)

        # 训练 Policy Network（策略梯度）
        policy_losses = train_policy_gradient(best_model,
                              value_model,
                              records_all,
                              epochs=train_epochs,
                              lr=policy_lr)

        value_loss_all.extend(value_losses)
        policy_loss_all.extend(policy_losses)
        # 保存候选模型
        candidate_path = os.path.join(model_pool_dir, f"candidate_cycle_{cycle:03d}.pt")
        torch.save(best_model.state_dict(), candidate_path)

        # 评估当前模型相较最优模型的胜率
        old_model = load_model(best_model_path, PolicyNetwork)
        win_rate = evaluate_win_rate(best_model, old_model, n_games=eval_games)
        print(f"[评估] 当前模型 VS 最优模型胜率：{win_rate*100:.2f}%")

        if win_rate > update_threshold:
            torch.save(best_model.state_dict(), best_model_path)
            print(f"[更新]  胜率高于 {update_threshold*100:.1f}%，替换最优模型")
        else:
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"[回退]  胜率未达 {update_threshold*100:.1f}%，保留原模型")

        plot_training_losses(policy_loss_all, value_loss_all)
        
if __name__ == '__main__':
    # selfplay_train_pipeline(
    #     best_model_path='checkpoints/best_model.pt',
    #     model_pool_dir='checkpoints',
    #     sgf_dir='games/sgf_selfplay',
    #     total_cycles=20,
    #     games_per_cycle=10,
    #     train_epochs=5
    # )

    selfplay_train_pipeline_v2(
        best_model_path='checkpoints/best_model.pt',
        model_pool_dir='checkpoints',
        sgf_dir='games/sgf_selfplay',
        total_cycles=20,
        games_per_cycle=10,
        train_epochs=10,
        policy_lr=1e-3,
        value_lr=1e-3,
        update_threshold=0.52,
        eval_games=20
    )
