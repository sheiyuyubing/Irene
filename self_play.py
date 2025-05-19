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


def train_policy_gradient(model, value_model, data, epochs=5, lr=5e-4):
    model.train()
    value_model.eval()

    states, actions, rewards = zip(*data)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    baseline = value_model(states).squeeze().detach()
    advantages = rewards - baseline

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(states), 64):
            batch_states = states[i:i+64]
            batch_actions = actions[i:i+64]
            batch_adv = advantages[i:i+64]

            logits = model(batch_states)
            log_probs = torch.log_softmax(logits, dim=1)
            selected_log_probs = log_probs[range(len(batch_actions)), batch_actions]
            loss = -(selected_log_probs * batch_adv).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[PolicyNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")


def train_value_network(model, data, epochs=5, lr=5e-4):
    model.train()
    states, rewards = zip(*data)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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


def evaluate_win_rate(model_a, model_b, n_games=10):
    wins = 0
    for i in range(n_games):
        black, white = (model_a, model_b) if i % 2 == 0 else (model_b, model_a)
        _, winner = self_play_game(black, white)
        if (i % 2 == 0 and winner == 1) or (i % 2 == 1 and winner == -1):
            wins += 1
    return wins / n_games


def selfplay_train_pipeline(best_model_path, model_pool_dir, sgf_dir, total_cycles=10, games_per_cycle=10, train_epochs=5):
    os.makedirs(model_pool_dir, exist_ok=True)
    os.makedirs(sgf_dir, exist_ok=True)

    best_model = load_model(best_model_path, PolicyNetwork)
    value_model = ValueNetwork().to(device)

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
        train_value_network(value_model, [(s, r) for s, a, r in records_all], epochs=train_epochs)
        train_policy_gradient(best_model, value_model, records_all, epochs=train_epochs)

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


if __name__ == '__main__':
    selfplay_train_pipeline(
        best_model_path='checkpoints/best_model.pt',
        model_pool_dir='checkpoints',
        sgf_dir='games/sgf_selfplay',
        total_cycles=20,
        games_per_cycle=10,
        train_epochs=5
    )