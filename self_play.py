# coding=utf-8
import os
import random
import torch
from go import Go
from net import PolicyNetwork,ValueNetwork
from genMove import getPolicyNetResult, toPosition
from sgfmill import sgf
from features import  *
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def self_play_game(black_model_path, white_model_path, sgf_path=None):
    board = Go()
    sgf_game = sgf.Sgf_game(size=19)
    node = sgf_game.get_root()  # 初始是根节点

    models = {1: black_model_path, -1: white_model_path}
    records = {1: [], -1: []}

    while not board.game_over():
        color = board.current_color
        model = models[color]
        policy_logits = getPolicyNetResult(board, model)
        sorted_indices = list(reversed(torch.argsort(policy_logits)))

        for index in sorted_indices:
            x, y = toPosition(index)

            if (x, y) == (None, None):  # pass
                board.passcount += 1
                node = node.new_child()
                node.set_move('b' if color == 1 else 'w', None)
                records[color].append((getAllFeatures(board), index.item()))
                break

            if board.move(x, y):
                node = node.new_child()
                node.set_move('b' if color == 1 else 'w', (x, y))
                records[color].append((getAllFeatures(board), index.item()))
                break

    winner = board.get_winner()
    sgf_game.get_root().set("RE", "B+" if winner == 1 else "W+")


    if sgf_path:
        os.makedirs(os.path.dirname(sgf_path), exist_ok=True)
        with open(sgf_path, "wb") as f:
            f.write(sgf_game.serialise())

    return records, winner



def train_policy_gradient(model, data, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    states, actions, rewards = zip(*data)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)

    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        for i in range(0, len(states), 64):
            batch_states = states[i:i+64]
            batch_actions = actions[i:i+64]
            batch_rewards = rewards[i:i+64]

            logits = model(batch_states)
            log_probs = torch.log_softmax(logits, dim=1)
            selected_log_probs = log_probs[range(len(batch_actions)), batch_actions]

            loss = -(selected_log_probs * batch_rewards).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



def update_best_model(model, best_model_path):
    torch.save(model.state_dict(), best_model_path)
    print(f"[更新] 最佳模型已保存到 {best_model_path}")

def train_value_network(value_model, data, epochs=5, lr=1e-3):
    value_model.to(device)
    optimizer = torch.optim.Adam(value_model.parameters(), lr=lr)
    value_model.train()

    states, rewards = zip(*data)
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        for i in range(0, len(states), 64):
            batch_states = states[i:i+64]
            batch_rewards = rewards[i:i+64]

            preds = value_model(batch_states).squeeze()
            loss = F.mse_loss(preds, batch_rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[ValueNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")


def selfplay_train_pipeline(
    best_model_path,
    model_pool_dir,
    sgf_dir,
    selfplay_games=100,
    train_epochs=5
):
    os.makedirs(model_pool_dir, exist_ok=True)

    value_model = ValueNetwork()

    best_model = PolicyNetwork()
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    value_model = ValueNetwork()
    for cycle in range(selfplay_games // 2):
        print(f"\n=== 自对弈 第 {cycle + 1} 轮 ===")
        game_data_black, game_data_white = [], []

        for game_idx in range(2):
            opponent_model = PolicyNetwork()
            opponent_path = random.choice([
                os.path.join(model_pool_dir, f)
                for f in os.listdir(model_pool_dir)
                if f.endswith('.pt')
            ])
            opponent_model.load_state_dict(torch.load(opponent_path, map_location=device))
            black_path, white_path = (best_model_path, opponent_path) if game_idx % 2 == 0 else (best_model_path, best_model_path)

            sgf_path = os.path.join(sgf_dir, f"game_{cycle:03d}_{game_idx}.sgf")
            records, winner = self_play_game(black_path, white_path, sgf_path)

            # 奖励分配：胜者 +1，负者 -1
            for color in [1, -1]:
                reward = 1.0 if color == winner else -1.0
                data = [(s, a, reward) for s, a in records[color]]
                if color == 1:
                    game_data_black.extend(data)
                else:
                    game_data_white.extend(data)

        value_data = [(s, r) for (s, a, r) in game_data_black + game_data_white]

        # 训练两个模型
        print("[训练] Black 模型")
        train_policy_gradient(best_model, game_data_black, epochs=train_epochs)
        print("[训练] White 模型")
        train_policy_gradient(best_model, game_data_white, epochs=train_epochs)

        #训练Value网络
        train_value_network(value_model, value_data, epochs=train_epochs)

        # 保存新的最优模型副本
        new_model_path = os.path.join(model_pool_dir, f"model_cycle_{cycle:03d}.pt")
        torch.save(best_model.state_dict(), new_model_path)

        # 保存 Value 网络模型
        value_model_path = os.path.join(model_pool_dir, f"value_model_cycle_{cycle:03d}.pt")
        torch.save(value_model.state_dict(), value_model_path)
        # 更新当前最优
        update_best_model(best_model, best_model_path)


if __name__ == '__main__':
    selfplay_train_pipeline(
        best_model_path='checkpoints/best_model.pt',
        model_pool_dir='checkpoints',
        sgf_dir='games/sgf_selfplay',
        selfplay_games=100,
        train_epochs=5
    )