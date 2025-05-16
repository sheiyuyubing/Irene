# coding=utf-8
import os
import random
import torch
from go import Go
from PrepareData_2 import preparePolicyData
from train_2 import trainPolicy
from net import PolicyNetwork
from genMove_2 import genMovePolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_self_play(best_model_path, model_pool_dir, sgf_output_dir, games_per_cycle=100):
    os.makedirs(sgf_output_dir, exist_ok=True)

    best_model = PolicyNetwork().to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()

    model_files = [f for f in os.listdir(model_pool_dir) if f.endswith('.pt')]
    assert len(model_files) > 0, "模型池为空"

    for game_idx in range(games_per_cycle):
        opponent_file = random.choice(model_files)
        opponent_model = PolicyNetwork().to(device)
        opponent_model.load_state_dict(torch.load(os.path.join(model_pool_dir, opponent_file), map_location=device))
        opponent_model.eval()

        sgf_content = self_play_game(best_model, opponent_model)

        save_path = os.path.join(sgf_output_dir, f"game_{game_idx:05d}.sgf")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sgf_content)


def self_play_game(black_model, white_model):
    board = Go()
    models = {1: black_model, -1: white_model}

    while not board.game_over():  # 你需要在 Go 类中添加 game_over()
        color = board.current_player()  # 添加 current_player()，返回 1 或 -1
        model = models[color]
        move = genMovePolicy(board, color,)  # 根据策略选择合法落子

    return board.history  # 添加 to_sgf()，返回 SGF 格式字符串


def generate_training_data_from_sgf(sgf_dir, output_dir, batch_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    preparePolicyData(sgf_dir, output_dir, batch_size)


def train_from_selfplay_data(data_dir, checkpoint_dir, max_epochs=10):
    net = PlayoutNetwork()
    trainPolicy(net, output_dir=checkpoint_dir, epoch=max_epochs)


def selfplay_train_pipeline(
    best_model_path,
    model_pool_dir,
    sgf_dir,
    data_dir,
    checkpoint_dir,
    selfplay_games=100,
    train_epochs=5
):
    print("Step 1: 自对弈开始")
    run_self_play(best_model_path, model_pool_dir, sgf_dir, games_per_cycle=selfplay_games)

    print("Step 2: SGF → 训练数据")
    generate_training_data_from_sgf(sgf_dir, data_dir, batch_size=1000)

    print("Step 3: 开始训练")
    train_from_selfplay_data(data_dir, checkpoint_dir, max_epochs=train_epochs)

    print("Pipeline 完成！")

