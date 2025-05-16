# coding=utf-8
import os
import random
import torch
from go import Go
from prepareData import preparePolicyData
from train import trainPolicy
from net import PolicyNetwork
from genMove import genMovePolicy,toPosition,toStrPosition
from sgfmill import sgf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_self_play(best_model_path, model_pool_dir, sgf_output_dir, games_per_cycle=100):
    os.makedirs(sgf_output_dir, exist_ok=True)

    # 加载当前最强模型
    best_model = PolicyNetwork().to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()

    model_files = [f for f in os.listdir(model_pool_dir) if f.endswith('.pt')]
    assert model_files, "模型池为空"

    for game_idx in range(games_per_cycle):
        opponent_file = random.choice(model_files)
        opponent_path = os.path.join(model_pool_dir, opponent_file)

        opponent_model = PolicyNetwork().to(device)
        opponent_model.load_state_dict(torch.load(opponent_path, map_location=device))
        opponent_model.eval()

        sgf_path = os.path.join(sgf_output_dir, f"game_{game_idx:05d}.sgf")
        self_play_game(best_model, opponent_model, sgf_path)


def self_play_game(black_model_path, white_model_path):
    from genMove import getPolicyNetResult, getValueResult  # 确保你能直接调用这些底层函数

    board = Go()
    sgf_game = sgf.Sgf_game(size=19)
    main_sequence = sgf_game.get_main_sequence()

    models = {
        1: torch.load(black_model_path, map_location=device).to(device).eval(),
        -1: torch.load(os.path.join('model_pool', white_model_path), map_location=device).to(device).eval()
    }

    while not board.game_over():
        color = board.current_color
        model = models[color]

        # 获取策略网络输出
        policy_logits = getPolicyNetResult(board, model)  # shape: [361+1]
        sorted_indices = list(reversed(torch.argsort(policy_logits)))

        move_played = False
        for index in sorted_indices:
            x, y = toPosition(index)
            if (x, y) == (None, None):  # 表示pass
                board.passcount += 1
                main_sequence[-1].set_move('b' if color == 1 else 'w', None)
                move_played = True
                break
            if board.move(x, y):  # 如果落子合法
                board.passcount = 0
                main_sequence.append(sgf_game.new_node())
                main_sequence[-1].set_move('b' if color == 1 else 'w', (x, y))
                move_played = True
                break

        if not move_played:
            # 所有动作都非法，强制 pass
            board.passcount += 1
            main_sequence[-1].set_move('b' if color == 1 else 'w', None)

    # 棋局完成，保存 SGF 字符串和胜负结果
    winner = board.get_winner()  # 你需要自己实现 board.get_winner()
    result_str = "B+" if winner == 1 else "W+"
    sgf_game.set_result(result_str)

    return sgf_game, winner



def generate_training_data_from_sgf(sgf_dir, output_dir, batch_size=1000):
    sgf_list_path = "games/allValid2.txt"
    os.makedirs(os.path.dirname(sgf_list_path), exist_ok=True)

    # 收集 SGF 文件路径
    sgf_files = [os.path.join(sgf_dir, f) for f in os.listdir(sgf_dir) if f.endswith('.sgf')]
    with open(sgf_list_path, 'w', encoding='utf-8') as f:
        for sgf_file in sgf_files:
            f.write(sgf_file + '\n')

    # 调用现有数据生成函数
    preparePolicyData(fileCount=len(sgf_files), batch_size=batch_size, save_dir=output_dir)


def train_from_selfplay_data(data_dir, checkpoint_dir, max_epochs=10):
    net = PolicyNetwork()
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

    print(" Pipeline 完成！")


if __name__ == '__main__':
    selfplay_train_pipeline(
        best_model_path='checkpoints\\best_model.pt',
        model_pool_dir='checkpoints',
        sgf_dir='games/sgf_selfplay',
        data_dir='data/policy_batches',
        checkpoint_dir='models/checkpoints',
        selfplay_games=100,
        train_epochs=5
    )
