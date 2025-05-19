from net import *
from go import *
import sys
import  os
from genMove import *
from analysis import  start_analysis, stop_analysis

def get_path(relative_path):
    try:
        base_path = sys._MEIPASS  # pyinstaller打包后的路径
    except AttributeError:
        base_path = os.path.abspath(".")  # 当前工作目录的路径

    return os.path.normpath(os.path.join(base_path, relative_path))  # 返回实际路径


go = Go()

policy_dir  = get_path("checkpoints/best_model.pt")
value_dir = get_path("checkpoints/value_model_cycle_048.pt")

# stderr output 'GTP ready'
sys.stderr.write('GTP ready\n')

while True:
    # implement GTP (Go Text Protocol)
    line = input().strip()

    stop_analysis()

    if line == 'quit':
        break
    print('= ', end='')
    if line == 'boardsize 19':
        print('boardsize 19')
    elif line.startswith('komi'):
        print('komi')
    if line == 'clear_board':
        go = Go()
        print('clear_board')

    elif line.startswith('play'):
        # play B F12
        color, position = line.split()[1:]

        if position == 'pass':
            print('play PASS')
            go.current_color = -color
            go.passcount = go.passcount+1
        else:
            # position = F12
            y, x = position[0], position[1:]

            #    A B C D E F G H J K L M N O P Q R S T
            # 19
            # 18
            # 17

            x = 19 - int(x)
            y = charToIndex[y]

            color = colorCharToIndex[color]

            if go.move(x, y) == False:
                print('Illegal move')
            else:
                print('ok')


    elif line.startswith('genmove'):
        colorChar = line.split()[1]
        go.current_color = colorCharToIndex[colorChar]

        if len(sys.argv) > 1 and sys.argv[1] == 'MCTS':
            genMoveMCTS(go,policy_dir)
        else:
            genMovePolicy(go,policy_dir)

    elif line.startswith('showboard'):
        for i in range(19):
            for j in range(19):
                if go.board[i][j] == 1:
                    print('X', end='')
                elif go.board[i][j] == -1:
                    print('O', end='')
                else:
                    print('.', end='')
            print()
    # name
    elif line.startswith('name'):
        print('Irene')
    # version
    elif line.startswith('version'):
        print('0.1')
    # protocol_version
    elif line.startswith('protocol_version'):
        print('2')
    # list_commands
    elif line.startswith('list_commands'):
        print('name')
        print('version')
        print('protocol_version')
        print('list_commands')
        print('clear_board')
        print('boardsize')
        print('showboard')
        print('play')
        print('genmove')
        print('analyze')
        print('quit')

    # 新增：分析模式
    elif line.startswith('analyze'):
        # 示例：analyze B 5
        tokens = line.split()
        if len(tokens) >= 2:
            colorChar = tokens[1]
            willPlayColor = colorCharToIndex[colorChar]
            interval =  int(tokens[2])
            start_analysis(go,interval,policy_dir,value_dir)

            # print(f"info move D4 visits 100 winrate 55.0 prior 0.05")
            # print(f"info move Q16 visits 80 winrate 52.5 prior 0.04")
            # print(f"info move C3 visits 50 winrate 51.2 prior 0.03")
            # print(f"info move pass visits 20 winrate 48.0 prior 0.01")
        else:
            print('invalid analyze command')

    else:
        print('Unknown command')


    print()
    # testKill()