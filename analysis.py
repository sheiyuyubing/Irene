import threading
import time
from genMove import *

stop_event = threading.Event()
analyze_thread = None

import torch

def get_top_moves(go, topk=5):
    policy = getPolicyNetResult(go)  # shape: (361,)
    value = getValueNetResult(go)    # float 0.0~1.0
    winrate = int(value * 100)  # ת��Ϊ GTP ��Ҫ��������ʽ

    # ��ȡԤ����ǰ topk ��������λ������������
    topk_indices = torch.topk(policy, topk).indices.tolist()

    # ģ�� visits�����Զ�����ԣ�
    max_visits = 1000
    decay = 100

    move_infos = []
    for i, index in enumerate(topk_indices):
        x, y = toPosition(index)
        if (x, y) == (None, None):
            print('pass')
            return
        moveResult = toStrPosition(x, y)  # �Զ��庯����������ת GTP �ַ���
        visits = max_visits - i * decay

        move_infos.append({
            'move': moveResult,
            'visits': visits,
            'winrate': winrate
        })

    return move_infos



def start_analysis(go, interval):
    global analyze_thread, stop_event

    stop_event.clear()

    def analyze_loop():

        while not stop_event.is_set():
            # ����ģ����� info move
            move_stats = get_top_moves(go)  # ���ض�� move �� dict
            for stat in move_stats:
                print("info", end=' ')
                for key, value in stat.items():
                    print(f"{key} {value}", end=' ')
                print()
            time.sleep(interval)  # ���Ʒ��������Ƶ��

    analyze_thread = threading.Thread(target=analyze_loop)
    analyze_thread.start()

def stop_analysis():
    global analyze_thread, stop_event
    stop_event.set()
    if analyze_thread:
        analyze_thread.join()
        analyze_thread = None