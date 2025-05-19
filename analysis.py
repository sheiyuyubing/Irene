#encoding=utf-8
import threading
import time
from genMove import *

stop_event = threading.Event()
analyze_thread = None

import torch

def get_top_moves(go,policy_model,value_model, topk=5):
    policy = getPolicyNetResult(go,policy_model)  # shape: (361,)
    value = getValueNetResult(go,value_model)    # float 0.0~1.0
    winrate = int(value * 100)  # 转换为 GTP 需要的整数格式

    # 获取预测中前 topk 个动作的位置索引（降序）
    topk_indices = torch.topk(policy, topk).indices.tolist()

    # 模拟 visits（可自定义策略）
    max_visits = 1000
    decay = 100

    move_infos = []
    for i, index in enumerate(topk_indices):
        x, y = toPosition(index)
        if (x, y) == (None, None):
            print('pass')
            return
        moveResult = toStrPosition(x, y)  # 自定义函数，将坐标转 GTP 字符串
        visits = max_visits - i * decay

        move_infos.append({
            'move': moveResult,
            'visits': visits,
            'winrate': winrate
        })

    return move_infos



def start_analysis(go, interval,policy_model,value_model):
    global analyze_thread, stop_event

    stop_event.clear()

    def analyze_loop():

        while not stop_event.is_set():
            # 这里模拟输出 info move
            move_stats = get_top_moves(go,policy_model,value_model,topk = 5)  # 返回多个 move 的 dict
            for stat in move_stats:
                print("info", end=' ')
                for key, value in stat.items():
                    print(f"{key} {value}", end=' ')
                print()
            time.sleep(interval)  # 控制分析输出的频率

    analyze_thread = threading.Thread(target=analyze_loop)
    analyze_thread.start()

def stop_analysis():
    global analyze_thread, stop_event
    stop_event.set()
    if analyze_thread:
        analyze_thread.join()
        analyze_thread = None