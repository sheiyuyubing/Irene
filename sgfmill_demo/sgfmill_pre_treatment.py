# -*- coding: utf-8 -*-
import numpy as np

from sgfmill import sgf, boards

def board_to_3tensor(board):
    size = board.side
    black_plane = np.zeros((size, size), dtype=np.float32)
    white_plane = np.zeros((size, size), dtype=np.float32)
    empty_plane = np.ones((size, size), dtype=np.float32)

    for row in range(size):
        for col in range(size):
            stone = board.get(row, col)
            if stone == 'b':
                black_plane[row, col] = 1
                empty_plane[row, col] = 0
            elif stone == 'w':
                white_plane[row, col] = 1
                empty_plane[row, col] = 0

    return np.stack([black_plane, white_plane, empty_plane])

def extract_all_step_tensors(sgf_path):
    with open(sgf_path, "rb") as f:
        sgf_game = sgf.Sgf_game.from_bytes(f.read())

    sequence = sgf_game.get_main_sequence()
    board_size = sgf_game.get_size()
    board = boards.Board(board_size)

    tensors = []

    for i, node in enumerate(sequence):
        color, move = node.get_move()
        if move is not None:
            row, col = move
            board.play(row, col, color)

        tensor = board_to_3tensor(board)
        tensors.append(tensor)

    return tensors
