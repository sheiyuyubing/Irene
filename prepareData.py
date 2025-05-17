# coding=utf-8
import os
import torch
import numpy as np
from pathlib import Path
from sgfmill import sgf
from go import *
from features import getAllFeatures

colorCharToIndex = {'B': 1, 'W': -1, 'b': 1, 'w': -1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preparePolicySgfFile(fileName):
    with open(fileName, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    # 跳过非 19x19 棋谱
    if game.get_size() != 19:
        raise Exception("Non-19x19 board")

    sequence = game.get_main_sequence()
    winnerChar = game.get_winner()

    validSequence = []
    for node in sequence:
        move = node.get_move()
        if move[0] is None or move[1] is None:
            continue
        if move[0] not in colorCharToIndex:
            continue
        validSequence.append(move)

    # 跳过无落子的棋谱
    if len(validSequence) == 0:
        raise Exception("Empty game")

    go = Go()

    inputData = []
    policyOutput = []

    for move in validSequence:
        willPlayColor = colorCharToIndex[move[0]]
        x, y = move[1]
        inputData.append(getAllFeatures(go))
        policyOutput.append(toDigit(x, y))

        if not go.move( x, y):
            raise Exception("Invalid move")

    willPlayColor = -willPlayColor
    inputData.append(getAllFeatures(go))
    policyOutput.append(19 * 19)  # pass

    inputData = torch.tensor(np.array(inputData)).bool()
    policyOutput = torch.tensor(np.array(policyOutput)).long().reshape(-1)

    return inputData, policyOutput


def preparePolicyData(fileCount, batch_size=1000, save_dir="data/policy_batches"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open('games/allValid2.txt', 'r',encoding='utf-8') as allValidFile:
        allValidLines = allValidFile.readlines()

    allInputData = []
    allPolicyOutput = []
    batch_idx = 0
    processed = 0

    for i, sgfFile in enumerate(allValidLines[:fileCount]):
        try:
            sgfFile = sgfFile.strip()
            inputData, policyOutput = preparePolicySgfFile(sgfFile)

            allInputData.append(inputData)
            allPolicyOutput.append(policyOutput)
            processed += 1

            if processed % batch_size == 0:
                inputTensor = torch.cat(allInputData)
                outputTensor = torch.cat(allPolicyOutput)

                save_path = os.path.join(save_dir, f"policy_batch_{batch_idx:03d}.pt")
                torch.save((inputTensor.cpu(), outputTensor.cpu()), save_path)
                print(f"Saved batch {batch_idx} with {inputTensor.shape[0]} samples to {save_path}")

                allInputData.clear()
                allPolicyOutput.clear()
                batch_idx += 1

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Error: {sgfFile}")
            print(e)

    if allInputData:
        inputTensor = torch.cat(allInputData)
        outputTensor = torch.cat(allPolicyOutput)
        save_path = os.path.join(save_dir, f"policy_batch_{batch_idx:03d}.pt")
        torch.save((inputTensor.cpu(), outputTensor.cpu()), save_path)
        print(f"Saved final batch {batch_idx} with {inputTensor.shape[0]} samples to {save_path}")





def prepareValueSgfFile(fileName):
    with open(fileName, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    # 跳过非19x19棋谱
    if game.get_size() != 19:
        raise Exception("Non-19x19 board")

    sequence = game.get_main_sequence()
    winnerChar = game.get_winner()
    if winnerChar is None:
        raise Exception("No winner info")

    winner = colorCharToIndex.get(winnerChar)
    if winner is None:
        raise Exception("Unknown winner color")

    validSequence = []
    for node in sequence:
        move = node.get_move()
        if move[0] is None or move[1] is None:
            continue
        if move[0] not in colorCharToIndex:
            continue
        validSequence.append(move)

    # 跳过无落子的棋谱
    if len(validSequence) == 0:
        raise Exception("Empty game")

    go = Go()

    for move in validSequence:
        willPlayColor = colorCharToIndex[move[0]]
        x, y = move[1]

        if not go.move( x, y):
            raise Exception("Invalid move")

    willPlayColor = -willPlayColor
    valueInputData = torch.tensor(np.array([getAllFeatures(go)])).bool()
    valueOutput = torch.tensor(np.array([winner == willPlayColor])).long().reshape(-1)

    return valueInputData, valueOutput


def prepareValueData(fileCount, batch_size=1000, save_dir="data/value_batches"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with open('games/allValid2.txt', 'r', encoding='utf-8') as allValidFile:
        allValidLines = allValidFile.readlines()

    allInputData = []
    allOutputData = []
    batch_idx = 0
    processed = 0

    for i, sgfFile in enumerate(allValidLines[:fileCount]):
        try:
            sgfFile = sgfFile.strip()
            valueInput, valueOutput = prepareValueSgfFile(sgfFile)

            allInputData.append(valueInput)
            allOutputData.append(valueOutput)
            processed += 1

            if processed % batch_size == 0:
                inputTensor = torch.cat(allInputData)
                outputTensor = torch.cat(allOutputData)

                save_path = os.path.join(save_dir, f"value_batch_{batch_idx:03d}.pt")
                torch.save((inputTensor.cpu(), outputTensor.cpu()), save_path)
                print(f"Saved batch {batch_idx} with {inputTensor.shape[0]} samples to {save_path}")

                allInputData.clear()
                allOutputData.clear()
                batch_idx += 1

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Error: {sgfFile}")
            print(e)

    if allInputData:
        inputTensor = torch.cat(allInputData)
        outputTensor = torch.cat(allOutputData)
        save_path = os.path.join(save_dir, f"value_batch_{batch_idx:03d}.pt")
        torch.save((inputTensor.cpu(), outputTensor.cpu()), save_path)
        print(f"Saved final batch {batch_idx} with {inputTensor.shape[0]} samples to {save_path}")


if __name__ == '__main__':
    preparePolicyData(300)
    # prepareValueData(20000)