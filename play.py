import read_board
import tkinter as tk
import neat
import json
import pyautogui
import random
import game
import time
READ_BOARD_RATE = 0.05
SAVE_FILE = './save.json'

activated = False


# def _main():
#     if activated:
#         print('looking for board')
#         board_state = read_board.read_board()
#         if board_state:
#             print('Board found!')
#             outputs = {weight: i for i, weight in enumerate(neat_genome.feedforward(
#                 [row/32768 for col in board_state for row in col]))}
#             g = game.Game(board_state)
#             g.print_board()
#             for weight, i in sorted(outputs.items(), reverse=True):
#                 if g.check_available(i):
#                     _move(i)
#                     break
#     window.after(READ_BOARD_RATE, _main)

def _calculate_move():
    board_state = read_board.read_board()
    if board_state:
        print('Board found!')
        print(board_state)
        outputs = {i: weight for i, weight in enumerate(neat_genome.feedforward(
            [row/32768 for col in board_state for row in col]))}
        g = game.Game(board_state)
        g.print_board()
        for i, weight in sorted(outputs.items(), key=lambda x: x[1], reverse=True):
            if g.check_available(i):
                _move(i)
                break


def _move(direction: int):
    if direction == game.UP:
        print('UP')
        pyautogui.press('w')
    elif direction == game.DOWN:
        print('DOWN')
        pyautogui.press('s')
    elif direction == game.LEFT:
        print('LEFT')
        pyautogui.press('a')
    elif direction == game.RIGHT:
        print('RIGHT')
        pyautogui.press('d')


# def _start_button_callback():
#     global activated
#     print('Start')
#     activated = True


# def _stop_button_callback():
#     global activated
#     print('Stop')
#     activated = False


if __name__ == '__main__':

    # print(read_board.read_board())
    # create genome
    connections = []
    with open(SAVE_FILE) as f:
        saves = json.loads(f.read())
        connections = [d['best_genome'] for d in saves][-1]
    neat_genome = neat.Genome.from_list(16, 4, connections)

    # window = tk.Tk()
    # title = tk.Label(text='Welcome!')
    # title.pack()
    # start_button = tk.Button(text='Start',
    #                          width=25, height=5, command=_start_button_callback)
    # start_button.pack()

    # stop_button = tk.Button(text='Stop',
    #                         width=25, height=5, command=_stop_button_callback)
    # stop_button.pack()

    # window.after(READ_BOARD_RATE, _main)
    # window.mainloop()
    # read_board.read_board()
    while True:
        _calculate_move()
        time.sleep(READ_BOARD_RATE)
