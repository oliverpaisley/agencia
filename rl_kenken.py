import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.base_class import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# import tensorflow as tf

from typing import List

from PIL import Image, ImageDraw, ImageFont

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

import torch
print('CUDA ENABLED:'      , torch.cuda.is_available())
print('CUDE DEVICE COUNT:' , torch.cuda.device_count())
print('CUDE DEVICE NUMBER:', torch.cuda.current_device())
print('CUDE DEVICE NAME:'  ,torch.cuda.device(torch.cuda.current_device()))




class Puzzle(gym.Env):

    def __init__(self, n, targets, ops, dir, puzzle_number):

        super().__init__()

        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        # https://gymnasium.farama.org/api/spaces/
        self.observation_space = spaces.Box(low=0, high=255, shape=(81,81,1), dtype=np.uint8) # 900x900 greyscale image
        self.action_space = spaces.Discrete(4+9) # UP, DOWN, LEFT, RIGHT, 1, ..., 9

        self.Gs = []

        self.n = n # How many columns and rows square are we working with
        self.puzzle_number = puzzle_number
        self.targets = targets
        self.ops     = ops
        self.dir     = dir
        self.reset()


    def reset(self, seed=123):

        # SPARSE REWARD TUNING: Start in a random location in puzzle
        self.i = np.random.choice(range(1,self.n+1))

        self.cells = self.make_cells(self.n)
        self.cages = self.make_cages(self.targets, self.ops, self.dir)
        # for cell in self.cells: print(cell.value) # DEBUG
        self.rows = self.make_rows(self.n)
        self.cols = self.make_cols(self.n)
        self.G = 0
        self.steps = 0

        # SPARSE REWARD TUNING: Start each value with a random value from 0 to n
        # I want 0 in there because it will learn to remove them due to other reward tuning
        for cell in self.cells:
            cell.value = np.random.choice(range(0,self.n+1))

        return (self.get_state(), {})


    def render(self):
        pass


    def close(self):
        pass


    def make_cells(self, n):
        cells = []
        for row in range(n):
            for col in range(n):
                index = (row*n)+col
                cells.append(Cell(index, row, col))
        return cells


    def make_cages(self, targets, ops, dir):

        cages = []
        max_cage_dir = max(dir)

        # Loop through cage_dir and if cage_number matches then add those attributes
        for cage_num in range(max_cage_dir+1):

            cage = Cage(cage_num) # create cage shell

            for e,v in enumerate(dir):
                if v == cage_num: # if cell is part of current cage_num then add its attributes

                    cage.target = targets[e]
                    cage.op     = ops[e]

                    cage.cells.append(self.cells[e])

            cages.append(cage)

        return cages


    def make_rows(self, n):

        rows = []
        row_num = 0

        while row_num != n:

            row = Row(row_num)

            # print("-"*10) # DEBUG
            for i in range(n):

                index = (row_num*n)+i
                # print(index) # DEBUG
                row.cells.append(self.cells[index])

            rows.append(row)
            row_num+=1

        return rows


    def make_cols(self, n):

        cols = []
        col_num = 0

        while col_num != n:

            col = Column(col_num)

            # print("-"*10) # DEBUG
            for i in range(n):

                index = (i*n)+col_num
                # print(index) # DEBUG
                col.cells.append(self.cells[index])

            cols.append(col)
            col_num+=1

        return cols


    def move_up(self, n, i):
        row = self.cells[i].row
        col = self.cells[i].col

        if row == 0:
            return i
        else:
            return i - n


    def move_down(self, n, i):
        row = self.cells[i].row
        col = self.cells[i].col

        if row == (n-1):
            return i
        else:
            return i + n


    def move_left(self, n, i):
        row = self.cells[i].row
        col = self.cells[i].col

        if col == 0:
            return i
        else:
            return i - 1


    def move_right(self, n, i):
        row = self.cells[i].row
        col = self.cells[i].col

        if col == (n-1):
            return i
        else:
            return i + 1


    def step(self, action):
        
        if self.steps == 0:
            # print("RESET")
            self.reset()

        truncated = False
        info = {}

        reward = 0

        # SPARSE REWARD TUNING: Exploration (eliminate 0s)
        for cell in self.cells:
            if cell.value == 0:
                reward -= 1

        # SPARSE REWARD TUNING: PENALIZE FOR EVERY INVALID CAGE
        for cage in self.cages:
            if not cage.evaluate():
                reward -= 1

        # SPARSE REWARD TUNING: PENALIZE FOR EVERY NON-DISTINCT ROW
        for row in self.rows:
            if not (row.evaluate() == self.n):
                reward -= 1

        # SPARSE REWARD TUNING: PENALIZE FOR EVERY NON-DISTINCT COL
        for col in self.cols:
            if not (col.evaluate() == self.n):
                reward -= 1

        self.G += reward

        start_pos = self.i
        start_val = self.cells[self.i].value

        if action   == 0: self.i = self.move_up(   self.n, self.i)
        elif action == 1: self.i = self.move_down( self.n, self.i)
        elif action == 2: self.i = self.move_left( self.n, self.i)
        elif action == 3: self.i = self.move_right(self.n, self.i)
        else:
            value = action - 3 # just aligns action number to cell value to be inserted
            if value <= self.n: self.cells[self.i].value = value

        end_pos = self.i
        end_val = self.cells[self.i].value

        done = self.evaluate()

        s = "ACTION {:2} START_POS {:2} START_VAL {:2} END_POS {:2} END_VAL {:2} RETURN {:5} DONE {:2} CELLS {}"
        s = s.format(action, start_pos, start_val, end_pos, end_val, self.G, done, [cell.value for cell in self.cells])
        # print(s) # print step message

        self.steps += 1
        
        if done:
            # REWARD TUNING: Give agent huge reward for solving a puzzle
            self.G += 10000 # give the agent a HUGE reward for solve
            self.Gs.append(self.G)
            self.steps = 0
            # self.logger.record("return", self.G)
            # print('PUZZLE #', self.puzzle_number, ":", [cell.value for cell in self.cells])
            print('SOLVED PUZZLE #', self.puzzle_number, ":", [cell.value for cell in self.cells], 'CURRENT+AVERAGE RETURN:', round(self.G,0), round(np.average(self.Gs),0))

        # if self.steps % 1000 == 0: print(info)

        # (observation, reward, terminated, truncated, info)
        return (self.get_state(), reward, done, truncated, info)


    def evaluate(self):

        # print('CELL VALUES:', [cell.value for cell in self.cells])

        # 1. Evaluate Cages
        for e,cage in enumerate(self.cages):
            # print(e, cage.evaluate())
            if not cage.evaluate():
                # print('CAGE', e, 'FAILED')
                return False

        # 2. Evaluate Rows
        for e,row in enumerate(self.rows):
            # print('n = {}, len(row) = {}'.format(self.n, row.evaluate()))
            if (row.evaluate() != self.n):
                return False

        # 3. Evaluate Columns
        for e,col in enumerate(self.cols):
            # print('n = {}, len(col) = {}'.format(self.n, col.evaluate()))
            if (col.evaluate() != self.n):
                return False

        return True


    def get_state(self):
        return self.create_image()


    def summary(self):

        # PRINT CAGES
        print("### CAGE SUMMARY ###")
        for cage in self.cages:
            print('CAGE NUMBER:\t'  , cage.num   )
            print('CAGE TARGET:\t'  , cage.target)
            print('CAGE OPERATOR:\t', cage.op    )
            for cell in cage.cells:
                print(cell.i, cell.value)

        # PRINT ROWS
        print("### ROW SUMMARY ###")
        for row in self.rows:
            print('ROW', row.m)
            for cell in row.cells:
                print(cell.i, cell.value)

        # PRINT COLS
        print("### COLUMN SUMMARY ###")
        for col in self.cols:
            print('COL', col.n)
            for cell in col.cells:
                print(cell.i, cell.value)

        # PRINT CELL VALUES
        print('CELL VALUES:', [cell.value for cell in self.cells])


    def draw_cage_line(self, im, draw, pos, cage_cell_locs):

        pos_row, pos_col = self.convert_pos(pos, self.n)
        # pos_row, pos_col = convert_pos(cell.i, n)
        center_x = (pos_col*100)+50
        center_y = (pos_row*100)+50
        dist     = 51
        # print(pos, pos_row, pos_col, center_x, center_y, dist)

        # print('cell_locs:', cage_cell_locs)
        # print(
        #      self.move_up(   self.n, pos)
        #     ,self.move_down( self.n, pos)
        #     ,self.move_left( self.n, pos)
        #     ,self.move_right(self.n, pos)
        # )

        # TOP
        if (im.getpixel((center_x,center_y-dist)) == 0) and not (center_y-dist < 0) and (self.move_up(self.n, pos) in cage_cell_locs):
            # print('white top')
            draw.line((pos_col*100+0  , pos_row*100+0  , pos_col*100+100, pos_row*100+0  ), fill="white", width=5) # top
        else:
            # print('black top')
            draw.line((pos_col*100+0  , pos_row*100+1  , pos_col*100+100, pos_row*100+1  ), fill="black", width=2) # top

        # BOTTOM
        if (im.getpixel((center_x, center_y+dist))) == 0 and not (center_y+dist > self.n*100) and (self.move_down(self.n, pos) in cage_cell_locs):
            # print('white bottom')
            draw.line((pos_col*100+0  , pos_row*100+100, pos_col*100+100, pos_row*100+100), fill="white", width=5) # bottom
        else:
            # print('black bottom')
            draw.line((pos_col*100+0  , pos_row*100+100-1, pos_col*100+100, pos_row*100+100-1), fill="black", width=2) # bottom

        # LEFT
        if (im.getpixel((center_x-dist, center_y)) == 0) and not (center_x-dist < 0) and (self.move_left(self.n, pos) in cage_cell_locs):
            # print('white left')
            draw.line((pos_col*100+0  , pos_row*100+0  , pos_col*100+0  , pos_row*100+100), fill="white", width=5) # left
        else:
            # print('black left')
            draw.line((pos_col*100+0+1  , pos_row*100+0  , pos_col*100+0+1  , pos_row*100+100), fill="black", width=2) # left

        # RIGHT
        if (im.getpixel((center_x+dist, center_y)) == 0) and not (center_x+dist > self.n*100) and (self.move_right(self.n, pos) in cage_cell_locs):
            # print('white right')
            draw.line((pos_col*100+100, pos_row*100+0  , pos_col*100+100, pos_row*100+100), fill="black", width=5) # right
        else:
            # print('black right')
            draw.line((pos_col*100+100-1, pos_row*100+0  , pos_col*100+100-1, pos_row*100+100), fill="black", width=2) # right

        return im


    def create_image(self):

        # color = [0,255] = [BLACK, WHITE]
        # 128 = grey

        # Configure Font for Later
        fnt_1 = ImageFont.truetype(".\\IBMPlexMono-Regular.ttf", 50)
        fnt_2 = ImageFont.truetype(".\\IBMPlexMono-Regular.ttf", 15)

        n   = self.n  # width and height of puzzle
        pos = self.i  # current position

        # 1. Create Image Background
        im = Image.new(mode="L", size=(900, 900), color=255) # White
        draw = ImageDraw.Draw(im)


         # 2. Shade Current Position
        pos_box = Image.new(mode="L", size=(100, 100), color=128) # Grey Box

        pos_row, pos_col = self.convert_pos(pos, n)
        im.paste(pos_box, (pos_col*100, pos_row*100))


        # 3. Draw Grid (needs to be after position shading)
        for i in range(n+1):
            draw.line((i*100,0,i*100,n*100), fill=0, width=1) # draw col
            draw.line((0,i*100,n*100,i*100), fill=0, width=1) # draw row


        # 4. Add Cell Values
        for e,cell in enumerate(self.cells):
            pos_row, pos_col = self.convert_pos(cell.i, n)
            x = (pos_col*100)+50
            y = (pos_row*100)+50
            draw.text((x,y), str(cell.value), fill="black", anchor='mm', font=fnt_1) # middle-middle


        # 5. Add Target Values + Operator Symbols
        for cage in self.cages:
            for e,cell in enumerate(cage.cells):
                if e == 0:
                    pos_row = cell.row
                    pos_col = cell.col
                    target = cage.target
                    operator = self.convert_op(cage.op)

            x = (pos_col*100)+10
            y = (pos_row*100)+fnt_2.size+5
            draw.text((x,y), "{} {}".format(target, operator), fill="black", anchor='ls', font=fnt_2) # left-baseline


        # 6. Add Cage Borders
        for e, cage in enumerate(self.cages):
            cage_cell_locs = [cell.i for cell in cage.cells]
            for cell in cage.cells:
                im = self.draw_cage_line(im, draw, cell.i, cage_cell_locs)

        #
        # im.save('test_900.jpeg') # OPTIONAL
        im = im.resize((81,81))
        # im.save('test_81.jpeg') # OPTIONAL
        # https://stackoverflow.com/questions/61578389/ppo-algorithm-converges-on-only-one-action
        # return np.reshape(np.array(im.getdata(), dtype=np.uint8)/255, (81,81,1)) # normalize values
        return np.reshape(np.array(im.getdata(), dtype=np.uint8), (81,81,1)) # normalize values


    def convert_pos(self, pos, n):
        row, col = 0,0
        for i in range(pos):
            col += 1
            if col >= n:
                row += 1
                col = 0

        return row, col


    def convert_op(self, op: int) -> str:

        # OPERATOR CODES:
        # 0 = ADD
        # 1 = SUB
        # 2 = MUL
        # 3 = DIV
        # 4 = NOP

        if op == 0: return "+"
        if op == 1: return "-"
        if op == 2: return "x"
        if op == 3: return "÷"
        if op == 4: return ""


class Cage:

    def __init__(self, num: int) -> None:
        self.num               = num
        self.cells: List[Cell] = []
        self.target: int       = -1
        self.op: int           = -1


    def evaluate(self) -> bool:

        # OPERATOR CODES:
        # 0 = ADD
        # 1 = SUB
        # 2 = MUL
        # 3 = DIV
        # 4 = NOP

        cur_val = 0

        if self.op == 0:
            cur_val = sum([cell.value for cell in self.cells])

        elif self.op == 1:
            sub_value_1 = self.cells[0].value - self.cells[1].value
            sub_value_2 = self.cells[1].value - self.cells[0].value

            if sub_value_1 > 0:
                cur_val = sub_value_1
            else:
                cur_val = sub_value_2

        elif self.op == 2:
            for e,i in enumerate(self.cells):
                if e == 0:
                    cur_val = self.cells[0].value
                else:
                    cur_val *= self.cells[e].value

        elif self.op == 3:
            if (self.cells[0].value == 0) or (self.cells[1].value == 0): cur_val = 0
            else:
                div_value_1 = self.cells[0].value / self.cells[1].value
                div_value_2 = self.cells[1].value / self.cells[0].value

                if div_value_1 > 1:
                    cur_val = div_value_1
                else:
                    cur_val = div_value_2

        elif self.op == 4:
            cur_val = self.cells[0].value

        # print('op, cur_val, target', self.op, cur_val, self.target)
        return cur_val == self.target


class Cell:

    def __init__(self, i: int, row: int, col: int) -> None:
        self.i          = i
        self.row        = row
        self.col        = col
        self.value: int = 0


class Row:

    def __init__(self, m: int) -> None:
        self.m                 = m  # index of row
        self.cells: List[Cell] = [] # cells that make up Row


    def evaluate(self) -> int:
        return len(set([cell.value for cell in self.cells]))


class Column:

    def __init__(self, n: int) -> None:
        self.n                 = n  # index of column
        self.cells: List[Cell] = [] # cells that make up Column


    def evaluate(self) -> int:
        return len(set([cell.value for cell in self.cells]))



def create_puzzle(number):

    # TEMPLATE
    # if number == 0:
    #     cage_targets = []
    #     cage_ops     = []
    #     cage_dir     = []
    #     n            = 0

    # HOLDOUTS
    if number == 559: # 3H
        cage_targets = [2, 18, 18, 2, 2, 18, 2, 2, 18]
        cage_ops     = [4, 2, 2, 1, 1, 2, 3, 3, 2]
        cage_dir     = [0, 1, 1, 2, 2, 1, 3, 3, 1]
        n            = 3

    if number == 153360: # 4H
        cage_targets = [12,12,12,7,2,9,7,7,1,9,9,2,1,2,2,2]
        cage_ops     = [2,2,2,0,4,0,0,0,1,0,0,1,1,3,3,1]
        cage_dir     = [0,0,0,1,2,3,1,1,4,3,3,5,4,6,6,5]
        n            = 4

    if number == 57244: # 5H
        cage_targets = [90,90,90,10,10,2,2,90,10,4,1,1,12,12,12,1,5,12,12,12,1,5,4,4,12]
        cage_ops     = [2,2,2,0,0,3,3,2,0,4,1,1,2,0,0,1,0,2,2,0,1,0,1,1,0]
        cage_dir     = [0,0,0,1,1,2,2,0,1,3,4,4,5,6,6,7,8,5,5,6,7,8,9,9,6]
        n            = 5


    # TRAINING

    if number == 2: # All no-ops
        cage_targets = [2,1,1,2]
        cage_ops     = [4,4,4,4]
        cage_dir     = [0,1,2,3]
        n            = 2

    if number == 22:
        cage_targets = [3,3,2,2]
        cage_ops     = [0,0,3,3]
        cage_dir     = [0,0,1,1]
        n            = 2

    if number == 618: # 3x3 -> TR1
        cage_targets = [5, 2, 2, 5, 3, 3, 2, 2, 2]
        cage_ops     = [0, 3, 3, 0, 3, 3, 1, 1, 4]
        cage_dir     = [0, 1, 1, 0, 2, 2, 3, 3, 4]
        n            = 3

    if number == 531: # 3x3 -> TR2
        cage_targets = [3,3,4,1,4,4,1,3,3]
        cage_ops     = [3,3,2,1,2,2,1,3,3]
        cage_dir     = [0,0,1,2,1,1,2,3,3]
        n            = 3

    if number == 495: # 3x3 -> TR3
        cage_targets = [3,5,3,2,5,3,2,2,2]
        cage_ops     = [4,0,0,3,0,0,3,1,1]
        cage_dir     = [0,1,2,3,1,2,3,4,4]
        n            = 3

    if number == 641: # 3x3 -> TR4
        cage_targets = [1,2,3,1,2,2,1,1,1]
        cage_ops     = [1,2,4,1,2,2,4,1,1]
        cage_dir     = [0,1,2,0,1,1,3,4,4]
        n            = 3

    if number == 682: # 3x3 -> TR5
        cage_targets = [3,3,3,1,2,3,1,2,2]
        cage_ops     = [0,0,3,1,1,3,1,1,4]
        cage_dir     = [0,0,1,2,3,1,2,3,4]
        n            = 3

    if number == 3: # All no-ops
        cage_targets = [3,2,1,2,1,3,1,3,2]
        cage_ops     = [4,4,4,4,4,4,4,4,4]
        cage_dir     = [0,1,2,3,4,5,6,7,8]
        n            = 3

    if number == 153443: # 4TR1
        cage_targets = [3,6,1,1,3,6,12,12,9,6,5,2,9,9,5,2]
        cage_ops     = [1,2,1,1,1,2,2,2,0,2,0,3,0,0,0,3]
        cage_dir     = [0,1,2,2,0,1,3,3,4,1,5,6,4,4,5,6]
        n            = 4

    if number == 54026: # 4TR2
        cage_targets = [1,1,1,1,1,12,2,1,8,12,2,7,8,8,8,7]
        cage_ops     = [1,1,1,1,1,2,3,1,0,2,3,0,0,0,0,0]
        cage_dir     = [0,1,1,2,0,3,4,2,5,3,4,6,5,5,5,6]
        n            = 4

    if number == 153331: # 4TR3
        cage_targets = [12,12,12,2,8,1,1,2,8,6,6,6,8,3,2,2]
        cage_ops     = [2,2,2,3,2,1,1,3,2,0,0,0,2,4,3,3]
        cage_dir     = [0,0,0,1,2,3,3,1,2,4,4,4,2,5,6,6]
        n            = 4

    if number == 5438: # 4TR4
        cage_targets = [6,6,6,1,1,3,2,1,1,3,2,2,8,8,8,2]
        cage_ops     = [2,2,2,1,1,1,3,1,1,1,3,3,0,0,0,3]
        cage_dir     = [0,0,0,1,2,3,4,1,2,3,4,5,6,6,6,5]
        n            = 4

    if number == 153602: # 4TR5
        cage_targets = [2,1,6,6,2,1,6,1,12,1,7,1,12,12,7,7]
        cage_ops     = [3,4,2,2,3,1,2,1,3,1,0,1,2,2,0,0]
        cage_dir     = [0,1,2,2,0,3,2,4,5,3,6,4,5,5,6,6]
        n            = 4

    if number == 57662: # 5TR1
        cage_targets = [12,30,30,3,3,12,2,30,1,1,12,2,10,15,15,1,12,10,10,15,1,12,12,2,2]
        cage_ops     = [0,2,2,1,1,0,3,2,1,1,0,3,0,2,2,1,0,0,0,2,1,0,0,3,3]
        cage_dir     = [0,1,1,2,2,0,3,1,4,4,0,3,5,6,6,7,8,5,5,6,7,8,8,9,9]
        n            = 5

    if number == 57630: # 5TR2
        cage_targets = [60,60,7,2,2,60,6,7,3,3,3,6,7,2,2,3,6,7,11,11,10,10,10,11,3]
        cage_ops     = [2,2,0,3,3,2,0,0,1,1,1,0,0,1,1,1,0,0,0,0,2,2,2,0,4]
        cage_dir     = [0,0,1,2,2,0,3,1,4,4,5,3,6,7,7,5,3,6,8,8,9,9,9,8,10]
        n            = 5

    if number == 57093: # 5TR3
        cage_targets = [2,1,24,24,24,4,1,2,2,24,4,1,2,2,2,10,1,20,20,2,10,10,20,3,3]
        cage_ops     = [4,1,2,2,2,1,1,1,1,2,1,1,3,3,1,0,1,2,2,1,0,0,2,1,1]
        cage_dir     = [0,1,2,2,2,3,1,4,4,2,3,5,6,6,7,8,5,9,9,7,8,8,9,10,10]
        n            = 5

    if number == 156619: # 5TR4
        cage_targets = [12,12,30,30,30,2,12,12,10,10,2,2,2,10,10,3,6,6,8,10,3,1,1,8,8]
        cage_ops     = [0,0,2,2,2,1,0,0,2,2,1,3,3,0,0,1,2,2,0,0,1,1,1,0,0]
        cage_dir     = [0,0,1,1,1,2,0,0,3,3,2,4,4,5,5,6,7,7,8,5,6,9,9,8,8]
        n            = 5

    if number == 57668: # 5TR5
        cage_targets = [1,1,6,1,3,20,20,6,1,3,20,2,2,1,1,30,30,4,4,10,30,2,2,10,10]
        cage_ops     = [1,1,0,1,1,2,2,0,1,1,2,3,3,1,1,2,2,0,0,0,2,3,3,0,0]
        cage_dir     = [0,0,1,2,3,4,4,1,2,3,4,5,5,6,6,7,7,8,8,9,7,10,10,9,9]
        n            = 5

    # TEMPLATE
    # if number == 0:
    #     cage_targets = []
    #     cage_ops     = []
    #     cage_dir     = []
    #     n            = 0

    # OPERATOR CODES
    # 0 = ADD
    # 1 = SUB
    # 2 = MUL
    # 3 = DIV
    # 4 = NOP

    # TEMPLATE
    # if number == 0:
    #     cage_targets = []
    #     cage_ops     = []
    #     cage_dir     = []
    #     n            = 0


    # DEBUG STATEMENTS
    # print('CAGE TARGETS LENGTH:\t'   , len(cage_targets))
    # print('CAGE OPERATIONS LENGTH:\t', len(cage_ops)    )
    # print('CAGE DIRECTOR LENGTH:\t'  , len(cage_dir)    )

    if len(cage_targets) != n*n: print('CAGE TARGETS ERROR!', number)
    if len(cage_ops    ) != n*n: print('CAGE OPS ERROR!'    , number)
    if len(cage_dir    ) != n*n: print('CAGE DIR ERROR!'    , number)

    return Puzzle(n, cage_targets, cage_ops, cage_dir, number)


def check_my_env(env):

    # Use the checker to validate the environment
    try:
    # For a more thorough check, set warn=True
        check_env(env, warn=False)
        print("Environment passed all checks!")
    except gym.error.Error as e:
        print("Environment check failed:")
        print(e)


def register_env(env):

    myEnv_id = 'agencia/KenKen-v0' # It is best practice to have a space name and version number.
    gym.envs.registration.register(
        id=myEnv_id
        ,entry_point=env
        # ,max_episode_steps=20000 # Customize to your needs.
        # ,reward_threshold=500 # Customize to your needs.
    )
    print('ENVIRONMENT REGISTERED')


def train(puzzle_number):

    # TENSORBOARD INTEGRTATION: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html

    total_timesteps = 10000
    model_name = f"./kenken_v0"
    logdir = f"logs/"

    if not os.path.exists(logdir): os.makedirs(logdir)
    
    puzzle = create_puzzle(puzzle_number) # create puzzle given puzzle number
    env = puzzle
    # vec_env = make_vec_env(myEnv_id, n_envs=1)#, env_kwargs={"KenKen": KenKen})

    # register_env(env)
    # check_my_env(env)

    # LOAD OR CREATE MODEL
    # try:

    model = PPO.load(
          model_name
        , env=env
        , verbose=1
        , tensorboard_log=logdir
        # , print_system_info=False
    ) # load PPO model

    # model = DQN.load(
    #       model_name
    #     , env=env
    #     , print_system_info=False
    #     # , handle_timeout_termination=False
    #     , optimize_memory_usage=True
    #     , replay_buffer_kwargs={'handle_timeout_termination': False}
    #     , tensorboard_log=logdir
    #     # , callback=rewards_callback
    #     , verbose=1
    # ) # load DQN model
    # print('loaded model')
    
    # except:
    
    # model = PPO(
    #      "CnnPolicy"
    #     , env
    #     , verbose=1
    #     , tensorboard_log=logdir
    #     # , optimize_memory_usage=True
    #     # , replay_buffer_kwargs={'handle_timeout_termination': False}
    # )
    # pass
    # print('CREATED MODEL')


    model.learn(
          total_timesteps=total_timesteps
        , progress_bar=True
        , tb_log_name=f"KENKEN"
        , reset_num_timesteps=False
    )

    model.save(model_name)
    print('MODEL SAVED')
    del model



# TRAINING
puzzle_lst = []

puzzle_lst.append(57662 ) # 5TR1
puzzle_lst.append(57630 ) # 5TR2
puzzle_lst.append(57093 ) # 5TR3
puzzle_lst.append(156619) # 5TR4
puzzle_lst.append(57668 ) # 5TR5

puzzle_lst.append(153443) # 4TR1
puzzle_lst.append(54026 ) # 4TR2
puzzle_lst.append(153331) # 4TR3
puzzle_lst.append(5438  ) # 4TR4
puzzle_lst.append(153602) # 4TR5

puzzle_lst.append(618   ) # 3TR1
puzzle_lst.append(531   ) # 3TR2
puzzle_lst.append(495   ) # 3TR3
puzzle_lst.append(641   ) # 3TR4
puzzle_lst.append(682   ) # 3TR5

# puzzle_lst.append(2     ) # 2TR1
# puzzle_lst.append(22    ) # 2TR2

loops = 1000
train_lst = puzzle_lst * loops

for e,puzzle_number in enumerate(train_lst):
    # if e % len(puzzle_lst) == 0: clear_output(wait=True) # clear output of previous runs (for running long training runs)
    print('RUNNING PUZZLE #{}'.format(puzzle_number))
    train(puzzle_number)

print('\n\nDONE')