# -*- coding: utf-8 -*-
"""
Created on Fri Dec 05 09:24:02 2014

@author: Jan
"""


import numpy as np


class Cell(object):

    def __init__(self):
        self.reset()

    def __str__(self):
        return 'value: {}, possibilities: {}'.format(self.value, self.possibilities)

    def reset(self):
        self.value = 0
        self.possibilities = range(1, 10)
        np.random.shuffle(self.possibilities)


class Field(object):

    def __init__(self):
        self.field = np.asarray([[Cell() for i in range(9)] for j in range(9)])
        self.fill()

    def __call__(self):
        return np.asarray([[self.field[j, i].value for i in range(9)] for j in range(9)])

    def get_i(self, pos):
        return pos % 9

    def get_j(self, pos):
        return pos // 9

    def get_cell(self, pos):
        return self.field[self.get_j(pos), self.get_i(pos)]

    def check_row(self, pos):
        cell_val = self.get_cell(pos).value
        row = self.field[self.get_j(pos), :]
        row_vals = [cell.value for cell in row]
        row_vals[self.get_i(pos)] = 0
        print 'i:{}, j:{}, row_vals:{}'.format(self.get_i(pos), self.get_j(pos), row_vals)
        print cell_val in row_vals
        if cell_val in row_vals:
            return False
        else:
            return True

    def check_square(self, pos):
        return True

    def fill(self):
        pos = 0
        while pos < 81:
            cell = self.get_cell(pos)
            cell.value = cell.possibilities.pop()
            print self()
            print cell
            if self.check_row(pos) and self.check_square(pos):
                pos += 1
            else:
                self.get_cell(pos).reset()
                pos -= 1
            print 'Position: {:2}, Value: {}'.format(pos, cell.value)

test = Field()
print test()
