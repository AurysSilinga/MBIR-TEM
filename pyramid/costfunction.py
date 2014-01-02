# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:29:11 2013

@author: Jan
"""# TODO: Docstring!


import numpy as np



class Costfunction:
    # TODO: Docstring!
    
    def __init__(self, y, F):
        # TODO: Docstring!
        self.y = y  # TODO: get y from phasemaps!
        self.F = F  # Forward Model
        self.Se_inv = np.eye(len(y))

    def __call__(self, x):
        # TODO: Docstring!
        y = self.y
        F = self.F
        Se_inv = self.Se_inv
        return (F(x)-y).dot(Se_inv.dot(F(x)-y))

    def jac(self, x):
        # TODO: Docstring!
        y = self.y
        F = self.F
        Se_inv = self.Se_inv
        return F.jac_T_dot(x, (Se_inv.dot(F(x)-y)))

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        F = self.F
        Se_inv = self.Se_inv
        return F.jac_T_dot(x, Se_inv.dot(F.jac_dot(x, vector)))
        # TODO: Murks!


class CFAdapter:
    # TODO: Useless at the moment, because the interface is exactly the same. Use as template!
    
    def __init__(self, costfunction):
        # TODO: Docstring!
        self.costfunction = costfunction

    def __call__(self, x):
        # TODO: Docstring!
        return self.costfunction(x)

    def jac(self, x):
        # TODO: Docstring!
        return self.costfunction.jac(x)

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        return self.costfunction.hess_dot(x, vector)
