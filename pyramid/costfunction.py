# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:29:11 2013

@author: Jan
"""# TODO: Docstring!


import numpy as np



class Costfunction:
    # TODO: Docstring!
    
    def __init__(self, y, F):
        '''TEST DOCSTRING FOR INIT'''
        # TODO: Docstring!
        self.y = y  # TODO: get y from phasemaps!
        self.F = F  # Forward Model
        self.Se_inv = np.eye(len(y))

    def __call__(self, x):
        '''TEST DOCSTRING FOR CALL'''
        # TODO: Docstring!
        y = self.y
        F = self.F
        Se_inv = self.Se_inv
#        print 'Costfunction - __call__ - input: ', len(x)
        result = (F(x)-y).dot(Se_inv.dot(F(x)-y))
#        print 'Costfunction - __call__ - output:', result
        return result

    def jac(self, x):
        # TODO: Docstring!
        y = self.y
        F = self.F
        Se_inv = self.Se_inv
#        print 'Costfunction - jac - input: ', len(x)
        result = F.jac_T_dot(x, Se_inv.dot(F(x)-y))
#        print 'Costfunction - jac - output:', len(result)
        return result

    def hess_dot(self, x, vector):
        # TODO: Docstring!
        F = self.F
        Se_inv = self.Se_inv
#        print 'Costfunction - hess_dot - input: ', len(vector)
        result = F.jac_T_dot(x, Se_inv.dot(F.jac_dot(x, vector)))
#        print 'Costfunction - hess_dot - output:', len(result)
        return result
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
