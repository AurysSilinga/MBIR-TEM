# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 14:52:48 2013

@author: Jan
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import abc

x = np.linspace(0, 2*pi, 1000)
y = np.where(x<pi, -1/np.tan(x+1E-30), 1/np.tan(x+1E-30))#20*np.arctan(np.exp(x))

def func(x, pos):
    return '{}'.format(x*10)

formatter = FuncFormatter(lambda x, pos: '{:g}'.format(x))

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(x, y)
axis.set_ylim(-10, 10)
axis.yaxis.set_major_formatter(formatter)

plt.show()


class PropTest(object):

    def __init__(self, a):
        self.a = a

    def __call__(self, b):
        print '{} + {} = {}'.format(self.a, b, self.a+b)
    
    @property
    def a(self):
        print 'getter'
        return self._a

    @a.setter
    def a(self, value):
        print 'setter'
        self._a = value + 3

    def __div__(self, other):
        return PropTest(self.a/other)


PropTest(1)(2)

karl = PropTest(1)

karl.a = 5

print karl.a

testo = PropTest(3) / 3

print testo.a

class ReturnTest(object):

    def test(self):
        return

print ReturnTest().test()



class AbstractTest(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self):
        print self.args
        print self.kwargs
        self._test(*self.args, **self.kwargs)

    @abc.abstractmethod
    def _test(self, *args, **kwargs):
        return

class ConcreteTest(AbstractTest):
    '''DOCSTRING'''
    
    def _test(self, a, b, c):
        print 'a =', a
        print 'b =', b
        print 'c =', c

print ''

ConcreteTest(a=1, c=2, b=3)()


