# -*- coding: utf-8 -*-
'''

@author: Rem
@contact: remch183@outlook.com
@site: 
@software: PyCharm Community Edition
@file: _test.py
@time: 2016/12/30 20:19
'''
__author__ = 'Rem'

import numpy as np

def a():
    for i in range(1):
        yield i

if __name__ == '__main__':
    t = a()

    print(next(t))
    print(next(t, -10))
    print(next(t, -10))
    print(next(t, -10))
