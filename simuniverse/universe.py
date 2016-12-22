# -*- coding: utf-8 -*-
'''

@author: Rem
@contact: remch183@outlook.com
@site: 
@software: PyCharm Community Edition
@file: universe.py
@time: 2016/12/14 19:32
'''
__author__ = 'Rem'

import numpy as np
import logging
from .civ import *


class Universe:

    def __init__(self):
        self.events = EventPool()
        self.gl = Galaxy()
        self.civ = Civilization(self.gl, self.events)

    def start_sim(self):
        """开始模拟"""
        # TODO: 事件记录，保存为csv格式(记录年限应当可以更改)
        # TODO: 保存模型运行状态，可以随时终止（或者可以设置年限终止）
        pass



