# -*- coding: utf-8 -*-
"""

@author: Rem
@contact: remch183@outlook.com
@site:
@software: PyCharm Community Edition
@file: universe.py
@time: 2016/12/14 19:32
"""
import os
import numpy as np
import pandas as pd
try:
    from simuniverse.civ import *
except ImportError:
    from civ import *
from time import time as systime
try:
    from simuniverse import helper
except ImportError:
    import helper

__author__ = 'Rem'
np.random.seed(int(systime()))

# TODO: change this sick hard code part
LOG_MASK = [True, True, True, True, False, False, False, True, True]
G_LOG_MASK = [True, True, True, True, True, False, True]


class Universe:
    def __init__(self, logname=None , gl_size=1000, civ_size=100):
        # TODO 对于各种模式的设置（构成比例，根据数量调整稀疏程度，）
        # TODO 战乱的国家发展速度下降设置
        self.events = EventPool()
        self.gl = Galaxy(size=gl_size)
        self.civ = Civilization(self.gl, self.events, civ_size)
        if logname is None:
            self.logpath = '../log/log/'
        else:
            self.logpath = '../log/log%s/'%(str(logname))
        self.gifpath = self.logpath + 'gif/'
        if not os.path.exists(self.logpath):
            os.mkdir(self.logpath)
        if not os.path.exists(self.gifpath):
            os.mkdir(self.gifpath)
        self.clear_path()

    def clear_path(self):
        for file in os.listdir(self.logpath):
            try:
                os.remove(os.path.join(self.logpath, file))
            except PermissionError:
                pass
        print("Clear old logs done.")

    def start_sim(self, max_year=1000):
        """开始模拟"""
        print("=" * 20, '\nSim begin')
        print("log:" + self.logpath)
        print('=' * 20)
        battle_num = 0
        for year in range(max_year):
            self.civ.time = year
            self.civ.refresh_all()
            if self.civ.dead_num >= len(self.civ.list) - 10:
                print("Sim Finish.Because of autocracy.")
                break
            battle_num += self.civ.battle_num
            while 1:
                new_event = self.events.get()
                if not new_event:
                    break
                if new_event.time > year:
                    self.events.put(new_event)
                    break
                self.events.handle(self.civ, self.gl, new_event)
            if year % 100 == 0:
                print("YEAR:", year)
                print("dead num:", self.civ.dead_num)
                print("battle num:", self.civ.battle_num, '\n')
                if battle_num == 0 and year != 0:
                    # 百年无一战事
                    print("Sim Finish.Because of piece.")
                    break
                battle_num = 0
            self.log_info(year)
        print("Sim end.\nDrawing...")
        # self._end_plot()

    def _end_plot(self):
        """
        存储可视化文件
        """
        helper.analysis()
        for name in ('mlst', 'lbob', 'oclnum', 'dtrange'):
            helper.draw_gif(
                lambda x: helper.plot_info(x, plot_type=name, shrink=100),
                self.gifpath+str(name)+'.gif'
            )

    def _log(self, data, path, log_mask=None, columns=None, is_civ=False, is_galaxy=False):
        df = pd.DataFrame(data, columns=columns)
        if log_mask:
            remove = df.columns[np.logical_not(log_mask)]
        if is_civ:
            df["pd"] = list(map(self.civ.get_sum_pd, df["ocl"]))
            df["mlst"] = list(map(self.civ.get_sum_mlst, df["ocl"]))
            df["know_num"] = list(map(lambda x: len(x) if x else 0, df["ctknown"]))
            df["allies_num"] = list(map(lambda x: len(x) if x else 0, df["allies"]))
            df["enemies_num"] = list(map(lambda x: len(x) if x else 0, df["enemies"]))
        elif is_galaxy:
            df["belong_to"] = list(map(
                lambda x: x[NAMES.name] if x is not None else None,
                data[:, NAMES.belong]
            )
            )
        if log_mask:
            df.drop(remove, axis=1, inplace=True)
        df.to_csv(path, index=False)

    def log_info(self, year):
        """
        记录日志信息

        :param year: 当前公元多少年
        :return:
        """
        if year == 0:
            self._log(self.gl.pos, os.path.join(self.logpath, "pos.csv"))

        # 记录文明信息
        self._log(self.civ.list, os.path.join(self.logpath, "%d_civ.csv" % (year,)),
                  log_mask=LOG_MASK,
                  columns=["name", "mlb", "btTch", "dfTch", "enemies", "allies", "ctknown", "ocl", "attr"],
                  is_civ=True
                  )
        # 记录星系信息
        self._log(self.gl.list, os.path.join(self.logpath, "%d_galaxy.csv" % (year,)),
                  log_mask=G_LOG_MASK,
                  columns=["lbf", "mlb", "lbob", "mlst", "dtrange", "belong", "pd"],
                  is_galaxy=True
                  )


if __name__ == "__main__":
    universe = Universe(logname='_s4_100', civ_size=100)
    universe.start_sim()
    universe = Universe(logname='_s4_300', civ_size=300)
    universe.start_sim()
    universe = Universe(logname='_s4_1000', civ_size=1000)
    universe.start_sim()

