# -*- coding: utf-8 -*-
"""
该文件可以帮助你分析与可视化log当中的信息。

通过sim(name)你可以指定log文件名，并将相应log存储到该文件中。
通过analysis(name)你可以指定log文件名进行分析，否则会替你选择默认log文件


@author: Rem
@contack: remch183@outlook.com
@time: 2017/1/3 17:06
"""
import numpy as np
import os
import pandas as pd
import seaborn as sns
from PIL import Image
from images2gif import writeGif
import imageio

try:
    from .universe import NAMES
    from simuniverse import universe
except ImportError:
    from universe import NAMES
    import universe
except SystemError:
    from universe import NAMES
    import universe

__author__ = "Rem"

root_log_path = '../log/'
log_path = '../log/log/'


def sim(name):
    """
    设定log名进行仿真

    :param name: log名
    :return: None
    """
    path = (root_log_path + 'log' + str(name) + '/')
    if not os.path.exists(path):
        os.mkdir(path)
    u = universe.Universe(logpath=path)
    u.start_sim()


# Basic Operation Definition


def get_end_year():
    """
    获取仿真的时间总长度
    :return: 仿真的时间总长度
    """

    def _get_title(x):
        ans = x.split('_')[0]
        if ans.isdecimal():
            return int(ans)
        return 0

    return max(list(map(_get_title, os.listdir(log_path))))


def get_year_df(year, is_civ=True):
    """
    获取指定时间的文明或星系变量

    :param year: 时间
    :param is_civ: 是否返回文明变量，否则返回星系变量
    :return: 相应变量
    """
    if is_civ:
        df = pd.read_csv('%s%d_%s.csv' % (log_path, year, 'civ'), index_col=False)
        df['oclnum'] = df.ocl.map(lambda x: len(eval(x)) if x is not np.NAN else 0)
    else:
        path = '%s%d_%s.csv' % (log_path, year, 'galaxy')
        df = pd.read_csv(path, index_col=False)
    return df


def get_pos():
    """
    获取星系坐标
    :return: 星系坐标
    """
    pos = pd.read_csv(log_path + 'pos.csv')
    pos.columns = ["x", "y"]
    return pos


class g:
    end_year = None
    gl_begin = None
    gl_mid = None
    gl_end = None
    pos = None
    df_begin = None
    df_end = None
    log_path = None


def analysis(name=None):
    """
    设置需要分析的log的名字

    :param name: 选择的log的名字
    :return: None
    """
    global log_path
    if name is None:
        log_path = root_log_path + 'log/'
    else:
        log_path = root_log_path + 'log' + str(name) + '/'

    g.log_path = log_path
    print(log_path)
    g.end_year = get_end_year()
    g.gl_begin = get_year_df(0, is_civ=False)
    g.gl_mid = get_year_df(g.end_year // 2, is_civ=False)
    g.gl_end = get_year_df(g.end_year, is_civ=False)
    g.pos = get_pos()
    g.df_begin = get_year_df(0)
    g.df_end = get_year_df(g.end_year)


def print_attr_info():
    """
    打印输出三种不同类型文明的起始分布数量和最终分布数量
    """
    print("模拟开始时，三类属性分布为:{0:结盟型,1:好战性,2:保守型}")
    print(g.df_begin.attr.value_counts())
    print("模拟结束时，三类属性分布为：{0:结盟型,1:好战性,2:保守型}")
    print(g.df_end[g.df_end.mlb != 0].attr.value_counts())


# =======================================
#
# Visualization
#
# =======================================

def get_type_color(galaxy, civ):
    """
    获取每一个星系所应当对应的颜色

    :param galaxy: 星系变量
    :param civ: 文明变量
    :return: 颜色向量
    """
    colors = []
    alpha = []
    cls = [[20, 220, 20], [220, 20, 20], [20, 20, 220], [220, 220, 220]]
    for row in galaxy.belong_to:
        if row is None or row is np.nan:
            colors.append(cls[3])
            alpha.append(0.2)
            continue
        try:
            colors.append(cls[civ.attr[row]])
            alpha.append(1)
        except TypeError:
            alpha.append(0.2)
            colors.append(cls[3])
    return np.array(colors) / 256


def plot_map(gl, civ, mask=None, plotsize=20, alpha=0.7):
    """
    通过星系和文明以及指定的mask和大小绘制散点图

    :param gl: 星系变量
    :param civ: 文明变量
    :param mask: 掩码
    :param plotsize: 绘制大小
    :param alpha: 不透明度
    :return: 绘制好的散点图
    """
    # print("红色为战斗文明，绿色为交好文明， 蓝色为保守文明。灰色为未被占领地")
    if mask is None:
        colors = get_type_color(gl, civ)
        return sns.plt.scatter(g.pos.x, g.pos.y, c=colors, alpha=alpha, s=plotsize + 4)
    else:
        colors = get_type_color(gl[mask], civ)
        return sns.plt.scatter(g.pos[mask].x, g.pos[mask].y, c=colors, alpha=alpha, s=plotsize + 4)


def plot_range(gl, civ, alive=True):
    """
    输出侦测范围散点图

    :param gl: 星系变量
    :param civ: 文明变量
    :param alive: 是否只输出存活者
    :return: 星系的探测范围散点图
    """
    mask = np.logical_not(gl.belong_to.isnull()) if alive else None
    return plot_map(gl, civ, mask=mask, plotsize=gl.dtrange, alpha=0.3)


def plot_time_info(year, ret=False, alive=True):
    """
    选择时间显示探测散点图并返回文明和星系

    :param year: 选择一年
    :param ret: 是否需要返回值
    :param alive: 是否只输出存活者
    :return: 文明,星系
    """
    sns.plt.xlim(-1000, 11000)
    sns.plt.ylim(-1000, 11000)
    civ = pd.read_csv(log_path + '%d_civ.csv' % (year,), index_col=False)
    gl = pd.read_csv(log_path + '%d_galaxy.csv' % (year,), index_col=False)
    plot_range(gl, civ, alive=alive)
    if ret:
        return civ, gl


def plot_info(year, plot_type='oclnum', shrink=1):
    """
    指定时间和类型，输出相应的散点图

    :param year: 指定时间
    :param plot_type:
        可选择['oclnum', 'mlst','lbob']，'oclnum'表示输出占有星系数量的散点图
        'mlst'表示输出兵力值的散点图,'lbob'表示自然资源剩余量
    :param shrink:缩小比例
    :return: 散点图变量
    """
    civ = get_year_df(year, is_civ=True)
    gl = get_year_df(year, is_civ=False)

    def get(name):
        def f(x):
            try:
                return civ.ix[int(x), name]
            except:
                return 0

        return f

    if plot_type == 'oclnum':
        sns.plt.xlim(-1000, 11000)
        sns.plt.ylim(-1000, 11000)
        capitals = civ.ocl.map(lambda x: eval(x)[0] if x is not np.NAN and x is not None else 0)
        size = gl.belong_to.map(get('oclnum'))
        mask = np.bool8(np.zeros_like(size))
        for i in capitals:
            mask[i] = True
        sns.plt.title(str(year) + " Occupied Galaxies Number")
        # print('每一个点为一个文明，范围大小表示星系数量')
        return plot_map(gl, civ, mask=mask, plotsize=size / shrink)
    elif 'ml' in plot_type:
        sns.plt.xlim(-1000, 11000)
        sns.plt.ylim(-1000, 11000)
        capitals = civ.ocl.map(lambda x: eval(x)[0] if x is not np.NAN and x is not None else 0)
        size = gl.belong_to.map(get('mlst'))
        mask = np.bool8(np.zeros_like(size))
        for i in capitals:
            mask[i] = True
        sns.plt.title("Year " + str(year) + ": Military Strength")
        # print('每一个点为一个文明，范围大小表示军事力量')
        return plot_map(gl, civ, mask=mask, plotsize=size / shrink)
    elif 'lbob' in plot_type:
        sns.plt.xlim(0, 100)
        sns.plt.ylim(0, 0.6)
        sns.plt.title("Year " + str(year) + ": Natural Resources")
        return sns.distplot(get_year_df(year, False).lbob)
    elif 'range' in plot_type:
        sns.plt.xlim(-1000, 11000)
        sns.plt.ylim(-1000, 11000)
        sns.plt.title("Year " + str(year) + ": Detect Range")
        return plot_range(gl, civ)


def p_lbob(year):
    return plot_info(year, 'lbob')


def plot_get_files(func):
    files = []
    for i in range(0, g.end_year, 10):
        name = str(i) + '_.png'
        fig = sns.plt.figure()
        img = func(i).get_figure()
        img.savefig(log_path + name)
        files.append(log_path + name)
        sns.plt.close(fig)
        del fig, img
    return files


def clear_files(files):
    for file in files:
        os.remove(file)


def draw_gif(func, filepath):
    files = plot_get_files(func)
    imgs = []
    for filename in files:
        imgs.append(imageio.imread(filename))
    imageio.mimwrite(filepath, imgs)
    # with imageio.get_writer(filepath, mode='I') as writer:
    #     for filename in files:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)


if __name__ == '__main__':
    analysis()
    for name in ('mlst', 'lbob', 'oclnum', 'dtrange'):
        draw_gif(
            lambda x: plot_info(x, plot_type=name, shrink=100),
            '../log/log/gif/' + str(name) + '.gif'
        )
