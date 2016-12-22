# -*- coding: utf-8 -*-
"""

@author: Rem
@contact: remch183@outlook.com
@site:
@software: PyCharm Community Edition
@file: civ.py
@time: 2016/12/14 19:37


TODO:

    Galaxy
    - 星系资源情况更新

    Civilization
    - 补助刷新
    - 饥荒判断
    - 技术发展

    EventPool
    - 时间循环（该处调用应当与同步更新相结合）

    汇总数据存储
"""
__author__ = 'Rem'

import numpy as np
from queue import PriorityQueue


# ============
# 文明变量
# ["name", "mlb",  "pd", "btTch", "dfTch", "enemies", "allies", "ctknown", "ocl",    "attr"]
# 分别对应为
# 名字      技术水平  产值   进攻能力  防守能力   敌对国      联盟国     已知地区    占有地区   天然属性(激进、外交、保守)
# ----------
# 星系变量
# ["lbf", "lbob", "mlst", "dtrange", "belong"]
# 分别对应为
#  劳动力   自然资源  兵力    侦测范围     归属国
# ============
# 激进 外交 保守 = ATT DIP CON

class NAMES:
    name, mlb, pd, btTch, dfTch, enemies, allies, ctknown, ocl, attr = \
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    lbf, mlb, lbob, mlst, dtrange, belong = 0, 1, 2, 3, 4, 5

    ATT, DIP, CON = 0, 1, 2


class InitialValue:
    mlst = 10
    lbf = 1
    mlb = 1
    pd = 10
    btTch = 100
    dfTch = 100
    detect_speed = 10

    @staticmethod
    def rand():
        v = np.random.rand()
        if v > 2 / 3:
            return 0
        if v > 1 / 3:
            return 1
        return 2


class Galaxy:
    def __init__(self, size=1000):
        # None为归属国
        self.list = np.array([0, 0, 0, 0, 0, None] * size)
        # 初始自然资源设置
        self.list.T[NAMES.lbob] = np.random.normal(50, 10, (1, size))
        # 设置位置: x\bl [0, 1000]  y\bl [0, 1000]
        self.pos = np.random.rand(size, size) * 1000
        self.dist = None
        self.argsort_dist = None
        self._caldist()

    def _caldist(self):
        """计算两星系之间的距离"""
        for i, p in enumerate(self.pos):
            for j, q in enumerate(self.pos):
                self.dist[i, j] = ((p - q) ** 2).sum()
        self.argsort_dist = self.dist.argsort()


class Event:
    """
    事件类

    对于探测事件TY_FOUND：
        - who 表示出发寻找的文明的name
        - whom 表示找到的星系的下标
        - start 出发找的地点
        - range 找到之后应当更新的range
    对于相遇事件TY_MET:
        - who  本文明
        - whom 被发现文明

    """
    TY_FOUND = 0
    TY_MET = 1

    def __init__(self, time, typ, who, whom, **kwargs):
        self.time = time
        self.typ = typ
        self.who = who
        self.whom = whom
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def __lt__(self, other):
        if not isinstance(other, Event):
            return True
        if self.time < other.time:
            return True
        return False


class EventPool:
    def __init__(self):
        self.queue = PriorityQueue()

    def put(self, event):
        self.queue.put(event)

    def get(self):
        return self.queue.get(False)

    def handle(self, civ, galaxy, event):
        assert isinstance(civ, Civilization) and isinstance(galaxy, Galaxy)
        if event.typ == event.TY_FOUND:
            civ.handle_found(event)
        if event.typ == event.TY_MET:
            civ.handle_met(event.who, event.whom)


class Civilization:
    def __init__(self, galaxy, eventpool, size=300):
        assert isinstance(galaxy, Galaxy) and isinstance(eventpool, EventPool)
        # 定义文明列表
        # 其中死亡的文明的Name.ocl is None(不会从列表中删除)
        self.list = np.array([0, InitialValue.mlb, InitialValue.pd,
                              InitialValue.btTch, InitialValue.dfTch,
                              None, None, None, None, None] * size)
        for i, row in enumerate(self.list):
            row[0], row[5], row[6], row[7], row[8], = i, [], [], [], [i]
            # 决定天然属性
            row[9] = InitialValue.rand()
            # 设定居住地
            galaxy.list[i, NAMES.belong] = i
            galaxy.list[i, NAMES.lbf] = InitialValue.lbf
            galaxy.list[i, NAMES.mlst] = InitialValue.mlst

        self.galaxy = galaxy
        self.eventpool = eventpool
        for i in range(len(list)):
            self.detect(i)

    def is_dead(self, who):
        x = self.list[who, NAMES.ocl]
        return x is None or len(x) == 0

    def handle_found(self, event):
        who_index, who = event.who, self.list[event.who]
        # 如果该国已死或者对方已经被发现则返回
        if self.is_dead(who):
            return
        if event.whom in who[NAMES.ctknown]:
            return
        # 将发现星系加入已知星系
        who[NAMES.ctknown].append(event.whom)
        # 更新对应星系范围
        self.galaxy.list[event.start, NAMES.dtrange] = event.range
        # 计算下一次侦查到的对象并插入事件队列池
        self.detect(event.start)

        # 如果发现的是无人区
        belong = self.galaxy.list[event.whom, NAMES.belong]
        if belong is None:
            # 签署为自己的地盘, 并赋予生产技术
            self.galaxy.list[event.whom, NAMES.belong] = who
            self.galaxy.list[event.whom, NAMES.mlb] = who[NAMES.mlb]
        else:
            # 如果发现的不是无人区
            self.handle_met(who_index, belong)

    def handle_met(self, one, another):
        # 处理前者碰到后者事件
        atype, btype = self.list[one, NAMES.attr], self.list[another, NAMES.attr]
        if atype == NAMES.DIP:
            # 若为外交型文明，对方也立即知道自己的存在
            # 因为现在探测速度都一致，所以此处意义不大(而且没有加入判断会出现尴尬的死环)
            # self.eventpool.put(Event(0, Event.TY_MET, another, one))
            if btype == NAMES.DIP:
                self._be_allies(one, another)
            if btype == NAMES.ATT:
                # 外交失败兵力受损10%~40%
                for gl in self.list[one, NAMES.ocl]:
                    self.galaxy.list[gl, NAMES.mlst] *= \
                        1 - np.random.rand() * 0.3 - 0.1
                self._be_enemies(one, another)
        if atype == NAMES.ATT:
            self._be_enemies(one, another)

    def _be_allies(self, a, b):
        """ as function name said"""
        if b not in self.list[a, NAMES.allies]:
            self.list[a, NAMES.allies].append(b)
        if a not in self.list[b, NAMES.allies]:
            self.list[b, NAMES.allies].append(a)

    def _be_enemies(self, a, b):
        """ as function name said"""
        if b not in self.list[a, NAMES.enemies]:
            self.list[a, NAMES.enemies].append(b)
        if a not in self.list[b, NAMES.enemies]:
            self.list[b, NAMES.enemies].append(a)

    def detect(self, index):
        """
        开始进行探测，并将探测到事件添加到事件池当中
        :param index: 开始探测的星系坐标
        :return: None
        """
        galaxy = self.galaxy
        args = galaxy.argsort_dist[index]
        owner = galaxy.list[index, NAMES.belong]
        closest = args[args != index][0]
        args[closest] = index
        # 两星系距离
        distG = galaxy.dist[closest, index]
        # 需要探测距离
        dist = distG - galaxy.list[index, NAMES.dtrange]
        time = dist // InitialValue.detect_speed
        if dist <= 0:
            raise ValueError("Dist <= 0")
        self.eventpool.put(
            Event(time, Event.TY_FOUND, owner, closest, start=index, range=distG)
        )
        return

    def refresh_all(self):
        """
        刷新所有状态：
        - 战斗
        - 支援
        - 补助
        :return:
        """
        self._refresh_aid()
        self._refresh_battle()
        self._refresh_starve()
        self._refresh_supply()

    def _refresh_battle(self):
        for one in self.list:
            for ano in one[NAMES.enemies]:
                # 避免计算两次
                if one > ano: continue
                abt, bbt = self.list[one, NAMES.btTch], self.list[ano, NAMES.btTch]
                adf, bdf = self.list[one, NAMES.dfTch], self.list[ano, NAMES.dfTch]
                # 随机地区减小兵力（如果是地区对地区则太过麻烦了）
                a = self.list[one, NAMES.ocl]
                b = self.list[ano, NAMES.ocl]
                a, b = int(np.random.rand() * len(a)), int(np.random.rand() * len(b))
                ga, gb = self.galaxy.list[a], self.galaxy.list[b]
                gb[NAMES.mlst], ga[NAMES.mlst] = \
                    gb[NAMES.mlst] - bbt / adf * ga[NAMES.mlst], \
                    ga[NAMES.mlst] - abt / bdf * gb[NAMES.mlst]

    def _refresh_aid(self):
        for one in range(len(self.galaxy)):
            if self.galaxy[one, NAMES.belong] is None:
                continue
            mlst = self.galaxy[one, NAMES.mlst]
            # 兵力大于二十则不需要支援
            if mlst >= 20:
                continue
            belong = self.galaxy[one, NAMES.belong]
            for ano in self.list[belong, NAMES.ocl]:
                ano_mlst = self.galaxy[ano, NAMES.mlst]
                if ano_mlst <= mlst: continue
                aid = (ano_mlst - mlst) / 2
                # 简单的平分法
                # TODO： 添加函数，对大兵力进行限制
                ano_mlst, mlst = mlst + aid, mlst + aid
                if mlst >= 20:
                    break

    # TODO:讨要救济金
    def _refresh_supply(self):
        pass

    # TODO:饥荒状态更新判断
    def _refresh_starve(self):
        pass

