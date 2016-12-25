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
import numpy as np
from queue import PriorityQueue
from time import time as systime
import random

__author__ = 'Rem'

random.seed(int(systime()))
np.random.seed(int(systime()))

__all__ = ["NAMES", "Event", "EventPool", "Civilization", "Galaxy"]


# ============
# 文明变量
# ["name", "mlb",  "pd", "btTch", "dfTch", "enemies", "allies", "ctknown", "ocl",    "attr"]
# 分别对应为
# 名字      技术水平  产值   进攻能力  防守能力   敌对国      联盟国     已知地区    占有地区   天然属性(激进、外交、保守)
# ----------
# 星系变量
# ["lbf", "mlb", "lbob",  "mlst", "dtrange", "belong", "pd"]
# 分别对应为
#  劳动力  技术水平  自然资源   兵力    侦测范围     归属国     产值
# ============
# 激进 外交 保守 = ATT DIP CON

class NAMES:
    name, mlb, btTch, dfTch, enemies, allies, ctknown, ocl, attr = \
        0, 1, 2, 3, 4, 5, 6, 7, 8
    lbf, mlb, lbob, mlst, dtrange, belong, pd = 0, 1, 2, 3, 4, 5, 6

    DIP, ATT, CON = 0, 1, 2


class InitialValue:
    mlst = 10
    lbf = 1
    mlb = 1
    pd = 10
    btTch = 1
    dfTch = 1000
    detect_speed = 1e2

    @staticmethod
    def rand():
        v = np.random.rand()
        if v > 2 / 3:
            return 0
        if v > 1 / 3:
            return 1
        return 2

    # \delta pd = pd_rate * (lbf*\delta lbob) ** (1/2) * mlb
    pd_rate = 1
    # del_lbob 每轮lbob消耗
    del_lbob = 0.05
    # del_mlb 每轮固定增加产量
    del_mlb = 0.03

    # DEBUG:
    # milli_resource  每轮为发展军队所用资源
    @staticmethod
    def milli_resource(pd, milli):
        return np.array((pd * 0.0, (100 - milli) * 0.05)).min(0)

    # mlst = (delta pd)**eta_milli_out
    eta_milli_out = 0.7

    # DEBUG:
    # tch_resource  每轮为发展军科所用资源
    @staticmethod
    def tch_resource(pd):
        return pd * 0.0


class Galaxy:
    def __init__(self, size=1000):
        # None为归属国
        self.list = np.array([[.0, .0, .0, .0, .0, None, 0] for _ in range(size)], dtype=np.object)
        # 初始自然资源设置
        self.list.T[NAMES.lbob] = np.random.normal(50, 10, (1, size))
        # 设置位置: x\bl [0, 1000]  y\bl [0, 1000]
        self.pos = np.random.rand(size, 2) * 1000
        self.dist = None
        self.argsort_dist = None
        self._caldist()

    def _caldist(self):
        """计算两星系之间的距离"""
        self.dist = np.ones((len(self.list), len(self.list)))
        for i, p in enumerate(self.pos):
            for j, q in enumerate(self.pos):
                self.dist[i, j] = ((p - q) ** 2).sum()**0.5
        self.argsort_dist = self.dist.argsort()

    def resource_refresh(self):
        """
        刷新各类资源，以及技能增长
        """

        # 选择有人区
        has_men = self.list[self.list[:, NAMES.mlb] > 1e-4]
        # 消耗自然资源
        has_men[:, NAMES.lbob] -= InitialValue.del_lbob
        cost_product = (has_men[:, NAMES.lbf] * InitialValue.del_lbob) ** 0.5
        mlb = has_men[:, NAMES.mlb]
        # 产值增加
        has_men[:, NAMES.pd] += InitialValue.pd_rate * cost_product * mlb
        # 产率增加
        has_men[:, NAMES.mlb] += InitialValue.del_mlb

        # 军科发展
        for one in has_men:
            belong = one[NAMES.belong]
            # TODO: DEBUG, 本处本不应该有下面的这种情况，因为当belong is None时
            # one[NAMES.mlb] > 1e-4 应当是不成立的。
            # 不知道代码的哪个部分违反了这个规则
            if belong is None:
                continue
            resource = InitialValue.tch_resource(one[NAMES.pd])
            type = belong[NAMES.attr]
            belong[NAMES.btTch] += [0, 1, 0][type] * resource
            belong[NAMES.dfTch] += [1, 0, 1][type] * resource

        # 兵力发展
        need_millitary = has_men[has_men[:, NAMES.mlst] < 99]
        use_resource = \
            InitialValue.milli_resource(
                need_millitary[:, NAMES.pd], need_millitary[:, NAMES.mlst]
            )
        need_millitary[:, NAMES.mlst] += use_resource ** InitialValue.eta_milli_out
        # TODO: 按道理来说不需要下面的两行。需要探究下numpy.array什么时候会发生复制，什么时候不会
        has_men[has_men[:, NAMES.mlst] < 99] = need_millitary
        self.list[self.list[:, NAMES.mlb] > 1e-4] = has_men


class Event:
    """
    事件类

    对于探测事件TY_FOUND：
        - who 表示出发寻找的文明的行
        - whom 表示找到的星系的下标
        - start 出发找的地点
        - range 找到之后应当更新的range
    对于相遇事件TY_MET:
        - who  本文明
        - whom 被发现文明

    """
    TY_FOUND = 0
    TY_MET = 1
    TYPES = {0: "FOUND", 1: "MET"}

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

    def __repr__(self):
        return "<Event time:%d type:%s who:%s whom:%s>" % \
               (self.time, self.TYPES[self.typ], str(self.who[NAMES.name]), str(self.whom))


class EventPool:
    def __init__(self):
        self.queue = PriorityQueue()

    def put(self, event):
        self.queue.put(event)

    def get(self):
        if self.queue.empty():
            return None
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
        self.list = np.array([[0, InitialValue.mlb,
                               InitialValue.btTch, InitialValue.dfTch,
                               [], [], [], [], []] for _ in range(size)], dtype=np.object)
        for i, row in enumerate(self.list):
            row[NAMES.name], row[NAMES.enemies], row[NAMES.allies], row[NAMES.ctknown], row[NAMES.ocl] \
                = i, [], [], [], [i]
            # 决定天然属性
            row[NAMES.attr] = InitialValue.rand()
            # 设定居住地
            galaxy.list[i, NAMES.belong] = self.list[i]
            galaxy.list[i, NAMES.lbf] = InitialValue.lbf
            galaxy.list[i, NAMES.mlst] = InitialValue.mlst
            galaxy.list[i, NAMES.mlb] = InitialValue.mlb
            galaxy.list[i, NAMES.pd] = InitialValue.pd

        self.galaxy = galaxy
        self.eventpool = eventpool
        self.time = 0
        self.dead_num = 0
        self.battle_num = 0
        for i in range(len(self.list)):
            self.detect(i)

    def is_dead(self, who):
        try:
            x = who[NAMES.ocl]
        except TypeError:
            x = self.list[who, NAMES.ocl]
        return x is None or len(x) == 0

    def handle_found(self, event):
        who_index, who = event.who[NAMES.name], event.who
        # 如果该国已死则返回
        if self.is_dead(who):
            return
        # 如果另一国家已经被发现,则发起下一个搜索之后返回
        if event.whom in who[NAMES.ctknown]:
            self.detect(event.start)
            return
        # 将发现星系加入已知星系
        who[NAMES.ctknown].append(event.whom)
        # 更新对应星系范围
        self.galaxy.list[event.start, NAMES.dtrange] = event.range
        # 计算下一次侦查到的对象并插入事件队列池
        self.detect(event.start)

        belong = self.galaxy.list[event.whom, NAMES.belong]
        try:
            # 如果发现的不是无人区
            self.handle_met(who_index, belong[NAMES.name])
        except TypeError:
            # 如果发现的是无人区
            # 签署为自己的地盘, 并赋予生产技术
            self.galaxy.list[event.whom, NAMES.belong] = who
            self.galaxy.list[event.whom, NAMES.mlb] = who[NAMES.mlb]

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
        # 下面这句只是调试方便，将来可以放别的地方
        # 其中默认情况下args[0]为它自身
        args[args == index] = -1
        owner = galaxy.list[index][NAMES.belong]
        closest = args[args >= 0]
        try:
            closest = closest[0]
        except IndexError:
            # 已经发现所有其他星系，直接返回
            return
        args[args == closest] = -1
        # 两星系距离
        distG = galaxy.dist[closest, index]
        # 需要探测距离
        dist = distG - galaxy.list[index, NAMES.dtrange]
        time = dist // InitialValue.detect_speed + self.time + 1
        # ERROR: 会出现下面的错误，按常理不应有这种情况
        if dist <= 0 or time <= self.time + 1:
            # raise ValueError("Dist <= 0")
            time = self.time + 1
        event = Event(time, Event.TY_FOUND, owner, closest, start=index, range=distG)
        self.eventpool.put(event)
        # if dist > 1000:
        #     print("detect put", event, "DIST:", dist, "delta_time", time-self.time-1)
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
        self._refresh_dead()
        self.galaxy.resource_refresh()

    def get_sum_mlst(self, galaxy_list):
        mlst = 0
        if galaxy_list is None or not len(galaxy_list):
            return 0
        for i in galaxy_list:
            mlst += self.galaxy.list[i, NAMES.mlst]
        return mlst

    def get_sum_pd(self, galaxy_list):
        pd = 0
        if galaxy_list is None or not len(galaxy_list):
            return 0
        for i in galaxy_list:
            pd += self.galaxy.list[i, NAMES.pd]
        return pd

    def _destroy_civ(self, index):
        self.dead_num += 1
        self.list[index, NAMES.mlb] = 0
        self.list[index, NAMES.enemies] = None
        self.list[index, NAMES.allies] = None
        self.list[index, NAMES.ctknown] = None
        for g in self.list[index, NAMES.ocl]:
            self.galaxy.list[g, NAMES.belong] = None
            self.galaxy.list[g, NAMES.mlb] = 0
            self.galaxy.list[g, NAMES.mlst] = 0
        self.list[index, NAMES.ocl] = None

    def _refresh_dead(self):
        """
        判断文明死亡
        :return:
        """
        for index, one in enumerate(self.list):
            if one[NAMES.mlb] == 0: continue
            # TODO: 更加合理的死亡判定
            if self.get_sum_pd(one[NAMES.ocl]) < 0.1:
                self._destroy_civ(index)
            if self.get_sum_mlst(one[NAMES.ocl]) < 0.002:
                self._destroy_civ(index)

    def _refresh_battle(self):
        self.battle_num = 0
        for one in self.list:
            if not one[NAMES.enemies]:
                continue
            for ano in one[NAMES.enemies]:
                if not self.list[ano, NAMES.enemies]:
                    # 敌人已死，移出列表
                    one[NAMES.enemies].remove(ano)
                    continue
                # # 避免计算两次
                # 先不管了 = =
                # if one[NAMES.name] > ano: continue
                ano = self.list[ano]
                self.battle_num += 1
                abt, bbt = one[NAMES.btTch], ano[NAMES.btTch]
                adf, bdf = one[NAMES.dfTch], ano[NAMES.dfTch]
                # TODO: 改成对地区的战斗
                # 随机地区减小兵力（如果是地区对地区则太过麻烦了）
                a = one[NAMES.ocl]
                b = ano[NAMES.ocl]
                a, b = random.choice(a), random.choice(b)
                # a, b 为对战的星系坐标
                ga, gb = self.galaxy.list[a], self.galaxy.list[b]
                # 计算兵力损耗
                gbf, gaf = gb[NAMES.mlst], ga[NAMES.mlst]
                gb[NAMES.mlst], ga[NAMES.mlst] = gbf - (abt / bdf) * gaf, gaf - (bbt / adf) * gbf

    def _refresh_aid(self):
        # TODO: 添加盟国支援
        for one in range(len(self.galaxy.list)):
            # 当该星系无人时到下一轮
            if self.galaxy.list[one, NAMES.belong] is None \
               or self.galaxy.list[one, NAMES.mlb] < 0.3:
                continue
            belong = self.galaxy.list[one, NAMES.belong]
            # 这是一处BUG,本不应该有这种情况
            if belong[NAMES.ocl] is None:
                self.galaxy.list[one, NAMES.belong] = None
                continue
            mlst = self.galaxy.list[one, NAMES.mlst]
            # 兵力大于二十则不需要支援
            if mlst >= 20:
                continue
            for ano in belong[NAMES.ocl]:
                ano_mlst = self.galaxy.list[ano, NAMES.mlst]
                if ano_mlst <= mlst:
                    # 过滤掉自己以及兵力更少的星系
                    continue
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
