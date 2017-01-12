# -*- coding: utf-8 -*-
"""


@author: Rem
@contact: remch183@outlook.com
@site:
@software: PyCharm Community Edition
@file: civ.py
@time: 2016/12/14 19:37
"""
import numpy as np
from queue import PriorityQueue
from time import time as systime
import random

__author__ = 'Rem'
__all__ = ["NAMES", "Event", "EventPool", "Civilization", "Galaxy"]

# Set random seed
random.seed(int(systime()))
np.random.seed(int(systime()))


# ============
# 文明变量
# ["name", "mlb",  "btTch", "dfTch", "enemies", "allies", "ctknown", "ocl",    "attr"]
# 分别对应为
# 1名字    2技术水平 3进攻能力 4防守能力  5敌对国     6联盟国    7已知地区   8占有地区  9天然属性(激进、外交、保守)
# ----------
# 星系变量
# ["lbf", "mlb", "lbob",  "mlst", "dtrange", "belong", "pd"]
# 分别对应为
#  1劳动力2技术水平 3自然资源 4兵力   5侦测范围    6归属国    7产值
# ============
# 激进 外交 保守 = ATT DIP CON

class NAMES:
    name, mlb, btTch, dfTch, enemies, allies, ctknown, ocl, attr = \
        0, 1, 2, 3, 4, 5, 6, 7, 8
    lbf, mlb, lbob, mlst, dtrange, belong, pd = \
        0, 1, 2, 3, 4, 5, 6

    DIP, ATT, CON = 0, 1, 2


class InitialValue:
    slow_rate = 1e4
    mlst = 10
    lbf = 1
    mlb = 1
    pd = 10
    btTch = 1
    dfTch = 1
    # TODO SET 探测速度设置
    detect_speed = 100

    attr_distributes = [1, 1, 1]

    @staticmethod
    def rand():
        bound = InitialValue.attr_distributes
        s = sum(bound)
        bound = [i / s for index, i in enumerate(bound)]
        for i in range(1, 3):
            bound[i] += bound[i - 1]
        v = np.random.rand()
        if v > bound[1]:
            return 2
        if v > bound[0]:
            return 1
        return 0

    # \delta pd = pd_rate * (lbf*\delta lbob) ** (1/2) * mlb
    pd_rate = 1
    # TODO SET 资源消耗设置
    # del_lbob 每轮lbob消耗
    del_lbob = 0.08
    # del_mlb 每轮固定增加产量
    del_mlb = 0.03
    # 地图大小
    map_range = 10000

    # DEBUG:
    # milli_resource  每轮为发展军队所用资源
    @staticmethod
    def milli_resource(pd, milli):
        return np.array((pd * 0.1, (100 - milli) * 0.05)).min(0)

    # mlst = (delta pd)**eta_milli_out
    eta_milli_out = 0.7

    # DEBUG:
    # tch_resource  每轮为发展军科所用资源
    @staticmethod
    def tch_resource(pd):
        return pd * 0.1


class Galaxy:
    def __init__(self, size=1000):
        # None为归属国
        self.list = np.array([[.0, .0, .0, .0, .0, None, 0] for _ in range(size)], dtype=np.object)
        # 初始自然资源设置
        self.list.T[NAMES.lbob] = np.random.normal(50, 10, (1, size))
        # 设置位置: x\bl [0, 1000]  y\bl [0, 1000]
        self.pos = np.random.rand(size, 2) * InitialValue.map_range
        self.dist = None
        self.argsort_dist = None
        self.civ = None
        self._caldist()

    def _caldist(self):
        """计算两星系之间的距离"""
        self.dist = np.ones((len(self.list), len(self.list)))
        for i, p in enumerate(self.pos):
            for j, q in enumerate(self.pos):
                self.dist[i, j] = ((p - q) ** 2).sum() ** 0.5
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
            # FIXME 本处本不应该有下面的这种情况，因为当belong is None时
            # one[NAMES.mlb] > 1e-4 应当是不成立的。
            # 不知道代码的哪个部分违反了这个规则
            if belong is None:
                continue
            resource = InitialValue.tch_resource(one[NAMES.pd])
            # TODO SET 战争控制
            # 根据自身类型发展对应科技
            type = belong[NAMES.attr]
            belong[NAMES.btTch] += [0.5, 2.6, 0.5][type] * resource
            belong[NAMES.dfTch] += [0.5, 0.4, 0.5][type] * resource

        # 兵力发展
        need_millitary = has_men[has_men[:, NAMES.mlst] < 99]
        use_resource = \
            InitialValue.milli_resource(
                need_millitary[:, NAMES.pd], need_millitary[:, NAMES.mlst]
            )
        need_millitary[:, NAMES.mlst] += use_resource ** InitialValue.eta_milli_out
        # FIXME: 按道理来说不需要下面的两行。需要探究下numpy.array什么时候会发生复制，什么时候不会
        has_men[has_men[:, NAMES.mlst] < 99] = need_millitary
        self.list[self.list[:, NAMES.mlb] > 1e-4] = has_men

        # 劳动力增加
        def sigmoid(x):
            x = np.array(x, dtype=np.float32)
            return 1/(1 + np.exp(-x))
        used = (1 - sigmoid(has_men[:, NAMES.lbf])) * 0.8
        use_pd = used * 10
        pd = has_men[:, NAMES.pd]
        mask = (use_pd > (pd - 5))
        used[mask] = (pd - 5)[mask]                 # 若产值充足则发展人口，否则下调人口
        has_men[:, NAMES.lbf] += used
        has_men[:, NAMES.pd] -= used
        assert isinstance(self.civ, Civilization)
        for i in range(len(self.list)):
            self.civ.galaxy_dead(i)


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
        galaxy.civ = self
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
            who = self.list[who]
        if x is None or len(x) == 0:
            return True
        if self.get_sum_pd(who[NAMES.ocl]) < 0.1:
            return True
        if self.get_sum_mlst(who[NAMES.ocl]) < 0.002:
            return True
        return False

    def handle_found(self, event):
        if event.who is None:
            return
        who_index, who = event.who[NAMES.name], event.who
        start = event.start
        # 如果该国已死,或该星系已不属于它则返回
        if self.is_dead(who) or start not in who[NAMES.ocl]:
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
        # 处理相遇事件
        self.handle_found_galaxy(who, event.whom)

    def handle_found_galaxy(self, who, whom):
        """
        处理发现新的星系时的事务
        :param who: 文明向量
        :param whom: 星系坐标
        """
        belong = self.galaxy.list[whom, NAMES.belong]
        try:
            # 如果发现的不是无人区
            self.handle_met(who[NAMES.name], belong[NAMES.name])
        except TypeError:
            # 如果发现的是无人区
            # 签署为自己的地盘, 并赋予生产技术
            self.list[who[NAMES.name], NAMES.ocl].append(whom)
            self.galaxy.list[whom, NAMES.belong] = who
            self.galaxy.list[whom, NAMES.mlb] = who[NAMES.mlb]
            self.galaxy.list[whom, NAMES.lbf] = 0.0
            # 在该处发起探测
            self.detect(whom)

    def handle_met(self, one, another):
        """
        处理前者碰到后者事件
        :param one: 文明名
        :param another: 文明名
        :return:
        """
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
            if btype != NAMES.ATT:
                # 当另一者为结盟者或者保守者，另一者兵力受损0%~20%
                for gl in self.list[another, NAMES.ocl]:
                    self.galaxy.list[gl, NAMES.mlst] *= \
                        1 - np.random.rand() * 0.2

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
        old_range = galaxy.list[index, NAMES.dtrange]
        time = (distG ** 2 - old_range ** 2) // InitialValue.detect_speed ** 2 + self.time + 1
        # 下面的探测速度为半径固定增长，上面为面积固定增长
        # dist = distG - galaxy.list[index, NAMES.dtrange]
        # time = dist // InitialValue.detect_speed + self.time + 1
        event = Event(time, Event.TY_FOUND, owner, closest, start=index, range=distG)
        self.eventpool.put(event)
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

    def galaxy_dead(self, index):
        """
        判断星系是否死亡（资源耗尽），如果死亡则将其标记为死亡,并且把产值和兵力转移走（损失30%），
        并且将其宿主中该元素剥夺并判断其宿主是否死亡

        :param index: 星系下标
        :return:
        """
        ga = self.galaxy.list[index]
        if ga[NAMES.lbob] <= 0:
            # 没有宿主则返回
            if ga[NAMES.belong] is None:
                return
            # 若宿主文明只占有该星系则死亡
            if len(ga[NAMES.belong][NAMES.ocl]) == 1:
                self._destroy_civ(ga[NAMES.belong][NAMES.name])
                return
            # 否则将资源转移
            cv = ga[NAMES.belong]
            cv[NAMES.ocl].remove(index)
            # 默认转移到第一个星系中
            ano = cv[NAMES.ocl][0]
            self.galaxy.list[ano, NAMES.mlst] += ga[NAMES.mlst] * 0.7
            self.galaxy.list[ano, NAMES.pd] += ga[NAMES.pd] * 0.7
            # 将该星系恢复到默认状态
            self.galaxy.list[index, NAMES.mlst] = 0
            self.galaxy.list[index, NAMES.pd] = 0
            self.galaxy.list[index, NAMES.dtrange] = 0
            self.galaxy.list[index, NAMES.belong] = None
            self.galaxy.list[index, NAMES.mlb] = 0
            self.galaxy.list[index, NAMES.lbf] = 0

    def _is_dead_transfer(self, a, b):
        """
        判断两个文明是否死亡，若一方死亡则所有星系传给另一方
        :param a: 文明向量
        :param b: 文明向量
        """
        if self.is_dead(a) and self.is_dead(b):
            return
        if self.is_dead(a):
            a, b = b, a
        if self.is_dead(b):
            glist = b[NAMES.ocl]
            self._destroy_civ(b[NAMES.name])
            for g in glist:
                if g is None or g is np.NaN:
                    continue
                a[NAMES.ocl].append(g)
                self.galaxy.list[g, NAMES.belong] = a
                self.galaxy.list[g, NAMES.mlb] = a[NAMES.mlb]
                # 在该处发起探测
                self.detect(g)

    def _trans_galaxy(self, civ, galaxy):
        """
        将后者转移给前者
        :param civ: 文明向量
        :param galaxy: 星系下标
        :return:
        """
        g = self.galaxy.list[galaxy]
        g[NAMES.belong][NAMES.ocl].remove(galaxy)
        g[NAMES.belong] = civ
        civ[NAMES.ocl].append(galaxy)
        self.detect(galaxy)

    def _refresh_battle(self):
        self.battle_num = 0
        for one in self.list:
            if not one[NAMES.enemies]:
                continue
            for ano in one[NAMES.enemies]:
                if not self.list[ano, NAMES.enemies]:
                    # 敌人已死，移出列表
                    try:
                        one[NAMES.enemies].remove(ano)
                    except AttributeError:
                        pass
                    continue
                ano = self.list[ano]
                self.battle_num += 1
                abt, bbt = one[NAMES.btTch], ano[NAMES.btTch]
                adf, bdf = one[NAMES.dfTch], ano[NAMES.dfTch]
                # TODO: 改成对地区的战斗
                # 随机地区减小兵力（如果是地区对地区则太过麻烦了）
                a = one[NAMES.ocl]
                b = ano[NAMES.ocl]
                if a is None or b is None:
                    continue
                a, b = random.choice(a), random.choice(b)
                # a, b 为对战的星系坐标
                ga, gb = self.galaxy.list[a], self.galaxy.list[b]
                # 计算兵力损耗
                gbf, gaf = gb[NAMES.mlst], ga[NAMES.mlst]
                slow = InitialValue.slow_rate
                gb[NAMES.mlst], ga[NAMES.mlst] = gbf - (abt / bdf) / slow * gaf, gaf - (bbt / adf) / slow * gbf
                if gb[NAMES.mlst] < 0:
                    self._trans_galaxy(one, b)
                self._is_dead_transfer(one, ano)

    def _refresh_aid(self):
        for one in range(len(self.galaxy.list)):
            # 当该星系无人时到下一轮
            if self.galaxy.list[one, NAMES.belong] is None \
               or self.galaxy.list[one, NAMES.mlb] < 0.3:
                continue
            belong = self.galaxy.list[one, NAMES.belong]
            # FIXME 这是一处BUG,本不应该有这种情况
            if belong[NAMES.ocl] is None:
                self.galaxy.list[one, NAMES.belong] = None
                continue
            mlst = self.galaxy.list[one, NAMES.mlst]
            # 兵力大于二十则不需要支援
            if mlst >= 20:
                continue
            # 在本文明中寻求支援
            # 在盟友当中寻求资源
            this = belong

            def next_belong():
                for a in this[NAMES.allies]:
                    yield self.list[a]

            ne = next_belong()
            while belong is not None:
                if belong[NAMES.ocl] is None:
                    belong = next(ne, None)
                    continue
                for ano in belong[NAMES.ocl]:
                    ano_mlst = self.galaxy.list[ano, NAMES.mlst]
                    if ano_mlst <= mlst:
                        # 过滤掉自己以及兵力更少的星系
                        continue
                    # TODO SET 支援设置
                    aid = min(((ano_mlst - mlst) / 2, 1000/(self.galaxy.dist[one, ano]), 5))
                    # aid = (ano_mlst - mlst) / 2
                    # 简单的平分法
                    # TODO： 添加函数，对大兵力进行限制
                    ano_mlst, mlst = ano_mlst - aid, mlst + aid
                    if mlst >= 20:
                        break
                    self.galaxy.list[ano, NAMES.mlst] = ano_mlst
                belong = next(ne, None)

            # 更新资源状态
            self.galaxy.list[one, NAMES.mlst] = mlst

    # TODO:讨要救济
    def _refresh_supply(self):
        pass

    # TODO:饥荒状态更新判断
    def _refresh_starve(self):
        pass
