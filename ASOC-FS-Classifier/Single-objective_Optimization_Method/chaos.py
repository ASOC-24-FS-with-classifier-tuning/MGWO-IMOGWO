import math
from random import random
import numpy


# 混沌算子
def chaos(Index, Initial_value, Max_iter):
    x = Initial_value
    pops = []
    if Index == 1:
        # Chebyshev map
        k = 4
        for i in range(Max_iter):
            x = math.cos(k * math.acos(x))
            pops.append(x)
    elif Index == 2:
        # Cricle map
        a = 0.5
        b = 2.2
        for i in range(Max_iter):
            x = x + b - ((a / (2 * math.pi)) * math.sin(2 * math.pi * x)) % 1
            pops.append(x)
    elif Index == 3:
        # Causs/mouse map
        for i in range(Max_iter):
            if x == 0:
                x = 0
            else:
                x = 1 / x - int(1/x)
            pops.append(x)
    elif Index == 4:
        # Iterative map
        k = 0.7
        a = -1
        b = 1
        c = 0
        d = 1
        for i in range(Max_iter):
            x = math.sin((k * math.pi) / x)
            # 此时x可能介于[-1,1]之间，将其映射到[0,1]之间
            temp = ((x - a) * (d- c)) / (b - a)
            pops.append(temp)
    elif Index == 5:
        # Logistic map
        a = 4
        for i in range(Max_iter):
            x = a * x * (1 - x)
            pops.append(x)
    elif Index == 6:
        # Piecewise map
        p = 0.4
        for i in range(Max_iter):
            if 0 <= x < p:
                x = x / p
            elif p <= x < 0.5:
                x = (x - p) / (0.5 - p)
            elif 0.5 <= x < 1 - p:
                x = (1 - p -x) / (0.5 - p)
            elif 1 - p <= x < 1:
                x = (1 - x) / p
            pops.append(x)
    elif Index == 7:
        # Sine map
        for i in range(Max_iter):
            x = math.sin(math.pi * x)
            pops.append(x)
    elif Index == 8:
        # Singer map
        u = 1.07
        for i in range(Max_iter):
            x = u * (7.86 * x - 23.31 * x **2 + 28.75 * x ** 3 - 13.302875 * x ** 4)
            pops.append(x)
    elif Index == 9:
        # Sinusoidal map
        for i in range(Max_iter):
            x = 2.3 * x ** 2 * math.sin(math.pi * x)
            pops.append(x)
    elif Index == 10:
        # Tent map
        x = 0.6
        for i in range(Max_iter):
            if x < 0.7:
                x = x / 0.7
            else:
                x = (10/3)*(1-x)
            pops.append(x)
    elif Index == 11:
        # Kent map
        beta = 0.5
        for i in range(Max_iter):
            if x <= beta:
                x = x / beta
            else:
                x = (1-x)/(1-beta)
            pops.append(x)
    elif Index == 12:
        # Bernoulli map
        a = 0.4
        for i in range(Max_iter):
            if x <= 1 - a:
                x = x / (1-a)
            else:
                x = (x-(1-a))/a
            pops.append(x)
    elif Index == 13:
        # Gaussian map
        mu = 0.5
        for i in range(Max_iter):
            if x == 0:
                x = 0
            else:
                x = (mu/x) % 1
            pops.append(x)
    elif Index == 14:
        # Cubic map
        rou = 2.3
        for i in range(Max_iter):
            x = rou*x*(1 - x*x)
            pops.append(x)
    return pops

# 映射函数
def shaped(index, soulution):
    soulution_fit = soulution * 1
    if index == 1:
        # s型映射函数（sigmoid 函数）
        p = random.random()
        for i in range(soulution.shape[0]): # soulution.shape[0]表示解的每一列
            a = 1 / (1 + numpy.exp(-soulution[i]))  # sigmoid函数 对位置进行二进制映射
            if a > 0.5:
                soulution_fit[i] = 1
            if a < 0.5:
                soulution_fit[i] = 0
        while (sum(soulution_fit) < 2):  # 有全False时重新生成个体
            b = 1 - 2 * numpy.random.rand(soulution.shape[0])
            soulution_fit = b * 1
            for i in range(soulution_fit.shape[0]):
                a = 1 / (1 + numpy.exp(-soulution_fit[i]))
                if a > 0.5:
                    soulution_fit[i] = 1
                if a < 0.5:
                    soulution_fit[i] = 0
    elif index == 2:
        # v型映射函数
        p = random.random()
        for i in range(soulution.shape[0]):  # soulution.shape[0]表示解的每一列
            a = math.fabs(soulution[i] / math.sqrt(soulution[i] ** 2 + 1))
            if a > 0.5:
                soulution_fit[i] = 1
            if a < 0.5:
                soulution_fit[i] = 0
        while (sum(soulution_fit) < 2):  # 有全False时重新生成个体
            b = 1 - 2 * numpy.random.rand(soulution.shape[0])
            soulution_fit = b * 1
            for i in range(soulution_fit.shape[0]):
                a = 1 / (1 + numpy.exp(-soulution_fit[i]))
                if a > 0.5:
                    soulution_fit[i] = 1
                if a < 0.5:
                    soulution_fit[i] = 0
    return soulution_fit

# 莱维飞行
def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step

# 反向学习
def opposition(solution):
    solution_opp = solution * 1
    # print("solution.shape[0]:")
    # print(solution.shape[0])
    # print("solution:")
    # print(solution)
    for i in range(solution.shape[0]):
        solution_opp[i] = 1 - solution[i]
    # print("solution_opp:")
    # print(solution_opp)
    return solution_opp


