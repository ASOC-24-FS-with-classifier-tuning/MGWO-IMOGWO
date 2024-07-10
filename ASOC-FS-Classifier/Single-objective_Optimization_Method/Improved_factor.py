import math

import numpy
# 汉明距离（就二进制解而言，欧式距离和汉明距离无区别）
def Hamming_Distance_0(id, population):
    # populationSize为种群中解的个数
    populationSize = population.shape[0]
    # 定义了一个populationSize*populationSize的二维数组，用于存储各个解之间的汉明距离（dis）
    dis = numpy.zeros((populationSize, populationSize))
    for i in range(0, populationSize):
        for j in range(0, populationSize):
            if (i == j):
                dis[i, j] = 0
            elif (i < j):
                dis[i, j] = sum(population[i, :] != population[j, :])
            else:
                dis[i, j] = dis[j, i]
    # indicator是一个1*populationSize的数组，indicator[i]表示种群中第i个解的汉明距离指标，越大说明这个解与其他解越不相似
    indicator = sum(dis)
    # 解与其本身汉明距离指标为0
    indicator[id] = 0
    # 按升序排序
    index = numpy.argsort(indicator)
    # 最大的汉明距离指标
    maxValue = indicator[index[populationSize - 1]]
    # 存储拥有最大汉明距离的解下标
    values = []
    for i in range(0, populationSize):
        if indicator[i] == maxValue:
            values.append(i)
    if len(values) == 1:
        solutionid = values[0]
    else:
        max = 0
        for i in range(0, len(values)):
            if (dis[id, values[i]] > max):
                max = dis[id, values[i]]
        tempid = []
        for j in range(0, len(values)):
            if (dis[id, values[j]] == max):
                tempid.append(values[j])
        # 有一个或者多个
        kk = numpy.random.randint(0, len(tempid))
        solutionid = tempid[kk]
    return solutionid

# 准随机序列初始化（Sobol, Halton）
from scipy.stats import qmc, bernoulli

# Sobol序列
def Sobol_Initialization(popSize, dim, para_size):
    pop = numpy.zeros((popSize, dim+para_size))
    sobol = qmc.Sobol(d = (dim + para_size))
    sobol_samples = sobol.random(popSize)
    # sobol_samples是一组popSize*(dim+para_size)的随机值，介于0,1之间，需要使用映射机制，将其转换为一组二进制解（使用伯努利分布映射）
    binary_population = []
    ##################################伯努利分布##################################
    n = 0
    for sample in sobol_samples:
        for j in range(dim):
            pop[n, j] = int(bernoulli.rvs(sample[j]))
        # para_size = 3
        pop[n, dim] = (2 ** (-1)) + sample[j] * (2 ** (5) - 2 ** (-1))
        pop[n, dim+1] = (2 ** (-4)) + sample[j] * (2 ** (5) - 2 ** (-4))
        pop[n, dim+2] = int(bernoulli.rvs(sample[j]))
        n += 1
    #################################S-映射（效果不行）###########################################
    # for sample in sobol_samples:
    #     binary_solution = transform_S(sample, dim)
    #     binary_population.append(binary_solution)
    print(pop)
    return pop

def find_Value(k, b):
    value = 0
    denominator = b
    current_k = k
    while current_k > 0:
        digit = current_k % b
        value += digit / denominator
        current_k //= b
        denominator *= b
    return value

# VDC序列
def van_der_corput(popSize, dim, para_size, base):
    n = popSize * (dim+para_size)
    vdc_sequence = []
    for i in range(n):
        vdc_value = find_Value(i, base)
        vdc_sequence.append(vdc_value)
    pop = numpy.zeros((popSize, dim + para_size))
    k = 0
    for i in range(popSize):
        for j in range(dim+para_size):
            pop[i,j] = vdc_sequence[k]
            k = k + 1
    #################################伯努利分布##################################
    n = 0
    for sample in pop:
        for j in range(dim):
            pop[n, j] = int(bernoulli.rvs(sample[j]))
        # para_size = 3
        pop[n, dim] = (2 ** (-1)) + sample[j] * (2 ** (5) - 2 ** (-1))
        pop[n, dim+1] = (2 ** (-4)) + sample[j] * (2 ** (5) - 2 ** (-4))
        pop[n, dim+2] = int(bernoulli.rvs(sample[j]))
        n += 1
    return pop

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def generate_primes(k):
    primes = []
    num = 2  # 从第一个质数开始
    while len(primes) < k:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes
# Halton序列
def Halton_Initialization(popsize, dim, para_size):
    base = generate_primes(dim + para_size)
    Hammersley_seq = numpy.zeros((popsize, dim + para_size))
    for d in  range(dim + para_size):
        for i in range(popsize):
            if d == 0:
                Hammersley_seq[i,d] = i / popsize
            else:
                Hammersley_seq[i, d] = find_Value(i, base[d-1])
    ##################################伯努利分布##################################
    n = 0
    for sample in Hammersley_seq:
        for j in range(dim):
            Hammersley_seq[n, j] = int(bernoulli.rvs(sample[j]))
        # para_size = 3
        Hammersley_seq[n, dim] = (2 ** (-1)) + sample[j] * (2 ** (5) - 2 ** (-1))
        Hammersley_seq[n, dim + 1] = (2 ** (-4)) + sample[j] * (2 ** (5) - 2 ** (-4))
        Hammersley_seq[n, dim + 2] = int(bernoulli.rvs(sample[j]))
        n += 1
    return Hammersley_seq


# Faure序列
# 整数k以b为基底可以表示成
def basexpflip(k, b):
    a = []
    if k == 0:
        a.append(0)
    else:
        j = int(numpy.fix(numpy.log(k)/numpy.log(b)) + 1)
        q = pow(b,(j-1))
        for i in range(j):
            temp = numpy.floor(k/q)
            a.append(temp)
            k = k - q * a[i]
            q = q/b
        a = numpy.flipud(a)
    return a
# n,k均为非负整数时，这是从n种情况中一次取出k种的组合的数量
def diycomb(n, k):
    if n < k:
        c = 0
    else:
        c = math.comb(n,k)
    return c
def Faure_Initialization(popSize, dim, para_size):
    temp = generate_primes(dim + para_size)
    for i in temp:
        if i >= dim:
            base = i
            break
    faure_seq = numpy.zeros((dim+para_size, popSize))
    for i in range(popSize):
        a = basexpflip(i, base)
        J = len(a)
        L = J - 1
        y = numpy.zeros((J,1))
        g = []
        for j in range(J):
            temp = []
            temp.append(pow(base, j+1))
            g.append(temp)
        for d in range(dim):
            for j in range(J):
                S = 0
                for l in range(L+1):
                    c = diycomb(l, j)
                    if d == 0 and (l-j) < 0:
                        h = 0
                    else:
                        h = pow(d, l - j)
                    S = S + c * h * a[l]
                y[j] = numpy.mod(S,base)
            faure_seq[d,i] = numpy.sum(y/g)
    faure_seq = numpy.transpose(faure_seq)
    print(faure_seq)
    #################################伯努利分布##################################
    n = 0
    for sample in faure_seq:
        for j in range(dim):
            faure_seq[n, j] = int(bernoulli.rvs(sample[j]))
        # para_size = 3
        faure_seq[n, dim] = (2 ** (-1)) + sample[j] * (2 ** (5) - 2 ** (-1))
        faure_seq[n, dim + 1] = (2 ** (-4)) + sample[j] * (2 ** (5) - 2 ** (-4))
        faure_seq[n, dim + 2] = int(bernoulli.rvs(sample[j]))
        n += 1
    return faure_seq

# CVT初始化
def CVT(popSize, dim):
    # step1：首先随机生成PopSize个解，最为初始generators（定义一个1*PopSize的集合，用于存储每个generator的附属解）
    # generators = numpy.random.uniform(0, 1, (popSize, dim))
    generators = numpy.random.randint(0, 2, (popSize, dim))
    generators = numpy.array(generators)
    # print(generators)
    G = []
    for n in range(0, popSize):
        G.append([])
    # step2：首先随机生成2*PopSize个额外粒子
    extral_solution = numpy.random.randint(0, 2, (2*popSize, dim))
    # print(extral_solution)
    for q in range(0, 2*popSize):
        res = Hamming_Distance(extral_solution[q,:], generators, popSize, dim)
        # 规则1：海明距离定义，相同位数越多，说明越接近，即属于该generator的子集
        # 额外解可能与多个生成器距离一样
        if len(res) == 1:
            k = res[0]
            # print(k)
            G[k].append(q)
        # 规则2：最少优先，即额外解到多个generator的海明距离相同时，G[k]中元素少的优先
        else:
            # print(res)
            temp = numpy.zeros(popSize)
            for i in range(0, popSize):
                temp[i] = float("inf")
            for v in res:
                temp[v] = len(G[v])
            Min = numpy.min(temp)
            Min_temp = []
            for i in range(0, popSize):
                if (temp[i] == Min):
                    Min_temp.append(i)
            if len(Min_temp) == 1:
                G[Min_temp[0]].append(q)
            else:
                # 规则3：随机规则，在前两个规则优先级均一致的情况下，随机选择一个子集加入
                id = numpy.random.randint(0,len(Min_temp))
                # print(Min_temp[id])
                G[Min_temp[id]].append(q)
    # for n in range(0, popSize):
    #     print(G[n])
    Value = numpy.zeros((popSize, dim))
    for n in range(0, popSize):
        for d in range(0, dim):
            Value[n, d] = Value[n, d] + generators[n, d]
    # print(Value)
    for n in range(0, popSize):
        for v in G[n]:
            for j in range(0, dim):
                Value[n, j] = Value[n, j] + extral_solution[v, j]
        # print(Value[n, :])
        # print(1+len(G[n]))
        kk = 1+len(G[n])
        for j in range(0, dim):
            Value[n, j] = 1.0 * Value[n, j] / kk
        # print(Value[n, :])

    binary_population = []
    ##################################伯努利分布##################################
    for sample in Value:
        binary_solution = [int(bernoulli.rvs(p)) for p in sample]
        binary_population.append(binary_solution)
    # print(binary_population)
    return binary_population


# 输入：额外解——solution，生成器——pop
# 输入：solution与pop中每个解的海明距离（越大说明solution越接近该解）1*N的数组，Res[i]=id，id为pop中解的下标
def Hamming_Distance(solution, pop, N, dim):
    # dis记录解solution与generators的海明距离
    dis = numpy.zeros(N)
    for n in range(0, N):
        for j in range(0, dim):
            if solution[j] == pop[n, j]:
                dis[n] = dis[n] + 1
    # print(dis)
    Max = numpy.max(dis)
    Res = []
    for n in range(0, N):
        if dis[n] == Max:
            Res.append(n)
    return Res

