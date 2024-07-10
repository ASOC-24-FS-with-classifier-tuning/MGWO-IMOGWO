import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import xlsxwriter
import copy
np.set_printoptions(threshold=np.inf)

def get_ObjectFunction(population, dim, x, y):
    f = []
    for i in range(population.shape[0]):
        while (sum(population[i][:dim]) < 2):
            population[i][:dim] = np.random.randint(0, 2, dim)
        column_use = (population[i][:dim] == 1)
        x_test = x.columns[column_use]
        if population[i][dim + 2] == 0:
            clf = OneVsOneClassifier(
                SVC(C=population[i][dim], gamma=population[i][dim + 1], random_state=None, kernel='sigmoid',
                    cache_size=2000))
        if population[i][dim + 2] == 1:
            clf = OneVsOneClassifier(
                SVC(C=population[i][dim], gamma=population[i][dim + 1], random_state=None, kernel='rbf',
                    cache_size=2000))
        X_train, X_test, y_train, y_test = train_test_split(x[x_test], y, test_size=0.3, stratify=y, random_state=42)
        clf.fit(X_train, y_train)
        ypred = clf.predict(X_test)
        fitness_1 = 1 - accuracy_score(y_test, ypred)
        precision = precision_score(y_test, ypred, average='weighted', zero_division=0)
        recall = recall_score(y_test, ypred, average='weighted', zero_division=0)
        f1score = f1_score(y_test, ypred, average='weighted')
        f.append([fitness_1, 1.0 * (sum(column_use)) / dim, precision, recall, f1score])
    f = np.array(f)
    return f

def dominates(x, y):
    if all(x <= y) and any(x < y):
        return True
    else:
        return False

def updateArchive(Archive_X, Archive_F, population, particles_F, dim):
    Archive_temp_X = np.vstack((Archive_X, population))
    Archive_temp_F = np.vstack((Archive_F, particles_F))
    Archive = np.hstack((Archive_temp_X, Archive_temp_F))
    Archive = copy.deepcopy(np.array(list(set([tuple(t) for t in Archive]))))
    Archive_temp_X = np.array(Archive[:, 0:dim + 3])
    Archive_temp_F = np.array(Archive[:, dim + 3:dim + 3 + 5])
    num_rows = Archive_temp_F.shape[0]
    o = np.zeros(num_rows)
    for i in range(0, num_rows):
        for j in range(0, num_rows):
            if i != j:
                temp_F_j = np.array(Archive_temp_F[j, 0:2])
                temp_F_i = np.array(Archive_temp_F[i, 0:2])
                if dominates(temp_F_j, temp_F_i):
                    o[i] = 1
                    break
        pass
    Archive_member_no = 0
    Archive_X_updated = []
    Archive_F_updated = []
    for i in range(Archive_temp_F.shape[0]):
        if o[i] == 0:
            Archive_member_no = Archive_member_no + 1
            Archive_X_updated.append(Archive_temp_X[i])
            Archive_F_updated.append(Archive_temp_F[i])
    Archive_X_updated = copy.deepcopy(np.array(Archive_X_updated))
    Archive_F_updated = copy.deepcopy(np.array(Archive_F_updated))
    return Archive_X_updated, Archive_F_updated, Archive_member_no

def RankingProcess(Archive_F, obj_no):
    my_min = [min(Archive_F[:, 0]), min(Archive_F[:, 1])]
    my_max = [max(Archive_F[:, 0]), max(Archive_F[:, 1])]

    r = [(my_max[0] - my_min[0]) / 10, (my_max[1] - my_min[1]) / 10]
    ranks = np.zeros(len(Archive_F))
    for i in range(len(Archive_F)):
        ranks[i] = 0
        for j in range(len(Archive_F)):
            flag = 0
            for k in range(obj_no):
                if math.fabs(Archive_F[j][k] - Archive_F[i][k]) <= r[k]:
                    flag = flag + 1
            if flag == obj_no:
                ranks[i] = ranks[i] + 1
                pass
            pass
        pass
    return ranks

def HandleFullArchive(Archive_X, Archive_F, ArchiveMaxSize, dim):
    function1_values = [Archive_F[i][0] for i in range(0, len(Archive_F))]
    function2_values = [Archive_F[i][1] for i in range(0, len(Archive_F))]

    non_dominated_sorted_solution = [i for i in range(0, len(Archive_F))]
    cd, front = crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution)
    front22 = sort_by_values1(front, cd)
    front22.reverse()
    Array01_X = np.zeros((ArchiveMaxSize, dim + 3))
    Array01_F = np.zeros((ArchiveMaxSize, 5))
    for i in range(0, ArchiveMaxSize):
        Array01_X[i] = copy.deepcopy(Archive_X[front22[i]])
        Array01_F[i] = copy.deepcopy(Archive_F[front22[i]])
        pass
    Archive_X = copy.deepcopy(Array01_X)
    Archive_F = copy.deepcopy(Array01_F)
    Archive_member_no = ArchiveMaxSize
    Archive_mem_ranks = RankingProcess(Archive_F, 2)
    return Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list
def sort_by_values1(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        sorted_list.append(list1[values.index(min(values))])
        values[index_of(min(values), values)] = math.inf
    return sorted_list
def RouletteWheelSelection(weights):
    accumulation = np.cumsum(weights)
    p = random.random() * accumulation[accumulation.shape[0] - 1]
    chosen_index = -1
    for i in range(accumulation.shape[0]):
        if (accumulation[i] > p):
            chosen_index = i
            break
    pass
    o = copy.deepcopy(chosen_index)
    return o

def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    front = copy.deepcopy(sorted1)
    distance[0] = 44444444444
    distance[len(front) - 1] = 44444444444
    for k in range(1, len(front) - 1):
        if max(values1) - min(values1) != 0:
            distance[k] = distance[k] + (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
                    max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        if max(values2) - min(values2) != 0:
            distance[k] = distance[k] + (values2[sorted1[k - 1]] - values2[sorted1[k + 1]]) / (
                    max(values2) - min(values2))
    return distance, front

def get_pbest(pbest_X, pbest_F, population, particles_F):
    for i in range(population.shape[0]):
        temp_particles_i = np.array(particles_F[i, 0:2])
        temp_pbest_i = np.array(pbest_F[i, 0:2])
        if dominates(temp_particles_i, temp_pbest_i):
            pbest_X[i] = copy.deepcopy(population[i])
            pbest_F[i] = copy.deepcopy(particles_F[i])
        else:
            r = random.random()
            if r < 0.5:
                pbest_X[i] = copy.deepcopy(population[i])
                pbest_F[i] = copy.deepcopy(particles_F[i])
            else:
                pbest_X[i] = copy.deepcopy(pbest_X[i])
                pbest_F[i] = copy.deepcopy(pbest_F[i])
    pass
    return pbest_X, pbest_F

def mopso_svc(name, arr):
    inputname = name
    for mm in arr:
        start1 = time.time()
        inputdata = '../' + inputname + '.csv'
        dataset = pd.read_csv(inputdata, header=None)
        path = '../ExperimentsData/Multi/' + inputname + '_MOPSO_' + str(int(time.time())) + '.xlsx'
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        workbook = xlsxwriter.Workbook(path)
        worksheet1 = workbook.add_worksheet("The 50th iteration")
        worksheet2 = workbook.add_worksheet("The 100th iteration")
        worksheet3 = workbook.add_worksheet("The 200th iteration")
        worksheet4 = workbook.add_worksheet("Running Time")
        worksheet5 = workbook.add_worksheet("Non-dominated solutions")
        ColName = ["f1（classification_error_rate）", 'f2（ratio）', 'f3（precision）', 'f4（recall）', 'f5（f1_score）']
        for i in range(len(ColName)):
            worksheet1.write(0, i, ColName[i])
            worksheet2.write(0, i, ColName[i])
            worksheet3.write(0, i, ColName[i])
        worksheet4.write(0, 0, 'cpu-runtime')
        x = dataset.iloc[:, 0:-1]
        x = pd.DataFrame(StandardScaler().fit_transform(x))
        y = dataset.iloc[:, -1]
        population_size = 20
        T = 200
        number = []
        dim = x.shape[1]
        obj_no = 2
        para_size = 3
        TMax = np.zeros(T)
        tt = np.zeros(T)
        for i in range(0, T):
            tt[i] = i
        ArchiveMaxSize = 20
        Archive_F1 = []
        Archive_F2 = []
        Archive_X1 = []
        Archive_X2 = []
        w = 2
        wmax = 0.9
        wmin = 0.4
        c1 = 2
        c2 = 2
        Vmax = 6
        Archive_X = np.zeros((ArchiveMaxSize, dim + para_size))
        Archive_F = np.ones((ArchiveMaxSize, 5)) * float("inf")
        Archive_member_no = 0
        V = (-1 + 2 * np.random.rand(population_size, dim + para_size)) * Vmax
        population = np.zeros((population_size, dim + para_size))
        for i in range(population_size):
            for j in range(dim + para_size):
                if j < dim or j == dim + 2:
                    population[i][j] = 0 + round(random.random() * (1 - 0))
                if j == dim:
                    population[i][j] = (2 ** (-1)) + random.random() * (2 ** (5) - 2 ** (-1))
                if j == dim + 1:
                    population[i][j] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))

                pass
            pass
        particles_F = get_ObjectFunction(population, dim, x, y)
        gbest_F = float("inf") * np.ones(5)
        gbest_X = np.zeros(dim + para_size)
        Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, population, particles_F, dim) * 1
        if Archive_member_no > ArchiveMaxSize:
            Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = HandleFullArchive(Archive_X, Archive_F, ArchiveMaxSize, dim)
        else:
            Archive_mem_ranks = RankingProcess(Archive_F, obj_no)
        index = RouletteWheelSelection(1 / Archive_mem_ranks)
        if index == -1:
            index = 0
        pbest_F = copy.deepcopy(particles_F)
        pbest_X = copy.deepcopy(population)
        gbest_X = copy.deepcopy(Archive_X[index])
        gbest_F = copy.deepcopy(Archive_F[index])
        for t in range(T):
            start = time.time()
            if t == 50:
                for q in range(len(Archive_F)):
                    Archive_F1.append(Archive_F[q])
                    Archive_X1.append(Archive_X[q])
            if t == 100:
                for q in range(len(Archive_F)):
                    Archive_F2.append(Archive_F[q])
                    Archive_X2.append(Archive_X[q])
            w = wmax - (wmax - wmin) / T
            for i in range(population.shape[0]):
                for j in range(dim + para_size):
                    V[i][j] = w * V[i][j] + c1 * random.random() * (
                            pbest_X[i][j] - population[i][j]) + c2 * random.random() * (
                                      gbest_X[j] - population[i][j])
                    if (V[i][j] > Vmax):
                        V[i][j] = Vmax
                    if (V[i][j] < -Vmax):
                        V[i][j] = -Vmax
                    if j < dim or j == dim + 2:
                        p = random.random()
                        s = 1 / (1 + math.exp((-1) * V[i][j]))
                        if (p < s):
                            population[i][j] = 1
                        else:
                            population[i][j] = 0
                            pass
                        pass
                    if j == dim:
                        population[i][j] = np.clip(population[i][j] + V[i][j], 2 ** (-1), 2 ** (5))
                    if j == dim + 1:
                        population[i][j] = np.clip(population[i][j] + V[i][j], 2 ** (-4), 2 ** (5))
                pass
            particles_F = get_ObjectFunction(population, dim, x, y)
            Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, population, particles_F, dim)
            if Archive_member_no > ArchiveMaxSize:
                Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = HandleFullArchive(Archive_X, Archive_F, ArchiveMaxSize, dim)
            else:
                Archive_mem_ranks = RankingProcess(Archive_F, obj_no)
            index = RouletteWheelSelection(1 / Archive_mem_ranks)
            if index == -1:
                index = 0
            gbest_X = copy.deepcopy(Archive_X[index])
            gbest_F = copy.deepcopy(Archive_F[index])
            pbest_X, pbest_F = get_pbest(pbest_X, pbest_F, population, particles_F)
            TMax[t] = Archive_member_no
            print(
                "At the iteration {} there are {} non-dominated solutions in the archive".format(t, Archive_member_no))
            end = time.time()

        Archive_F1 = np.array(Archive_F1)
        Archive_F2 = np.array(Archive_F2)
        Archive_F = np.array(Archive_F)

        len1 = Archive_F1.shape[0]
        for i in range(0, len1):
            worksheet1.write(i + 1, 0, Archive_F1[i][0])
            worksheet1.write(i + 1, 1, Archive_F1[i][1])
            worksheet1.write(i + 1, 2, Archive_F1[i][2])
            worksheet1.write(i + 1, 3, Archive_F1[i][3])
            worksheet1.write(i + 1, 4, Archive_F1[i][4])
            for j in range(0, dim + para_size):
                worksheet1.write(i + 1, 5 + j, Archive_X1[i][j])
        len2 = Archive_F2.shape[0]
        for i in range(0, len2):
            worksheet2.write(i + 1, 0, Archive_F2[i][0])
            worksheet2.write(i + 1, 1, Archive_F2[i][1])
            worksheet2.write(i + 1, 2, Archive_F2[i][2])
            worksheet2.write(i + 1, 3, Archive_F2[i][3])
            worksheet2.write(i + 1, 4, Archive_F2[i][4])
            for j in range(0, dim + para_size):
                worksheet2.write(i + 1, 5 + j, Archive_X2[i][j])
        len3 = Archive_F.shape[0]
        for i in range(0, len3):
            worksheet3.write(i + 1, 0, Archive_F[i][0])
            worksheet3.write(i + 1, 1, Archive_F[i][1])
            worksheet3.write(i + 1, 2, Archive_F[i][2])
            worksheet3.write(i + 1, 3, Archive_F[i][3])
            worksheet3.write(i + 1, 4, Archive_F[i][4])
            for j in range(0, dim + para_size):
                worksheet3.write(i + 1, 5 + j, Archive_X[i][j])
        for i in range(0, T):
            worksheet5.write(i, 0, TMax[i])

        end1 = time.time()
        worksheet4.write(1, 0, end1 - start1)
        workbook.close()
        print("over~")