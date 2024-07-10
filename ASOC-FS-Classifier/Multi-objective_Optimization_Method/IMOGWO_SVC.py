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
import chaos
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
    pass

def updateArchive(Archive_X, Archive_F, pop, particles_F, dim):
    Archive_temp_X = np.vstack((Archive_X, pop))
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
    for i in range(num_rows):
        if o[i] == 0:
            Archive_member_no = Archive_member_no + 1
            Archive_X_updated.append(Archive_temp_X[i])
            Archive_F_updated.append(Archive_temp_F[i])
    Archive_X_updated = copy.deepcopy(np.array(Archive_X_updated))
    Archive_F_updated = copy.deepcopy(np.array(Archive_F_updated))
    return Archive_X_updated, Archive_F_updated, Archive_member_no

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

def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] <= values1[q] and values2[p] <= values2[q]) and (
                    values1[p] < values1[q] or values2[p] < values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] <= values1[p] and values2[q] <= values2[p]) and (
                    values1[q] < values1[p] or values2[q] < values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
    del front[len(front) - 1]
    return front

def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    front = copy.deepcopy(sorted1)
    distance[0] = 44444444444
    distance[len(front) - 1] = 44444444444
    for k in range(1, len(front) - 1):
        if max(values1) - min(values1) != 0:
            distance[k] = distance[k] + (
                    (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (max(values1) - min(values1)))
            pass
    for k in range(1, len(front) - 1):
        if max(values2) - min(values2) != 0:
            distance[k] = distance[k] + (
                    (values2[sorted1[k - 1]] - values2[sorted1[k + 1]]) / (max(values2) - min(values2)))
    return distance, front

def init(population_size, dim, para_size):
    if dim < 30:
        half = (int)(population_size / 2)
        pop = np.zeros((half, dim + para_size))
        x_list = chaos.tentmap12(0.7, 0.6, half * (dim + para_size))
        count = 0
        for i in range(half):
            for j in range(dim + para_size):
                if j < dim or j == dim + 2:
                    pop[i][j] = 0 + round(abs(x_list[count]) * (1 - 0))
                if j == dim:
                    pop[i][j] = (2 ** (-1)) + abs(x_list[count]) * (2 ** (5) - 2 ** (-1))
                if j == dim + 1:
                    pop[i][j] = (2 ** (-4)) + abs(x_list[count]) * (2 ** (5) - 2 ** (-4))
                count = count + 1
                pass
            pass
        population = np.zeros((half, dim))
        for i in range(half):
            for j in range(dim):
                population[i][j] = 0 + round(random.random() * (1 - 0))
        parameter = np.zeros((half, para_size))
        for i in range(0, parameter.shape[0]):
            parameter[i][0] = (2 ** (-1)) + random.random() * (2 ** (5) - 2 ** (-1))
            parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
            parameter[i][2] = 0 + round(random.random() * (1 - 0))
        pop01 = np.hstack((population, parameter))
        pop = np.vstack((pop, pop01))
        pass
    elif dim >= 30:
        pop = np.zeros((population_size, dim + para_size))
        x_list = chaos.sinusoidalmap11(0.7, 2.3, population_size * (dim + para_size))
        count = 0
        for i in range(population_size):
            for j in range(dim + para_size):
                if j < dim or j == dim + 2:
                    pop[i][j] = 0 + round(abs(x_list[count]) * (1 - 0))
                if j == dim:
                    pop[i][j] = (2 ** (-1)) + abs(x_list[count]) * (2 ** (5) - 2 ** (-1))
                if j == dim + 1:
                    pop[i][j] = (2 ** (-4)) + abs(x_list[count]) * (2 ** (5) - 2 ** (-4))
                count = count + 1
        pass
    pop01 = np.zeros((population_size, dim + para_size))
    for i in range(population_size):
        for j in range(dim + para_size):
            if j < dim or j == dim + 2:
                pop01[i][j] = 0 + 1 - pop[i][j]
            if j == dim:
                pop01[i][j] = 2 ** (-1) + 2 ** (5) - pop[i][j]
            if j == dim + 1:
                pop01[i][j] = 2 ** (-4) + 2 ** (5) - pop[i][j]
    pop02 = np.vstack((pop, pop01))
    return pop02

def Crowded_distance_rank(Archive_F):
    function1_values = [Archive_F[i][0] for i in range(0, len(Archive_F))]
    function2_values = [Archive_F[i][1] for i in range(0, len(Archive_F))]
    non_dominated_sorted_solution = [i for i in range(0, len(Archive_F))]
    cd, front = crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution)
    front22 = sort_by_values1(front, cd)
    front22.reverse()
    if len(front22) >= 2:
        aa = front22[0]
        front22[0] = front22[1] * 1
        front22[1] = aa * 1
    return front22
def sort_pop(pop, particles_F, population_size):
    solution2 = copy.deepcopy(np.array(pop))
    function1_values2 = [particles_F[i][0] for i in range(0, len(particles_F))]
    function2_values2 = [particles_F[i][1] for i in range(0, len(particles_F))]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        cd, front = crowding_distance(function1_values2[:], function2_values2[:],
                                      non_dominated_sorted_solution2[i][:])
        non_dominated_sorted_solution2[i] = copy.deepcopy(front)
        crowding_distance_values2.append(cd)
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        front22 = sort_by_values1(non_dominated_sorted_solution2[i],
                                  crowding_distance_values2[i])  # (list,values)
        front22.reverse()
        for value in front22:
            new_solution.append(value)
            if (len(new_solution) == population_size):
                break
        if (len(new_solution) == population_size):
            break
    pop = [solution2[i] for i in new_solution]
    pop = copy.deepcopy(np.array(pop))
    particles_F = [particles_F[i] for i in new_solution]
    particles_F = copy.deepcopy(np.array(particles_F))
    return pop, particles_F

def imogwo_svc(name, arr):
    inputname = name
    for mm in arr:
        start1 = time.time()
        inputdata = '../' + inputname + '.csv'
        dataset = pd.read_csv(inputdata, header=None)
        path = '../ExperimentsData/Multi/' + inputname + '_IMOGWO_' + str(int(time.time())) + '.xlsx'
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
        dim = x.shape[1]
        obj_no = 2
        para_size = 3
        TMax = np.zeros(T)
        tt = np.zeros(T)
        for i in range(0, T):
            tt[i] = i
        ArchiveMaxSize = 20
        Archive_member_no = 0
        Archive_F1 = []
        Archive_F2 = []
        Archive_X1 = []
        Archive_X2 = []
        Archive_X = np.zeros((ArchiveMaxSize, dim + para_size))
        Archive_F = np.ones((ArchiveMaxSize, 5)) * float("inf")
        Alpha_pos = np.zeros(dim + para_size)
        Alpha_f = float("inf") * np.ones(2)
        Beta_pos = np.zeros(dim + para_size)
        Beta_f = float("inf") * np.ones(2)
        Delta_pos = np.zeros(dim + para_size)
        Delta_f = float("inf") * np.ones(2)
        pop = init(population_size, dim, para_size)
        particles_F = get_ObjectFunction(pop, dim, x, y)
        pop, particles_F = sort_pop(pop, particles_F, population_size)
        Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, pop, particles_F, dim)
        front22 = Crowded_distance_rank(Archive_F)
        if Archive_member_no > ArchiveMaxSize:
            Array01_X = np.zeros((ArchiveMaxSize, dim + para_size))
            Array01_F = np.zeros((ArchiveMaxSize, 5))
            for i in range(0, ArchiveMaxSize):
                Array01_X[i] = copy.deepcopy(Archive_X[front22[i]])
                Array01_F[i] = copy.deepcopy(Archive_F[front22[i]])
                pass
            Archive_X = copy.deepcopy(Array01_X)
            Archive_F = copy.deepcopy(Array01_F)
            Archive_member_no = ArchiveMaxSize
            front22 = [i for i in range(0, len(Archive_F))]
            pass

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
            Alpha_pos = copy.deepcopy(Archive_X[front22[0]])
            Alpha_f = copy.deepcopy(Archive_F[front22[0],0:2])
            if len(front22) == 1:
                Beta_pos = copy.deepcopy(Alpha_pos)
                Beta_f = copy.deepcopy(Alpha_f)
                Delta_pos = copy.deepcopy(Alpha_pos)
                Delta_f = copy.deepcopy(Alpha_f)
            if len(front22) == 2:
                Beta_pos = copy.deepcopy(Archive_X[front22[0]])
                Beta_f = copy.deepcopy(Archive_F[front22[0],0:2])
                Delta_pos = copy.deepcopy(Archive_X[front22[1]])
                Delta_f = copy.deepcopy(Archive_F[front22[1],0:2])
                pass
            if len(front22) > 2:
                Beta_pos = copy.deepcopy(Archive_X[front22[1]])
                Beta_f = copy.deepcopy(Archive_F[front22[1],0:2])
                Delta_pos = copy.deepcopy(Archive_X[front22[2]])
                Delta_f = copy.deepcopy(Archive_F[front22[2],0:2])
                if len(front22) > 3 and (
                        (Delta_pos[0:dim] == Beta_pos[0:dim]).all() or (Delta_pos[0:dim] == Alpha_pos[0:dim]).all()):
                    Delta_pos = copy.deepcopy(Archive_X[front22[3]])
                    Delta_f = copy.deepcopy(Archive_F[front22[3],0:2])
                    pass
                pass
            if random.random() < 0.5 * (t / T) and dim < 60:
                index0list = np.where(Alpha_pos[0:dim] == 0)[0]
                if len(index0list) > 0:
                    index0 = random.randint(0, len(index0list) - 1)
                    Alpha_pos[index0list[index0]] = 0 + 1 - Alpha_pos[index0list[index0]]
                    pass
            if t <= 175 or dim >= 1000:
                pop_size = 20
            if t > 175 and dim < 1000:
                pop_size = 10
                pop, particles_F = sort_pop(pop, particles_F, population_size)
            a = 2 - 2 * (t / T)
            for i in range(0, pop_size):
                for j in range(dim + para_size):
                    r1 = random.random()
                    r2 = random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * Alpha_pos[j] - pop[i, j])
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    r1 = random.random()
                    r2 = random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * Beta_pos[j] - pop[i, j])
                    X2 = Beta_pos[j] - A2 * D_beta
                    r1 = random.random()
                    r2 = random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * Delta_pos[j] - pop[i, j])
                    X3 = Delta_pos[j] - A3 * D_delta
                    v = (X1 + X2 + X3) / 3
                    if j < dim or j == dim + 2:
                        p = random.random()
                        V = math.fabs(v / math.sqrt(v * v + 1))
                        if (p < V):
                            if pop[i][j] == 1:
                                pop[i][j] = 0
                            else:
                                pop[i][j] = 1
                        else:
                            pop[i][j] = pop[i][j]
                        pass
                    else:
                        if j == dim:
                            pop[i][j] = np.clip(v, 2 ** (-1), 2 ** (5))
                        if j == dim + 1:
                            pop[i][j] = np.clip(v, 2 ** (-4), 2 ** (5))
                        pass
                pass
            particles_F = get_ObjectFunction(pop, dim, x, y)
            Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, pop, particles_F, dim)
            if t > 175 and dim < 1000:
                pop_new = []
                for i in range(0, Archive_member_no):
                    Arch_new = copy.deepcopy(Archive_X[i])
                    if random.random() < 0.5:
                        Arch_new[dim] = np.clip(Archive_X[i][dim] + (random.random() - 0.5) * Archive_X[i][dim],
                                                2 ** (-1), 2 ** (5))
                    else:
                        Arch_new[dim + 1] = np.clip(
                            Archive_X[i][dim + 1] + (random.random() - 0.5) * Archive_X[i][dim],
                            2 ** (-4), 2 ** (5))
                        pass
                    pop_new.append(Arch_new)
                pop_new_f = get_ObjectFunction(np.array(pop_new), dim, x, y)
                Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, np.array(pop_new),
                                                                        pop_new_f, dim)
            front22 = Crowded_distance_rank(Archive_F)
            if Archive_member_no > ArchiveMaxSize:
                Array01_X = np.zeros((ArchiveMaxSize, dim + para_size))
                Array01_F = np.zeros((ArchiveMaxSize, 5))
                for i in range(0, ArchiveMaxSize):
                    Array01_X[i] = copy.deepcopy(Archive_X[front22[i]])
                    Array01_F[i] = copy.deepcopy(Archive_F[front22[i]])
                    pass
                Archive_X = copy.deepcopy(Array01_X)
                Archive_F = copy.deepcopy(Array01_F)
                Archive_member_no = ArchiveMaxSize
                front22 = [i for i in range(0, len(Archive_F))]
                pass
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