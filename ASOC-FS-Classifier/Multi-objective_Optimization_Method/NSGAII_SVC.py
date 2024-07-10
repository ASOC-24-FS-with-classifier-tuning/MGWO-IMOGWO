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
            distance[k] = distance[k] + (values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
                        max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        if max(values2) - min(values2) != 0:
            distance[k] = distance[k] + (values2[sorted1[k - 1]] - values2[sorted1[k + 1]]) / (
                        max(values2) - min(values2))
    return distance, front

def crossover(a, b, dim):
    r1 = np.random.randint(0, dim + 3, int((dim + 3) / 6))  # 范围区间为[low,high）
    for i in r1:
        temp = a[i]
        a[i] = b[i]
        b[i] = temp
        pass
    r3 = np.random.randint(0, dim + 3, int((dim + 3) * 0.1))  # 范围区间为[low,high）
    for i in r3:
        if i < dim:
            if a[i] == 0:
                a[i] = 1
            else:
                a[i] == 0
            if b[i] == 0:
                b[i] = 1
            else:
                b[i] == 0
        if i == dim:
            a[i] = 2 ** (5) + 2 ** (-1) - a[i]
            b[i] = 2 ** (5) + 2 ** (-1) - b[i]
        if i == dim + 1:
            a[i] = 2 ** (5) + 2 ** (-4) - a[i]
            b[i] = 2 ** (5) + 2 ** (-4) - b[i]
        if i == dim + 2:
            a[i] = 0 + 1 - a[i]
            b[i] = 0 + 1 - b[i]
    return a, b

def rank(a, non_dominated_sorted_solution):
    for i in range(len(non_dominated_sorted_solution)):
        if a in non_dominated_sorted_solution[i]:
            return i

def nsga2_svc(name, arr):
    inputname = name
    for mm in arr:
        start1 = time.time()
        inputdata = '../' + inputname + '.csv'
        dataset = pd.read_csv(inputdata, header=None)
        path = '../ExperimentsData/Multi/' + inputname + '_NSGAII_' + str(int(time.time())) + '.xlsx'
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
        max_gen = 200
        gen_no = 0
        dim = x.shape[1]
        obj_no = 2
        para_size = 3
        TMax = np.zeros(max_gen)
        tt = np.zeros(max_gen)
        for i in range(0, max_gen):
            tt[i] = i
        solution = np.zeros((population_size, dim + para_size))
        for i in range(population_size):
            for j in range(dim + para_size):
                if j < dim:
                    solution[i][j] = 0 + round(random.random() * (1 - 0))
                if j == dim:
                    solution[i][j] = (2 ** (-1)) + random.random() * (2 ** (5) - 2 ** (-1))
                if j == dim + 1:
                    solution[i][j] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
                if j == dim + 2:
                    solution[i][j] = 0 + round(random.random() * (1 - 0))
                pass
            pass
        solution2 = []
        Archive_F1 = []
        Archive_F2 = []
        Archive_F3 = []
        Archive_X1 = []
        Archive_X2 = []
        while (gen_no < max_gen):
            start = time.time()
            particles_F = get_ObjectFunction(solution, dim, x, y)
            function1_values = [particles_F[i][0] for i in range(0, len(particles_F))]
            function2_values = [particles_F[i][1] for i in range(0, len(particles_F))]
            non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
            if gen_no == 50:
                for valuez in non_dominated_sorted_solution[0]:
                    Archive_F1.append(particles_F[valuez])
                    Archive_X1.append(solution[valuez])
            if gen_no == 100:
                for valuez in non_dominated_sorted_solution[0]:
                    Archive_F2.append(particles_F[valuez])
                    Archive_X2.append(solution[valuez])
            crowding_distance_values = []
            for i in range(0, len(non_dominated_sorted_solution)):
                cd, front = crowding_distance(function1_values[:], function2_values[:],
                                              non_dominated_sorted_solution[i][:])
                non_dominated_sorted_solution[i] = copy.deepcopy(front)
                crowding_distance_values.append(cd)
            solution2 = copy.deepcopy(solution)
            solution2 = solution2.tolist()
            while (len(solution2) != 2 * population_size):
                a1 = random.randint(0, population_size - 1)
                a2 = random.randint(0, population_size - 1)
                rank1 = rank(a1, non_dominated_sorted_solution)
                rank2 = rank(a2, non_dominated_sorted_solution)
                if rank1 < rank2:
                    parent1 = a1
                if rank1 > rank2:
                    parent1 = a2
                if rank1 == rank2:
                    ii = index_of(a1, non_dominated_sorted_solution[rank1])
                    jj = index_of(a2, non_dominated_sorted_solution[rank1])
                    if crowding_distance_values[rank1][ii] > crowding_distance_values[rank1][jj]:
                        parent1 = a1
                    else:
                        parent1 = a2
                a1 = random.randint(0, population_size - 1)
                a2 = random.randint(0, population_size - 1)
                rank1 = rank(a1, non_dominated_sorted_solution)
                rank2 = rank(a2, non_dominated_sorted_solution)
                if rank1 < rank2:
                    parent2 = a1
                if rank1 > rank2:
                    parent2 = a2
                if rank1 == rank2:
                    ii = index_of(a1, non_dominated_sorted_solution[rank1])
                    jj = index_of(a2, non_dominated_sorted_solution[rank1])
                    if crowding_distance_values[rank1][ii] > crowding_distance_values[rank1][jj]:
                        parent2 = a1
                    else:
                        parent2 = a2
                son1, son2 = crossover(solution[parent1], solution[parent2], dim)
                solution2.append(son1)
                solution2.append(son2)
            solution2 = copy.deepcopy(np.array(solution2))
            solution2 = copy.deepcopy(np.array(list(set([tuple(t) for t in solution2]))))
            if solution2.shape[0] < population_size:
                len1 = solution2.shape[0]
                solution2 = solution2.tolist()
                for i in range(len1, population_size):
                    solution2.append(np.random.randint(0, 2, dim))
            solution2 = copy.deepcopy(np.array(solution2))
            particles_F1 = get_ObjectFunction(solution2, dim, x, y)
            function1_values2 = [particles_F1[i][0] for i in range(0, len(particles_F1))]
            function2_values2 = [particles_F1[i][1] for i in range(0, len(particles_F1))]
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
                                          crowding_distance_values2[i])
                front22.reverse()
                for value in front22:
                    new_solution.append(value)
                    if (len(new_solution) == population_size):
                        break
                if (len(new_solution) == population_size):
                    break
            solution = [solution2[i] for i in new_solution]
            solution = copy.deepcopy(np.array(solution))

            TMax[gen_no] = len(non_dominated_sorted_solution[0])
            print("At the iteration {} there are {} non-dominated solutions in the archive".format(gen_no, len(
                non_dominated_sorted_solution[0])))

            end = time.time()
            gen_no = gen_no + 1
        Archive_F1 = np.array(Archive_F1)
        Archive_F2 = np.array(Archive_F2)
        for valuez in non_dominated_sorted_solution[0]:
            Archive_F3.append(particles_F[valuez])
        Archive_F3 = np.array(Archive_F3)
        for i in range(0, len(Archive_F1[:, 0])):
            worksheet1.write(i + 1, 0, Archive_F1[i][0])
            worksheet1.write(i + 1, 1, Archive_F1[i][1])
            worksheet1.write(i + 1, 2, Archive_F1[i][2])
            worksheet1.write(i + 1, 3, Archive_F1[i][3])
            worksheet1.write(i + 1, 4, Archive_F1[i][4])
            for j in range(0, dim + para_size):
                worksheet1.write(i + 1, 5 + j, Archive_X1[i][j])
        for i in range(0, len(Archive_F2[:, 0])):
            worksheet2.write(i + 1, 0, Archive_F2[i][0])
            worksheet2.write(i + 1, 1, Archive_F2[i][1])
            worksheet2.write(i + 1, 2, Archive_F2[i][2])
            worksheet2.write(i + 1, 3, Archive_F2[i][3])
            worksheet2.write(i + 1, 4, Archive_F2[i][4])
            for j in range(0, dim + para_size):
                worksheet2.write(i + 1, 5 + j, Archive_X2[i][j])
        for i in range(0, len(Archive_F3[:, 0])):
            worksheet3.write(i + 1, 0, Archive_F3[i][0])
            worksheet3.write(i + 1, 1, Archive_F3[i][1])
            worksheet3.write(i + 1, 0, Archive_F3[i][0])
            worksheet3.write(i + 1, 1, Archive_F3[i][1])
            worksheet3.write(i + 1, 2, Archive_F3[i][2])
            worksheet3.write(i + 1, 3, Archive_F3[i][3])
            worksheet3.write(i + 1, 4, Archive_F3[i][4])
        for i in range(0, max_gen):
            worksheet5.write(i, 0, TMax[i])
        end1 = time.time()
        worksheet4.write(1, 0, end1 - start1)
        workbook.close()
        print("over~")