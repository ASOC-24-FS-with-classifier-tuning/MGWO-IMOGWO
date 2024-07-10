import os
import random
import math
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
import Improved_factor
import chaos

def get_single_fitness(solution, dim, x, y):
    while(sum(solution[:dim]) < 2):
        solution[:dim] = np.random.randint(0, 2, dim)
    column_use = (solution[:dim] == 1)
    x_test = x.columns[column_use]
    if solution[dim + 2] == 0:
        clf = OneVsOneClassifier(
            SVC(C=solution[dim], gamma=solution[dim + 1], random_state=None, kernel='sigmoid',
                cache_size=2000))
    if solution[dim + 2] == 1:
        clf = OneVsOneClassifier(
            SVC(C=solution[dim], gamma=solution[dim + 1], random_state=None, kernel='rbf',
                cache_size=2000))
    X_train, X_test, y_train, y_test = train_test_split(x[x_test], y, test_size=0.3, stratify=y, random_state=42)
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    fitness_1 = accuracy_score(y_test, ypred)
    fitness_2 = 1.0 * (sum(column_use)) / dim
    fitness_final = 0.99 * (1 - fitness_1) + 0.01 * fitness_2
    precision = precision_score(y_test, ypred, average='weighted', zero_division=0)
    recall = recall_score(y_test, ypred, average='weighted', zero_division=0)
    f1score = f1_score(y_test, ypred, average='weighted')
    return fitness_1, fitness_final, precision, recall, f1score
def get_fitness(population, dim, x, y):
    fitness = []
    accuracyrecord = []
    precisionrecord = []
    recallrecord = []
    f1scorerecord = []
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
        fitness_1 = accuracy_score(y_test, ypred)
        accuracyrecord.append(fitness_1)
        fitness_2 = 1.0 * (sum(column_use)) / dim
        fitness_final = 0.99 * (1 - fitness_1) + 0.01 * fitness_2
        fitness.append(fitness_final)
        precision = precision_score(y_test, ypred, average='weighted', zero_division=0)
        recall = recall_score(y_test, ypred, average='weighted', zero_division=0)
        f1score = f1_score(y_test, ypred, average='weighted')
        precisionrecord.append(precision)
        recallrecord.append(recall)
        f1scorerecord.append(f1score)
    return accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord

def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step
def init_chaos(population_size, dim, para_size):
    pop = np.zeros((population_size, dim + para_size))
    x_list = chaos.chaos(4, 0.7, population_size * (dim + para_size))
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
    return pop

def mgwo_svc(inputname, runTime):
    inputdata = '../' + inputname + '.csv'
    dataset = pd.read_csv(inputdata, header=None)
    path = '../ExperimentsData/Single/' + inputname + '_MGWO_' + str(int(time.time())) + '.xlsx'
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    ColName = ["fitness", 'accuracy', 'cpu-runtime', 'number', 'precision', 'recall', 'f1score']
    for i in range(len(ColName)):
        worksheet.write(0, i, ColName[i])
    worksheet2 = workbook.add_worksheet()
    x = dataset.iloc[:, 0:-1]
    x = pd.DataFrame(StandardScaler().fit_transform(x))
    y = dataset.iloc[:, -1]
    population_size = 20
    T = 200
    dim = x.shape[1]
    para_size = 3
    for q in range(runTime):
        start = time.time()
        best_in_history = []
        best_solution_in_history = []
        Alpha_pos = np.zeros(dim + para_size)
        Alpha_score = float("inf")
        Beta_pos = np.zeros(dim + para_size)
        Beta_score = float("inf")
        Delta_pos = np.zeros(dim + para_size)
        Delta_score = float("inf")
        if dim < 1000:
            population = init_chaos(population_size, dim, para_size)
        else:
            population = Improved_factor.Faure_Initialization(population_size, dim, para_size)
        accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, dim, x, y)
        best_fitness = min(fitness)
        best_index = fitness.index(best_fitness)
        best_individual = copy.deepcopy(population[best_index])
        best_accuracy = accuracyrecord[best_index]
        best_precision = precisionrecord[best_index]
        best_recall = recallrecord[best_index]
        best_f1score = f1scorerecord[best_index]
        best_in_history.append(best_fitness)
        best_solution_in_history.append(best_individual)
        for t in range(T):
            for i in range(population.shape[0]):
                if fitness[i] < Alpha_score:
                    Alpha_score = fitness[i]
                    Alpha_pos = copy.deepcopy(population[i])
                if fitness[i] > Alpha_score and fitness[i] < Beta_score:
                    Beta_score = fitness[i]
                    Beta_pos = copy.deepcopy(population[i])
                if fitness[i] > Alpha_score and fitness[i] > Beta_score and fitness[i] < Delta_score:
                    Delta_score = fitness[i]
                    Delta_pos = copy.deepcopy(population[i])
            a = 2 - 2 * np.log((t / T)**2 + 1)
            num1 = population_size
            ignore_solution_id = []
            for i in range(num1):
                if i in ignore_solution_id:
                    continue
                for j in range(dim + para_size):
                    r1 = random.random()
                    r2 = random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * Alpha_pos[j] - population[i][j])
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    r1 = random.random()
                    r2 = random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * Beta_pos[j] - population[i][j])
                    X2 = Beta_pos[j] - A2 * D_beta
                    r1 = random.random()
                    r2 = random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * Delta_pos[j] - population[i][j])
                    X3 = Delta_pos[j] - A3 * D_delta
                    v = (X1 + X2 + X3) / 3
                    if j < dim or j == dim + 2:
                        p = random.random()
                        V = math.fabs(v / math.sqrt(v * v + 1))
                        if (p < V):
                            if population[i][j] == 1:
                                population[i][j] = 0
                            else:
                                population[i][j] = 1
                        else:
                            population[i][j] = population[i][j]
                        pass
                    else:
                        if j == dim:
                            population[i][j] = np.clip(v, 2 ** (-1), 2 ** (5))
                        if j == dim + 1:
                            population[i][j] = np.clip(v, 2 ** (-4), 2 ** (5))
                        pass
                temp_accuracy, temp_fitness, temp_precision, temp_recall, temp_f1score = get_single_fitness(population[i], dim, x, y)
                accuracyrecord[i] = temp_accuracy
                fitness[i] = temp_fitness
                precisionrecord[i] = temp_precision
                recallrecord[i] = temp_recall
                f1scorerecord[i] = temp_f1score
            if (min(fitness) < best_fitness):
                best_fitness = min(fitness)
                best_index = fitness.index(best_fitness)
                best_individual = copy.deepcopy(population[best_index])
                best_accuracy = accuracyrecord[best_index]
                best_precision = precisionrecord[best_index]
                best_recall = recallrecord[best_index]
                best_f1score = f1scorerecord[best_index]
            best_in_history.append(best_fitness)
            best_solution_in_history.append(best_individual)

        end = time.time()
        worksheet.write(q + 1, 0, best_fitness)
        worksheet.write(q + 1, 1, best_accuracy)
        worksheet.write(q + 1, 2, end - start)
        worksheet.write(q + 1, 3, sum(best_individual[:dim]) / dim)
        worksheet.write(q + 1, 4, best_precision)
        worksheet.write(q + 1, 5, best_recall)
        worksheet.write(q + 1, 6, best_f1score)
        for j in range(dim+para_size):
            worksheet.write(q + 1, 7 + j, best_individual[j])
        for k in range(len(best_in_history)):
            worksheet2.write(q, k, best_in_history[k])
    workbook.close()
    print("over~")
