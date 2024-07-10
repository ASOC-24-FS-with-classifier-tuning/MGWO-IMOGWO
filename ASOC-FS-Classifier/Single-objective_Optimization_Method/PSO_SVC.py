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

def get_pbest(pbest, pbest_fitness, population, fitness):
    for i in range(population.shape[0]):
        if pbest_fitness[i] > fitness[i]:
            pbest_fitness[i] = fitness[i]
            pbest[i] = copy.deepcopy(population[i])
    return pbest, pbest_fitness

def pso_svc(inputname, runTime):
    inputdata = '../' + inputname + '.csv'
    dataset = pd.read_csv(inputdata, header=None)
    path = '../ExperimentsData/Single/' + inputname + '_PSO_' + str(int(time.time())) + '.xlsx'
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
    wmax = 0.9
    wmin = 0.4
    c1 = 2
    c2 = 2
    Vmax = 6
    for q in range(runTime):
        start = time.time()
        best_in_history = []
        V = (-1 + 2 * np.random.rand(population_size, dim + para_size)) * Vmax
        population = np.random.randint(0, 2, (population_size, dim))
        parameter = np.zeros((population_size, para_size))
        for i in range(0, population_size):
            parameter[i][0] = (2 ** (-1)) + random.random() * (
                    2 ** (5) - 2 ** (-1))
            parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
            parameter[i][2] = 0 + round(random.random() * (1 - 0))
        population = np.hstack((population, parameter))
        accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, dim, x, y)
        pbest = copy.deepcopy(population)
        pbest_fitness = copy.deepcopy(fitness)
        gbest_fitness = min(fitness)
        gbest_index = fitness.index(gbest_fitness)
        gbest = copy.deepcopy(population[gbest_index])
        best_accuracy = accuracyrecord[gbest_index]
        best_precision = precisionrecord[gbest_index]
        best_recall = recallrecord[gbest_index]
        best_f1score = f1scorerecord[gbest_index]
        best_in_history.append(gbest_fitness)
        for t in range(T):
            w = wmax - (wmax - wmin) * t / T
            for i in range(population.shape[0]):
                for j in range(dim + para_size):
                    V[i][j] = w * V[i][j] + c1 * random.random() * (
                                pbest[i][j] - population[i][j]) + c2 * random.random() * (gbest[j] - population[i][j])
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
            accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, dim, x, y)
            pbest, pbest_fitness = get_pbest(pbest, pbest_fitness, population, fitness)
            if min(fitness) < gbest_fitness:
                gbest_fitness = min(fitness)
                gbest_index = fitness.index(gbest_fitness)
                gbest = copy.deepcopy(population[gbest_index])
                best_accuracy = accuracyrecord[gbest_index]
                best_precision = precisionrecord[gbest_index]
                best_recall = recallrecord[gbest_index]
                best_f1score = f1scorerecord[gbest_index]
            best_in_history.append(gbest_fitness)
        end = time.time()
        worksheet.write(q + 1, 0, gbest_fitness)
        worksheet.write(q + 1, 1, best_accuracy)
        worksheet.write(q + 1, 2, end - start)
        worksheet.write(q + 1, 3, sum(gbest) / dim)
        worksheet.write(q + 1, 4, best_precision)
        worksheet.write(q + 1, 5, best_recall)
        worksheet.write(q + 1, 6, best_f1score)
        for j in range(dim):
            worksheet.write(q + 1, 7 + j, gbest[j])
        for k in range(len(best_in_history)):
            worksheet2.write(q, k, best_in_history[k])
    workbook.close()
    print("over~")
