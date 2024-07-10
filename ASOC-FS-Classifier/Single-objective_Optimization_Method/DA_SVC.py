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

def init(population_size, dim, para_size):
    population = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
            population[i][j] = 0 + round(random.random() * (1 - 0))
    parameter = np.zeros((population_size, para_size))
    for i in range(0, parameter.shape[0]):
        parameter[i][0] = (2 ** (-1)) + random.random() * (
                2 ** (5) - 2 ** (-1))
        parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
        parameter[i][2] = 0 + round(random.random() * (1 - 0))
    pop = np.hstack((population, parameter))
    return pop

def da_svc(inputname, runTime):
    inputdata = '../' + inputname + '.csv'
    dataset = pd.read_csv(inputdata, header=None)
    path = '../ExperimentsData/Single/' + inputname + '_DA_' + str(int(time.time())) + '.xlsx'
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
        food_fitness = float("inf")
        food_pos = np.zeros((1, dim + para_size))
        enemy_fitness = -float("inf")
        enemy_pos = np.zeros((1, dim + para_size))
        population = init(population_size, dim, para_size)
        Delta_population = init(population_size, dim, para_size)
        accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, dim, x, y)
        best_fitness = min(fitness)
        best_index = fitness.index(best_fitness)
        best_individual = copy.deepcopy(population[best_index])
        best_accuracy = accuracyrecord[best_index]
        best_precision = precisionrecord[best_index]
        best_recall = recallrecord[best_index]
        best_f1score = f1scorerecord[best_index]
        best_in_history.append(best_fitness)
        for t in range(T):
            w = 0.9 - t * ((0.9 - 0.4) / T)
            if 2 * t <= T:
                pct = 0.1 - (0.2 * t) / T
            else:
                pct = 0
            r1 = random.random()
            s = 2 * r1 * pct
            a = 2 * r1 * pct
            c = 2 * r1 * pct
            f = 2 * r1
            e = pct
            for i in range(len(fitness)):
                if fitness[i] < food_fitness:
                    food_fitness = fitness[i]
                    food_pos = copy.deepcopy(population[i])
            for i in range(len(fitness)):
                if fitness[i] > enemy_fitness:
                    enemy_fitness = fitness[i]
                    enemy_pos = copy.deepcopy(population[i])
            for i in range(population.shape[0]):
                sumS1 = np.zeros(dim + para_size)
                Si = np.zeros(dim + para_size)
                for j in range(population.shape[0]):
                    if j != i:
                        sumS1 = copy.deepcopy(sumS1 + (population[j] - population[i]))
                Si = copy.deepcopy(-sumS1)
                sumS2 = np.zeros(dim + para_size)
                Ai = np.zeros(dim + para_size)
                for j in range(population.shape[0]):
                    if j != i:
                        sumS2 = copy.deepcopy(sumS2 + Delta_population[j])
                Ai = sumS2 / (population.shape[0] - 1)
                Ci = (sumS2 / (population.shape[0] - 1)) - population[i]
                Fi = food_pos - population[i]
                Ei = enemy_pos - population[i]
                Delta_population[i] = (s * Si + a * Ai + c * Ci + f * Fi + e * Ei) + w * Delta_population[i]
                pass
            for i in range(population_size):
                for j in range(dim + para_size):
                    if j < dim or j == dim + 2:
                        v = math.fabs(
                            Delta_population[i][j] / math.sqrt(Delta_population[i][j] ** 2 + 1))
                        p = random.random()
                        if p < v:
                            population[i][j] = 0 + 1 - population[i][j]
                        pass
                    if j == dim:
                        population[i][j] = np.clip(population[i][j] + Delta_population[i][j], 2 ** (-1), 2 ** (5))
                    if j == dim + 1:
                        population[i][j] = np.clip(population[i][j] + Delta_population[i][j], 2 ** (-4), 2 ** (5))
            pass
            accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, dim, x, y)
            if (min(fitness) < best_fitness):
                best_fitness = min(fitness)
                best_index = fitness.index(best_fitness)
                best_individual = copy.deepcopy(population[best_index])
                best_accuracy = accuracyrecord[best_index]
                best_precision = precisionrecord[best_index]
                best_recall = recallrecord[best_index]
                best_f1score = f1scorerecord[best_index]
            best_in_history.append(best_fitness)

        end = time.time()
        worksheet.write(q + 1, 0, best_fitness)
        worksheet.write(q + 1, 1, best_accuracy)
        worksheet.write(q + 1, 2, end - start)
        worksheet.write(q + 1, 3, sum(best_individual[0:dim]) / dim)
        worksheet.write(q + 1, 4, best_precision)
        worksheet.write(q + 1, 5, best_recall)
        worksheet.write(q + 1, 6, best_f1score)
        for j in range(dim):
            worksheet.write(q + 1, 7 + j, best_individual[j])
        for k in range(len(best_in_history)):
            worksheet2.write(q, k, best_in_history[k])
            pass
    workbook.close()
    print("over~")
