'''
The codes are implemented according to the following paper. If there are any discrepancies, please refer to the original paper.
Gao, Jinrui, et al.
"Information gain ratio-based subfeature grouping empowers particle swarm optimization for feature selection."
Knowledge-Based Systems 286 (2024): 111380.
'''

import os
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import xlsxwriter
import copy

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

def entropy(labels):
    probs = pd.value_counts(labels) / len(labels)
    return sum(np.log(probs) * probs * (-1))


def information_gain(data, idx):
    total_entropy = entropy(data.iloc[:,-1])
    feature_values = data.iloc[:, idx].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = data[data.iloc[:, idx] == value]
        weighted_entropy += len(subset) / len(data) * entropy(subset.iloc[:, -1])
    return total_entropy - weighted_entropy

def information_gain_ratio(data, idx):
    gain = information_gain(data, idx)
    intrinsic_value = entropy(data.iloc[:, idx])
    if intrinsic_value == 0:
        return 0
    return gain / intrinsic_value

def get_pbest(pbest, pbest_fitness, population, fitness):
    for i in range(population.shape[0]):
        if pbest_fitness[i] > fitness[i]:
            pbest_fitness[i] = fitness[i]
            pbest[i] = copy.deepcopy(population[i])
    return pbest, pbest_fitness

def ispso_svc(inputname, runTime):
    inputdata = '../' + inputname + '.csv'
    dataset = pd.read_csv(inputdata, header=None)
    path = '../ExperimentsData/Single/' + inputname + '_ISPSO_' + str(int(time.time())) + '.xlsx'
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    ColName = ["fitness", 'accuracy', 'cpu-runtime','number', 'precision', 'recall', 'f1score']
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
    c1 = 2
    c2 = 2
    Vmax = 6
    for q in range(runTime):
        start = time.time()
        best_in_history = []
        P = (-1 + 2 * np.random.rand(population_size, dim + para_size)) * Vmax
        information_gain_ratios = []
        for i in range(dim):
            gain_ratio = information_gain_ratio(dataset, i)
            information_gain_ratios.append(gain_ratio)
        information_gain_ratios = np.array(information_gain_ratios)
        sorted_indices = np.argsort(information_gain_ratios)[::-1]
        n = len(sorted_indices)
        n_30 = int(0.3 * n)
        F1 = sorted_indices[:n_30]
        F3 = sorted_indices[-n_30:]
        F2 = sorted_indices[n_30:-n_30]
        population = np.zeros((population_size, dim))
        for i in range(population_size):
            random_value = random.randint(1, 3)
            selected_F = []
            if random_value == 1:
                selected_F = F1
            elif random_value == 2:
                selected_F = F2
            elif random_value == 3:
                selected_F = F3
            threshold = np.ones(dim) * 0.5
            for j in range(dim):
                if j in selected_F:
                    threshold[j] = information_gain_ratios[j]
            for j in range(dim):
                population[i][j] = random.random() < threshold[j]
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
        gbest_index = np.argmin(fitness)
        gbest = copy.deepcopy(population[gbest_index])
        best_accuracy = accuracyrecord[gbest_index]
        best_precision = precisionrecord[gbest_index]
        best_recall = recallrecord[gbest_index]
        best_f1score = f1scorerecord[gbest_index]
        best_in_history.append(gbest_fitness)
        PT = np.ones(dim) * 10
        for t in range(T):
            S = np.zeros((population_size, dim+para_size))
            S_accuracyrecord = []
            S_fitnessrecord = []
            S_precisionrecord = []
            S_recallrecord = []
            S_f1scorerecord = []
            w = 0.1
            for i in range(population.shape[0]):
                R = np.zeros(dim)
                for j in range(dim):
                    if PT[j] <= 0:
                        S[i][j] = 0
                    else:
                        r1 = random.random()
                        r2 = random.random()
                        r3 = random.random()
                        R[j] = information_gain_ratios[j]
                        P[i][j] = w * P[i][j] + r1 * (1-R[j]) * (pbest[i][j] -population[i][j])
                        + r2 * R[j] * (gbest[j] - population[i][j])
                        + r3 * (random.random() - population[i][j])
                        if P[i][j] > random.random():
                            S[i][j] = 1 - population[i][j]
                        else:
                            S[i][j] = population[i][j]
                for j in range(para_size):
                    P[i][dim+j] = w * P[i][dim+j] + c1 * random.random() * (pbest[i][dim+j] - population[i][dim+j]) \
                                  + c2 * random.random() * ( gbest[dim+j] - population[i][dim+j])
                    if j == 2:
                        p = random.random()
                        s = 1 / (1 + math.exp((-1) * P[i][dim+j]))
                        if (p < s):
                            S[i][dim+j] = 1
                        else:
                            S[i][dim+j] = 0
                    if j == 0:
                        S[i][dim+j] = np.clip(S[i][dim+j] + P[i][j], 2 ** (-1), 2 ** (5))
                    if j == 1:
                        S[i][dim+j] = np.clip(S[i][dim+j] + P[i][j], 2 ** (-4), 2 ** (5))
                S_accuracy, S_fitness, S_precision, S_recall, S_f1score = get_single_fitness(S[i,:], dim, x, y)
                S_accuracyrecord.append(S_accuracy)
                S_fitnessrecord.append(S_fitness)
                S_precisionrecord.append(S_precision)
                S_recallrecord.append(S_recall)
                S_f1scorerecord.append(S_f1score)
                O_fitness = fitness[i]
                if S_fitness < O_fitness:
                    for j in range(dim):
                        PT[j] = PT[j] - 5 * R[j] *(S[i][j] - population[i][j])
            for j in range(dim):
                PT[j] = PT[j] + 0.05 * R[j]
            new_pop = np.concatenate((S, population), axis=0)
            new_accuracyrecord = S_accuracyrecord + accuracyrecord
            new_fitness = S_fitnessrecord + fitness
            new_precisionrecord = S_precisionrecord + precisionrecord
            new_recallrecord = S_recallrecord + recallrecord
            new_f1scorerecord = S_f1scorerecord + f1scorerecord
            indexs = np.argsort(new_fitness)
            selected_indexs = indexs[0:population_size]
            population = new_pop[selected_indexs]
            accuracyrecord = [new_accuracyrecord[i] for i in selected_indexs]
            fitness = [new_fitness[i] for i in selected_indexs]
            precisionrecord = [new_precisionrecord[i] for i in selected_indexs]
            recallrecord = [new_recallrecord[i] for i in selected_indexs]
            f1scorerecord = [new_f1scorerecord[i] for i in selected_indexs]
            pbest, pbest_fitness = get_pbest(pbest, pbest_fitness, population, fitness)
            if min(fitness) < gbest_fitness:
                gbest_fitness = min(fitness)
                gbest_index = np.argmin(fitness)
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
        worksheet.write(q + 1, 3, sum(gbest[:dim]) / dim)
        worksheet.write(q + 1, 4, best_precision)
        worksheet.write(q + 1, 5, best_recall)
        worksheet.write(q + 1, 6, best_f1score)
        for j in range(dim + para_size):
            worksheet.write(q + 1, 7 + j, gbest[j])
        for k in range(len(best_in_history)):
            worksheet2.write(q, k, best_in_history[k])
    workbook.close()
    print("over~")
