import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
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
def get_total(fitness):
    total = 0
    for i in range(len(fitness)):
        total += 1.0 / float(fitness[i])
    return total
def selection(population, fitness):
    total_fitness = get_total(fitness)
    new_fitness = []
    for i in range(len(fitness)):
        new_fitness.append((1.0 / fitness[i]) / total_fitness)
    new_fitness = np.cumsum(new_fitness)
    chosen_index = 0
    p = random.random()
    for i in range(new_fitness.shape[0]):
        if (p < new_fitness[i]):
            chosen_index = i
            break
    pass
    parent1 = copy.deepcopy(population[chosen_index])
    chosen_index = 0
    p = random.random()
    for i in range(new_fitness.shape[0]):
        if (p < new_fitness[i]):
            chosen_index = i
            break
    pass
    parent2 = copy.deepcopy(population[chosen_index])
    return parent1, parent2
def crossover(parent1, parent2):
    cpoint = np.random.randint(0, parent1.shape[0])
    temp = copy.deepcopy(parent1[cpoint:parent1.shape[0]])
    parent1[cpoint:parent1.shape[0]] = copy.deepcopy(parent2[cpoint:parent1.shape[0]])
    parent2[cpoint:parent1.shape[0]] = copy.deepcopy(temp)
    return parent1, parent2
def mutation(parent1, parent2, pm):
    r1 = random.random()
    r2 = random.random()
    if (r1 < pm):
        mpoint = np.random.randint(0, parent1.shape[0]-3)
        parent1[mpoint] = 0 + 1 - parent1[mpoint]
        pass
    if (r2 < pm):
        mpoint = np.random.randint(0, parent2.shape[0]-3)
        parent2[mpoint] = 0 + 1 - parent2[mpoint]
        pass
    return parent1, parent2


def ga_svc(inputname, runTime):
    inputdata = '../' + inputname + '.csv'
    dataset = pd.read_csv(inputdata, header=None)
    path = '../ExperimentsData/Single/' + inputname + '_GA_' + str(int(time.time())) + '.xlsx'
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
    T = 200
    population_size = 20
    chromosome_length = x.shape[1]
    para_size = 3
    pm = 0.1
    for q in range(runTime):
        start = time.time()
        best_in_history = []
        population = np.random.randint(0, 2, (population_size, chromosome_length))
        parameter = np.zeros((population_size, para_size))
        for i in range(0, population_size):
            parameter[i][0] = (2 ** (-1)) + random.random() * (
                    2 ** (5) - 2 ** (-1))
            parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
            parameter[i][2] = 0 + round(random.random() * (1 - 0))
        population = np.hstack((population, parameter))
        accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, chromosome_length, x, y)
        best_fitness = min(fitness)
        best_index = fitness.index(best_fitness)
        best_individual = copy.deepcopy(population[best_index])
        best_accuracy = accuracyrecord[best_index]
        best_precision = precisionrecord[best_index]
        best_recall = recallrecord[best_index]
        best_f1score = f1scorerecord[best_index]
        best_in_history.append(best_fitness)
        for t in range(T):
            for i in range(0, int(population_size / 2)):
                parent1, parent2 = selection(population, fitness)
                parent1, parent2 = crossover(parent1, parent2)
                parent1, parent2 = mutation(parent1, parent2, pm)
                population[i] = copy.deepcopy(parent1)
                population[i + int(population_size / 2)] = copy.deepcopy(parent2)
            accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(population, chromosome_length, x, y)
            if (min(fitness) < best_fitness):
                best_fitness = min(fitness)
                best_index = fitness.index(best_fitness)
                best_individual = copy.deepcopy(population[best_index])
                best_accuracy = accuracyrecord[best_index]
                best_precision = precisionrecord[best_index]
                best_recall = recallrecord[best_index]
                best_f1score = f1scorerecord[best_index]
                best_in_history.append(best_fitness)
            best_in_history.append(best_fitness)
        end = time.time()
        worksheet.write(q + 1, 0, best_fitness)
        worksheet.write(q + 1, 1, best_accuracy)
        worksheet.write(q + 1, 2, end - start)
        worksheet.write(q + 1, 3, sum(best_individual) / chromosome_length)
        worksheet.write(q + 1, 4, best_precision)
        worksheet.write(q + 1, 5, best_recall)
        worksheet.write(q + 1, 6, best_f1score)
        for j in range(chromosome_length):
            worksheet.write(q + 1, 7 + j, best_individual[j])
        for k in range(len(best_in_history)):
            worksheet2.write(q, k, best_in_history[k])
    workbook.close()
    print("over~")
