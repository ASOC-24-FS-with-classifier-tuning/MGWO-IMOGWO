'''
The codes are implemented according to the following paper. If there are any discrepancies, please refer to the original paper.
Wang, Ziqian, et al.
"Symmetric uncertainty-incorporated probabilistic sequence-based ant colony optimization for feature selection in classification."
Knowledge-Based Systems 256 (2022): 109874.
'''
import os
import random
import warnings
import numpy as np
import pandas as pd
from skfeature.utility.entropy_estimators import entropyd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import xlsxwriter
import copy
from skfeature.utility.mutual_information import su_calculation, information_gain
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

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
def Calculate_Information_Gain(X):
    X = np.array(X)
    dim = X.shape[1]
    gains = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            gains[j][i] = gains[i][j] = information_gain(X[:, i], X[:, j])
    return gains
def SU(f1, f2, gain):
    t2 = entropyd(f1)
    t3 = entropyd(f2)
    if t2+t3 == 0:
        su = 0
    else:
        su = 2.0 * gain / (t2 + t3)
    return su

def symmetric_uncertainty(dataset, gains):
    dataset = np.array(dataset)
    feature_num = dataset.shape[1]
    visibility = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    for i in range(feature_num):
        for j in range(feature_num):
            fi = dataset[:, i]
            fj = dataset[:, j]
            su = SU(fi, fj, gains[i,j])
            visibility[0, i, j] = su
            visibility[1, i, j] = 1-su
            visibility[2, i, j] = su
            visibility[3, i, j] = 1-su
    return visibility

def pick_next_location(probs, feature_num):
    r = np.random.random_sample(size=probs.shape[0])[:, np.newaxis]
    cumulative_probs = np.cumsum(probs, axis=1)
    indices = np.argmax(r < cumulative_probs, axis=1)
    zero_or_ones = (indices >= feature_num).astype(int)
    indices %= feature_num
    return indices, zero_or_ones

def baco_road_selection(pheremones, visibility, alpha, beta, ant_num, feature_num, pop):
    road_map = np.zeros((ant_num, feature_num), dtype=np.int64)
    pointer = np.zeros((ant_num, feature_num), dtype=np.int64)
    parameters = pop[:, -3:]
    indx = np.multiply(np.power(pheremones, alpha), np.power(visibility, beta))
    cur_features = np.random.randint(0, feature_num, ant_num)
    pointer[:, 0] = cur_features
    temp = np.sum(pheremones[0, :, cur_features] + pheremones[2, :, cur_features], axis=1) / np.sum(pheremones[0, :, cur_features] + pheremones[1, :, cur_features] + pheremones[2, :, cur_features] + pheremones[3, :, cur_features], axis=1)
    rand = np.random.random_sample(ant_num)
    road_map[np.arange(ant_num), cur_features] = (rand >= temp).astype(int)
    for j in range(1, feature_num):
        mask = road_map[np.arange(ant_num), pointer[:, j - 1]] == 1
        indexs = 2 * mask.astype(int)
        nominator = np.hstack((indx[indexs, pointer[:, j - 1], :], indx[indexs + 1, pointer[:, j - 1], :]))
        denominator = np.sum(nominator, axis=1)
        len_nom = len(nominator[0,:])
        denominator[denominator == 0] = len_nom
        nominator[denominator == 0, :] = np.ones(len_nom)
        probability = np.divide(nominator, denominator[:, np.newaxis])
        (selected_feature_indx, zero_or_one) = pick_next_location(probability, feature_num)
        pointer[:, j] = selected_feature_indx
        road_map[np.arange(ant_num), selected_feature_indx] = zero_or_one
        indx[:, :, pointer[:, j]] = 0
    mask = np.random.random_sample(ant_num) < 0.5
    parameters[mask, 0] = 2 ** 5 + 2 ** (-1) - parameters[mask, 0]
    parameters[mask, 1] = 2 ** 5 + 2 ** (-4) - parameters[mask, 1]
    parameters[mask, 2] = 0 + 1 - parameters[mask, 2]
    population = np.hstack((road_map, parameters))
    return population

def trial_update(fitnesses, pheremones, Min_T, Max_T, rou, iter_best_road, feature_num):
    pheremones= (1-rou) * pheremones
    min_fit = min(fitnesses)
    change_pheremones = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    for i in range(0, len(iter_best_road)-3):
        if(iter_best_road[i] == 0):
            change_pheremones[0, :, i] = 1
            change_pheremones[2, :, i] = 1
        else:
            change_pheremones[1, :, i] = 1
            change_pheremones[3, :, i] = 1
    if(min_fit == 0):
        change_pheremones = (1/(min_fit+0.001)) * change_pheremones
    else:
        change_pheremones = (1/(min_fit)) * change_pheremones
    pheremones= pheremones + change_pheremones
    pheremones = np.where(pheremones > Max_T, Max_T, pheremones)
    pheremones = np.where(pheremones < Min_T, Min_T, pheremones)
    return pheremones

def spaco_svc(inputname, runTime):
    inputdata = '../' + inputname + '.csv'
    dataset = pd.read_csv(inputdata, header=None)
    path = '../ExperimentsData/Single/' + inputname + '_SPACO_' + str(int(time.time())) + '.xlsx'
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
    alpha = 0.7
    beta = 0.1
    rou = 0.5
    Min_T = 0.1
    Max_T = 6
    for q in range(runTime):
        start = time.time()
        best_in_history = []
        road_map = np.random.randint(2, size=population_size * dim).reshape((population_size, dim))
        parameter = np.zeros((population_size, para_size))
        for i in range(0, population_size):
            parameter[i][0] = (2 ** (-1)) + random.random() * (
                    2 ** (5) - 2 ** (-1))
            parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
            parameter[i][2] = 0 + round(random.random() * (1 - 0))
        road_map = np.hstack((road_map, parameter))
        pheremones = (np.ones(dim * dim * 4, dtype="float64") * 0.5).reshape(4, dim, dim)
        accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(road_map, dim, x, y)
        best_fitness = min(fitness)
        best_index = fitness.index(best_fitness)
        best_individual = copy.deepcopy(road_map[best_index])
        best_accuracy = accuracyrecord[best_index]
        best_precision = precisionrecord[best_index]
        best_recall = recallrecord[best_index]
        best_f1score = f1scorerecord[best_index]
        best_in_history.append(best_fitness)
        gains = Calculate_Information_Gain(x)
        visibility = symmetric_uncertainty(x, gains)
        for t in range(T):
            road_map = baco_road_selection(pheremones, visibility, alpha, beta, population_size, dim, road_map)
            accuracyrecord, fitness, precisionrecord, recallrecord, f1scorerecord = get_fitness(road_map, dim, x, y)

            if (min(fitness) < best_fitness):
                best_fitness = min(fitness)
                best_index = fitness.index(best_fitness)
                best_individual = copy.deepcopy(road_map[best_index])
                best_accuracy = accuracyrecord[best_index]
                best_precision = precisionrecord[best_index]
                best_recall = recallrecord[best_index]
                best_f1score = f1scorerecord[best_index]
                best_in_history.append(best_fitness)
            best_in_history.append(best_fitness)
            iter_best = road_map[fitness.index(min(fitness)),:]
            pheremones = trial_update(fitness, pheremones, Min_T, Max_T, rou, iter_best, dim)

        end = time.time()
        worksheet.write(q + 1, 0, best_fitness)
        worksheet.write(q + 1, 1, best_accuracy)
        worksheet.write(q + 1, 2, end - start)
        worksheet.write(q + 1, 3, sum(best_individual)/dim)
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
