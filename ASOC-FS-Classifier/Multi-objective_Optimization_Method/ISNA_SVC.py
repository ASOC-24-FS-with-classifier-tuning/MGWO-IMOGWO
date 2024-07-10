'''
The codes are implemented according to the following paper. If there are any discrepancies, please refer to the original paper.
Wang, Ziqian, et al.
"Information-theory-based nondominated sorting ant colony optimization for multiobjective feature selection in classification."
IEEE Transactions on Cybernetics 53.8 (2022): 5276-5289.
'''
import os
import random
import math
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skfeature.utility.entropy_estimators import *
from skfeature.utility.mutual_information import information_gain
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import xlsxwriter
import copy
from sklearn.svm import SVC
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore", category=UserWarning)

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
    for i in range(num_rows):
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

def handleFullArchive(Archive_X, Archive_F, ArchiveMaxSize, dim):
    function1_values = [Archive_F[i][0] for i in range(0, len(Archive_F))]
    function2_values = [Archive_F[i][1] for i in range(0, len(Archive_F))]
    non_dominated_sorted_solution = [i for i in range(0, len(Archive_F))]
    cd, front = crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution)
    front22 = sort_by_values1(front, cd)
    front22.reverse()
    Array01_X = np.zeros((ArchiveMaxSize, dim+3))
    Array01_F = np.zeros((ArchiveMaxSize, 5))
    for i in range(0, ArchiveMaxSize):
        Array01_X[i] = copy.deepcopy(Archive_X[front22[i]])
        Array01_F[i] = copy.deepcopy(Archive_F[front22[i]])
        pass
    Archive_X = copy.deepcopy(Array01_X)
    Archive_F = copy.deepcopy(Array01_F)
    Archive_member_no = ArchiveMaxSize
    Archive_mem_ranks = RankingProcess(Archive_F,2)
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

def init(population_size, dim, para_size):
    population = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
            population[i][j] = 0 + round(random.random() * (1 - 0))
    parameter = np.zeros((population_size, para_size))
    for i in range(0, population_size):
        parameter[i][0] = (2 ** (-1)) + random.random() * (
                2 ** (5) - 2 ** (-1))
        parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
        parameter[i][2] = 0 + round(random.random() * (1 - 0))
    pop = np.hstack((population, parameter))
    return pop

def symmetric_uncertainty(feature_num, dataset, Psi, gains):
    dataset = np.array(dataset)
    visibility = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    for i in range(feature_num):
        for j in range(feature_num):
            fi = dataset[:, i]
            fj = dataset[:, j]
            su = SU(fi, fj, gains[i, j])
            visibility[0, i, j] = su
            visibility[1, i, j] = 1 - su
            visibility[2, i, j] = su
            visibility[3, i, j] = 1 - su
    Psi = np.tile(Psi, (feature_num, 1))
    visibility[0, :, :] = np.multiply(visibility[0, :, :], (1 - Psi))
    visibility[1, :, :] = np.multiply(visibility[1, :, :], Psi)
    visibility[2, :, :] = np.multiply(visibility[2, :, :], (1 - Psi))
    visibility[3, :, :] = np.multiply(visibility[3, :, :], Psi)
    return visibility

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
    if t2 + t3 == 0:
        su = 0
    else:
        su = 2.0 * gain / (t2 + t3)
    return su

def init_improved(X, y, gains):
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    t1 = np.zeros(n_features)
    t2 = np.zeros(n_features)
    beta = 1.0 / (n_features-1)
    t = []
    for i in range(n_features):
        f = X[:, i]
        t1[i] = information_gain(f, y)
        t2[i] = sum(gains[i, :])
        temp = t1[i] - beta * t2[i]
        t.append(temp)
    return t

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
    temp = np.sum(pheremones[0, :, cur_features] + pheremones[2, :, cur_features], axis=1) / np.sum(pheremones[0, :, cur_features] + pheremones[1, :, cur_features] + pheremones[2, :, cur_features] + pheremones[3, :, cur_features], axis=1)  # N
    rand = np.random.random_sample(ant_num)
    road_map[np.arange(ant_num), cur_features] = (rand >= temp).astype(int)  # N*dim
    for j in range(1, feature_num):
        mask = road_map[np.arange(ant_num), pointer[:, j - 1]] == 1   # N
        indexs = 2 * mask.astype(int)   # N
        nominator = np.hstack((indx[indexs, pointer[:, j - 1], :], indx[indexs + 1, pointer[:, j - 1], :]))   # N*(2*dim)
        denominator = np.sum(nominator, axis=1)  # N
        len_nom = len(nominator[0,:])   # dim
        denominator[denominator == 0] = len_nom    # 表示某个解
        nominator[denominator == 0, :] = np.ones(len_nom)
        probability = np.divide(nominator, denominator[:, np.newaxis])   # 长度为 N*dim
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
    err = fitnesses[:,0]
    err_min = min(err)
    idx = np.argmax(err == err_min)
    num = fitnesses[idx,1]
    fit = err_min + num
    change_pheremones = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)
    for i in range(0, len(iter_best_road)-3):
        if(iter_best_road[i] == 0):
            change_pheremones[0, :, i] = 1
            change_pheremones[2, :, i] = 1
        else:
            change_pheremones[1, :, i] = 1
            change_pheremones[3, :, i] = 1

    change_pheremones = (1/(fit)) * change_pheremones

    pheremones= pheremones + change_pheremones
    pheremones = np.where(pheremones > Max_T, Max_T, pheremones)
    pheremones = np.where(pheremones < Min_T, Min_T, pheremones)

    return pheremones

def isna_svc(name, arr):
    inputname = name
    for mm in arr:
        start1 = time.time()
        inputdata = '../' + inputname + '.csv'
        dataset = pd.read_csv(inputdata, header=None)
        path = '../ExperimentsData/Multi/' + inputname + '_ISNA_' + str(int(time.time())) + '.xlsx'
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
        Archive_X = np.zeros((ArchiveMaxSize, dim + para_size))
        Archive_F = np.ones((ArchiveMaxSize, 5)) * float("inf")
        Archive_F1 = []
        Archive_F2 = []
        Archive_X1 = []
        Archive_X2 = []
        if dim <= 147:
            alpha = 0.9
            beta = 0.1
            rou = 0.7
        elif dim > 300:
            alpha = 1
            beta = 0.7
            rou = 0.3
        else:
            alpha = 0.7
            beta = 0.1
            rou = 0.5
        Min_T = 0.1
        Max_T = 6
        pop = init(population_size, dim, para_size)
        pheremones = (np.ones(dim * dim * 4, dtype="float64") * 0.5).reshape(4, dim, dim)
        init_time = time.time()
        particles_F = get_ObjectFunction(pop, dim, x, y)
        Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, pop, particles_F, dim)
        if Archive_member_no > ArchiveMaxSize:
            Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = handleFullArchive(Archive_X, Archive_F, ArchiveMaxSize, dim)
            pass
        else:
            Archive_mem_ranks = RankingProcess(Archive_F, obj_no)

        obj_achieve_time = time.time()
        gains = Calculate_Information_Gain(x)
        Psi = init_improved(x, y, gains)
        min_Psi = min(Psi)
        max_Psi = max(Psi)
        Psi = (Psi - min_Psi) / (max_Psi - min_Psi)
        visibility = symmetric_uncertainty(dim, x, Psi, gains)

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
            pop_start = time.time()
            pop = baco_road_selection(pheremones, visibility, alpha, beta, population_size, dim, pop)
            pop_end = time.time()
            particles_F = get_ObjectFunction(pop, dim, x, y)
            Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, pop, particles_F, dim)
            if Archive_member_no > ArchiveMaxSize:
                Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = handleFullArchive(Archive_X, Archive_F, ArchiveMaxSize, dim)
                pass
            else:
                Archive_mem_ranks = RankingProcess(Archive_F, obj_no)
            update_time = time.time()
            index = RouletteWheelSelection(1 / Archive_mem_ranks)
            if index == -1:
                index = 0
            iter_best = copy.deepcopy(Archive_X[index])
            pheremones = trial_update(particles_F, pheremones, Min_T, Max_T, rou, iter_best, dim)
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