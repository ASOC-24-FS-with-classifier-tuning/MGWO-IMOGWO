'''
The codes are implemented according to the following paper. If there are any discrepancies, please refer to the original paper.
Wang, Ziqian, et al.
"An information-based elite-guided evolutionary algorithm for multi-objective feature selection."
IEEE/CAA Journal of Automatica Sinica 11.1 (2024): 264-266.
'''
import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skfeature.utility.mutual_information import su_calculation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
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
def tournament_selection(dim, su, tournament_size, num):
    selected_indices = []
    for _ in range(num):
        tournament_indices = np.random.choice(dim, tournament_size, replace=False)
        tournament_fitness = [su[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices.append(winner_index)
    return selected_indices

def init(dim, N, para_size, su):
    pop = np.random.randint(0, 2, (N, dim))
    parameter = np.zeros((N, para_size))
    for i in range(0, parameter.shape[0]):
        parameter[i][0] = (2 ** (-1)) + random.random() * (
                2 ** (5) - 2 ** (-1))
        parameter[i][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
        parameter[i][2] = 0 + round(random.random() * (1 - 0))
    pop = np.hstack((pop, parameter))
    number = 200
    if dim >= 200:
        T_e = 0.1 * number
    else:
        T_e = 0.3 * number
    t_e = 20
    R = (0.3 - 0.001) * ((T_e - t_e) / t_e) + 0.001
    R_int = math.ceil(R)
    selected_indices = tournament_selection(dim, su, 2, dim)
    selected_indices = list(set(selected_indices))
    pop[0, selected_indices] = 1
    selected_indices = tournament_selection(dim, -su, 2, R_int)
    selected_indices = list(set(selected_indices))
    pop[0, selected_indices] = 0
    return pop
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2
def mutation(parent, su):
    solution = parent[:-3]
    parameters = parent[-3:]
    selected_indices = np.where(solution == 1)[0]
    unselected_indices = np.where(solution == 0)[0]
    rand = random.random()
    if rand > 0.5:
        if len(unselected_indices) < 2:
            return parent
        selected_features = np.random.choice(unselected_indices, size=2, replace=False)
        if su[selected_features[0]] > su[selected_features[1]]:
            solution[selected_features[0]] = 1
        else:
            solution[selected_features[1]] = 1
    else:
        if len(selected_indices) < 2:
            return parent
        selected_features = np.random.choice(selected_indices, size=2, replace=False)
        if su[selected_features[0]] < su[selected_features[1]]:
            solution[selected_features[0]] = 0
        else:
            solution[selected_features[1]] = 0
    if random.random() < 0.5:
        parameters[0] = 2 ** (5) + 2 ** (-1) - parameters[0]
        parameters[1] = 2 ** (5) + 2 ** (-4) - parameters[1]
        parameters[2] = 0 + 1 - parameters[2]
    parent_new = np.concatenate((solution, parameters))
    return parent_new
def NDSort(PopObj, nSort):
    PopObj, _, Loc = np.unique(PopObj, axis=0, return_index=True, return_inverse=True)
    Table = np.zeros(max(Loc) + 1)
    for i in range(len(Loc)):
        Table[Loc[i]] += 1
    N, M = PopObj.shape
    FrontNo = np.full(N, np.inf)
    MaxFNo = 0
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(Loc)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                Dominated = False
                for j in range(i - 1, -1, -1):
                    if FrontNo[j] == MaxFNo:
                        m = 1
                        while m < M and PopObj[i, m] >= PopObj[j, m]:
                            m += 1
                        Dominated = m >= M
                        if Dominated or M == 2:
                            break
                if not Dominated:
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[Loc]
    return FrontNo, MaxFNo
def CrowdingDistance(PopObj, FrontNo=None):
    N, M = np.shape(PopObj)
    if FrontNo is None:
        FrontNo = np.ones(N)
    CrowdDis = np.zeros(N)
    Fronts = np.setdiff1d(np.unique(FrontNo), [np.inf])
    for f in Fronts:
        Front = np.where(FrontNo == f)[0]
        Fmax = np.max(PopObj[Front, :], axis=0)
        Fmin = np.min(PopObj[Front, :], axis=0)
        for i in range(M):
            Rank = np.argsort(PopObj[Front, i])
            CrowdDis[Front[Rank[0]]] = np.inf
            CrowdDis[Front[Rank[-1]]] = np.inf
            for j in range(1, len(Front) - 1):
                CrowdDis[Front[Rank[j]]] += (PopObj[Front[Rank[j + 1]], i] - PopObj[Front[Rank[j - 1]], i]) / (
                            Fmax[i] - Fmin[i])
    return CrowdDis
def modified_crowding_distance(pop_obj, front_no):
    N, M = pop_obj.shape
    crowd_dis = np.zeros(N)
    unique_values = np.unique(front_no)
    unique_values = unique_values[unique_values != np.inf]
    fronts = unique_values
    for f in fronts:
        front = np.where(front_no == f)[0]
        f_max = np.max(pop_obj[front], axis=0)
        f_min = np.min(pop_obj[front], axis=0)
        for i in range(M):
            rank = np.argsort(pop_obj[front, i])
            crowd_dis[front[rank[0]]] += 1
            for j in range(1, len(front) - 1):
                if np.allclose(f_max[i], f_min[i]):
                    crowd_dis[front[rank[j]]] += 1
                else:
                    crowd_dis[front[rank[j]]] += (pop_obj[front[rank[j + 1]], i] - pop_obj[front[rank[j - 1]], i]) / (
                                f_max[i] - f_min[i])
    return crowd_dis
def find_same_rows_indices(pop_obj):
    row_dict = {}
    for i, row in enumerate(pop_obj):
        row_tuple = tuple(row)
        if row_tuple not in row_dict:
            row_dict[row_tuple] = [i]
        else:
            row_dict[row_tuple].append(i)
    same_rows_indices = [indices for indices in row_dict.values() if len(indices) > 1]
    return same_rows_indices

def iemoea_svc(name, arr):
    inputname = name
    for mm in arr:
        start1 = time.time()
        inputdata = '../' + inputname + '.csv'
        dataset = pd.read_csv(inputdata, header=None)
        path = '../ExperimentsData/Multi/' + inputname + '_IEMOEA_' + str(int(time.time())) + '.xlsx'
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
        para_size = 3
        TMax = np.zeros(max_gen)
        tt = np.zeros(max_gen)
        for i in range(0, max_gen):
            tt[i] = i
        su = np.zeros(dim)
        for i in range(dim):
            fi = x.iloc[:, i]
            su[i] = su_calculation(fi, y)
        population = init(dim, population_size, para_size, su)
        Archive_F1 = []
        Archive_F2 = []
        Archive_X1 = []
        Archive_X2 = []
        temp_fitness = np.ones((population_size, 5)) * float("inf")
        while (gen_no < max_gen):
            start = time.time()
            parents = []
            selected_indexs = np.random.choice(population_size, size=population_size, replace=True)
            for idx in selected_indexs:
                parents.append(population[idx, :])
            parents = np.array(parents)
            OO = []
            for i in range(population_size):
                parent1, parent2 = crossover(population[i,:], parents[i, :])
                parent1 = mutation(parent1, su)
                parent2 = mutation(parent2, su)
                OO.append(parent1)
                OO.append(parent2)
            OO = np.array(OO)
            merged_pop = np.concatenate((OO, population), axis=0)
            merged_pop = np.unique(merged_pop, axis=0)
            rows = merged_pop.shape[0]
            if rows < population_size:
                num = population_size-rows
                rand_solutions = np.random.randint(0, 2, (num, dim))
                rand_parameters = np.zeros((num, para_size))
                for rr in range(num):
                    rand_parameters[rr][0] = (2 ** (-1)) + random.random() * (2 ** (5) - 2 ** (-1))
                    rand_parameters[rr][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
                    rand_parameters[rr][2] = 0 + round(random.random() * (1 - 0))
                rand_solutions = np.hstack((rand_solutions, rand_parameters))
                merged_pop = np.concatenate((merged_pop, rand_solutions), axis=0)
            else:
                particles_F = get_ObjectFunction(merged_pop, dim, x, y)
                fitnesses = particles_F[:,[0,1]]
                fronts,_ = NDSort(fitnesses, population_size)
                cds = modified_crowding_distance(fitnesses, fronts)
                same_rows_indices = find_same_rows_indices(fitnesses)
                del_index = []
                for _, same_rows_tuple in enumerate(same_rows_indices):
                    len_tuple = len(same_rows_tuple)
                    solu_cds = np.zeros(len_tuple)
                    for idx in range(len_tuple):
                        solu_cds[idx] = cds[same_rows_tuple[idx]]
                    max_cd = max(solu_cds)
                    for idx in range(len_tuple):
                        if solu_cds[idx] < max_cd:
                            del_index.append(same_rows_tuple[idx])
                merged_pop = np.delete(merged_pop, del_index, axis=0)
                particles_F = np.delete(particles_F, del_index, axis=0)
                merged_size = merged_pop.shape[0]
                if merged_size > population_size:
                    Objs = particles_F[:, [0, 1]]
                    Front, MaxF = NDSort(Objs, population_size)
                    Selected = Front < MaxF
                    Candidate = Front == MaxF
                    CD = CrowdingDistance(Objs, Front)
                    while np.sum(Selected) < population_size:
                        S = Objs[Selected, 1]
                        IC = np.where(Candidate)[0]
                        ID = np.argsort(CD[IC])[::-1]
                        IC = IC[ID]
                        C = Objs[IC, 1]
                        Div_Vert = np.zeros(len(C))
                        for i in range(len(C)):
                            Div_Vert[i] = np.sum(S == C[i])
                        IDiv_Vert = np.argsort(Div_Vert)
                        IS = IC[IDiv_Vert[0]]
                        Selected[IS] = True
                        Candidate[IS] = False
                    merged_pop = merged_pop[Selected]
                if merged_pop.shape[0] < population_size:
                    num = population_size - merged_pop.shape[0]
                    rand_solutions = np.random.randint(0, 2, (num, dim))
                    rand_parameters = np.zeros((num, para_size))
                    for rr in range(num):
                        rand_parameters[rr][0] = (2 ** (-1)) + random.random() * (2 ** (5) - 2 ** (-1))
                        rand_parameters[rr][1] = (2 ** (-4)) + random.random() * (2 ** (5) - 2 ** (-4))
                        rand_parameters[rr][2] = 0 + round(random.random() * (1 - 0))
                    rand_solutions = np.hstack((rand_solutions, rand_parameters))
                    merged_pop = np.concatenate((merged_pop, rand_solutions), axis=0)
            population = merged_pop

            temp_fitness = get_ObjectFunction(population, dim, x, y)
            if gen_no == 50:
                for q in range(len(population)):
                    Archive_F1.append(temp_fitness[q])
                    Archive_X1.append(population[q])
            if gen_no == 100:
                for q in range(len(population)):
                    Archive_F2.append(temp_fitness[q])
                    Archive_X2.append(population[q])
            end = time.time()
            print(
                "At the iteration {}".format(gen_no))
            gen_no = gen_no + 1

        Archive_F3 = temp_fitness
        Archive_F1 = np.array(Archive_F1)
        Archive_F2 = np.array(Archive_F2)
        Archive_F3 = np.array(Archive_F3)

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
        len3 = Archive_F3.shape[0]
        for i in range(0, len3):
            worksheet3.write(i + 1, 0, Archive_F3[i][0])
            worksheet3.write(i + 1, 1, Archive_F3[i][1])
            worksheet3.write(i + 1, 2, Archive_F3[i][2])
            worksheet3.write(i + 1, 3, Archive_F3[i][3])
            worksheet3.write(i + 1, 4, Archive_F3[i][4])
            for j in range(0, dim + para_size):
                worksheet3.write(i + 1, 5 + j, population[i][j])

        end1 = time.time()
        worksheet4.write(1, 0, end1 - start1)
        workbook.close()
        print("over~")
