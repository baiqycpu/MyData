from __future__ import print_function
from gurobipy import *
import pandas as pd
import numpy as np


class Data:
    """Define the data to be used"""
    repairnode_num = 0
    repair_time_LB = []
    repair_time_UB = []
    repair_time_average = []
    repair_time_MAD = []
    demandnode = []
    repairnode = []
"""........"""
    crew_num = 0
    current_starttime0=0
    current_starttime1=0
    current_starttime2=0


def readData(data):
    data.demandnode =  pd.read_excel(r'cede_location\demandnode_CX.xlsx', header=None)


    data.repairnode = "Input the remaining damaged nodes after the first step repaired "
    data.original_repairnode = pd.read_excel(r'cede_location\repairnode_CX.xlsx', header=None)
    data.delet_repairnode = "Damaged nodes that have been repaired"
    data.current_starttime0 = "Departure time of repair crew1"

    data.current_starttime1  ="Departure time of repair crew2"
    data.current_starttime2=  "Departure time of repair crew3"




    index = [data.original_repairnode.index(x) for x in data.delet_repairnode if x in data.original_repairnode]
    boolz = np.ones(len(data.original_repairnode), dtype=bool)
    boolz[index] = False

    data.repairnode = data.delet_repairnode[-3:] + data.repairnode

    data.repairnode_num = len(data.repairnode)

    data.traveltime_full = pd.read_excel(r'codelocation\traveltime_full_CX.xlsx', header=None)
    data.traveltime_part = pd.read_excel(r'codelocation\traveltime_part_CX.xlsx',
                                         header=None)


    for i in range(len(data.delet_repairnode)):
        data.traveltime_part.iloc[data.delet_repairnode[i], :] = data.traveltime_full.iloc[data.delet_repairnode[i], :]
        data.traveltime_part.iloc[:, data.delet_repairnode[i]] = data.traveltime_full.iloc[:, data.delet_repairnode[i]]

    repairtime_average = pd.read_excel(r'codelocation\repairtime_average_CX.xlsx', header=None)
    data.repair_time_average = np.insert(repairtime_average, 0, [0])
    data.repair_time_average = data.repair_time_average[boolz]
    data.repair_time_average = np.insert(data.repair_time_average, 0, [0])
    data.repair_time_average = np.insert(data.repair_time_average, 0, [0])

    repairtime_LB = pd.read_excel(r'codelocation\repairtime_LB_CX.xlsx', header=None)
    data.repair_time_LB = np.insert(repairtime_LB, 0, [0])
    data.repair_time_LB = data.repair_time_LB[boolz]
    data.repair_time_LB = np.insert(data.repair_time_LB, 0, [0])
    data.repair_time_LB = np.insert(data.repair_time_LB, 0, [0])

    repairtime_UB = pd.read_excel(r'codelocation\repairtime_UB_CX.xlsx', header=None)
    data.repair_time_UB = np.insert(repairtime_UB, 0, [0])
    data.repair_time_UB = data.repair_time_UB[boolz]
    data.repair_time_UB = np.insert(data.repair_time_UB, 0, [0])
""".........."""


def dijkstra(adj_matrix, start, end):

    num_vertices = len(adj_matrix)


    distances = [float('infinity')] * num_vertices
    distances[start] = 0  #


    path = [None] * num_vertices


    visited = set()

    while len(visited) < num_vertices:

        min_distance = float('infinity')
        min_vertex = None
        for v in range(num_vertices):
            if v not in visited and distances[v] < min_distance:
                min_distance = distances[v]
                min_vertex = v

        if min_vertex is None:
            break


        visited.add(min_vertex)


        for neighbor in range(num_vertices):
            if neighbor not in visited and adj_matrix[min_vertex][neighbor] > 0:
                new_distance = distances[min_vertex] + adj_matrix[min_vertex][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    path[neighbor] = min_vertex


    shortest_path = []
    current_vertex = end
    while current_vertex is not None:
        shortest_path.append(current_vertex)
        current_vertex = path[current_vertex]
    shortest_path.reverse()


    return shortest_path, distances[end]


import heapq


def dijkstra2(df, start, end, blocked_nodes):

    num_nodes = df.shape[0]


    distances = np.full(num_nodes, np.inf)
    predecessors = np.full(num_nodes, -1, dtype=int)
    visited = np.zeros(num_nodes, dtype=bool)

    distances[start] = 0
    heap = [(0, start)]

    while heap:
        current_dist, u = heapq.heappop(heap)
        if u == end:
            break
        if visited[u] or (u in blocked_nodes and u != start and u != end):
            continue
        visited[u] = True

        # 遍历所有邻居
        for v in range(num_nodes):
            if visited[v] or np.isinf(df.iloc[u, v]) or (v in blocked_nodes and v != start and v != end):
                continue

            new_dist = current_dist + df.iloc[u, v]

            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(heap, (new_dist, v))

    # 回溯路径
    path = []
    current = end
    if distances[end] != np.inf:
        while current != -1:
            path.append(current)
            current = predecessors[current]
        path.reverse()

    return path, distances[end]


data = Data()
readData(data)



add0demandnode = np.hstack([0, data.demandnode])
a = []
b = []
soledemand_index = []
buso = []
for i in range(1, len(add0demandnode)):
    _, aa = dijkstra(data.traveltime_part, 0, add0demandnode[i])
    a.append(aa)
    if aa == float("inf"):
        soledemand_index.append(i)

soledemand_num = len(soledemand_index)

distance = np.empty((data.repairnode_num, data.repairnode_num))

for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        _, distance[i][j] = dijkstra2(data.traveltime_full, data.repairnode[i], data.repairnode[j], data.repairnode[2:])

for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        if distance[i][j] == float("inf"):
            distance[i][j] = data.M / 10

model = Model("MP-P1")
u = model.addVars(data.repairnode_num, data.crew_num, vtype=GRB.CONTINUOUS, name='u')

h = model.addVars(soledemand_num, vtype=GRB.CONTINUOUS, name="h")
hf = model.addVars(soledemand_num, vtype=GRB.CONTINUOUS, name="hf")
F = model.addVars(data.repairnode_num, data.crew_num, vtype=GRB.CONTINUOUS, name="F")
R = model.addVars(data.repairnode_num, data.repairnode_num, data.crew_num, vtype=GRB.BINARY, name="R")
N = model.addVars(data.repairnode_num, data.repairnode_num, data.repairnode_num, vtype=GRB.BINARY, name="N")


dual_lamda1 = model.addVar(vtype=GRB.CONTINUOUS, name="dual_lamda1")
dual_kesei1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei1")
dual_kesei2 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei2")
dual_fai11 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_fai11")
""".........."""

dual_lamda1_1 = model.addVar(vtype=GRB.CONTINUOUS, name="dual_lamda1_1")
dual_kesei1_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei1_1")
dual_kesei2_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei2_1")
"""........."""

z = model.addVars(data.repairnode_num, data.crew_num, vtype=GRB.BINARY, name="z")

FU = model.addVars(data.repairnode_num, data.repairnode_num, data.repairnode_num, data.crew_num, vtype=GRB.BINARY,
                   name="FU")


"""In accordance with the journal's data and code sharing policy, we provide the core implementation of the proposed algorithm. 
Due to proprietary technology limitations, some parts have been omitted."""

model.update()

model.setObjective(quicksum(h[i] for i in range(soledemand_num)) + quicksum(R[k,m,f] * distance[k][m] for k in range(data.repairnode_num) for m in range(data.repairnode_num) for f in range(data.crew_num))/1000)



for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        current_traveltime_part = data.traveltime_part.copy()
        if i != j and i != 0 and j != 1:
            if i != 1 and j != 0:
                current_traveltime_part.iloc[data.repairnode[i], :] = data.traveltime_full.iloc[data.repairnode[i], :]
                current_traveltime_part.iloc[:, data.repairnode[i]] = data.traveltime_full.iloc[:, data.repairnode[i]]
                current_traveltime_part.iloc[data.repairnode[j], :] = data.traveltime_full.iloc[data.repairnode[j], :]
                current_traveltime_part.iloc[:, data.repairnode[j]] = data.traveltime_full.iloc[:, data.repairnode[j]]

                ctoute, AA = dijkstra(current_traveltime_part, data.repairnode[i], data.repairnode[j])
                if AA != float("inf"):
                    BB = set(data.repairnode) & set(ctoute)
                    BB = list(BB)
                    BB.remove(data.repairnode[i])
                    BB.remove(data.repairnode[j])
                    if data.repairnode[0] in BB:
                        BB.remove(data.repairnode[0])
                    if data.repairnode[1] in BB:
                        BB.remove(data.repairnode[1])
                    if data.repairnode[2] in BB:
                        BB.remove(data.repairnode[2])

                    if len(BB) > 0:
                        model.addConstr(N[i, j, data.repairnode.index(BB[0])] == 1)
                    else:
                        for k in range(data.repairnode_num):
                            if i != j and j != k:
                                model.addConstr(N[i, j, k] == 0)
                else:
                    for l in range(data.repairnode_num):
                        if i != j and j != l:
                            current_traveltime_part1 = current_traveltime_part.copy()
                            current_traveltime_part1.iloc[data.repairnode[l], :] = data.traveltime_full.iloc[
                                                                                   data.repairnode[l], :]
                            current_traveltime_part1.iloc[:, data.repairnode[l]] = data.traveltime_full.iloc[:,
                                                                                   data.repairnode[l]]
                            dtoute, CC = dijkstra(current_traveltime_part1, data.repairnode[i], data.repairnode[j])
                            if CC != float("inf"):
                                DD = set(data.repairnode) & set(dtoute)
                                DD.remove(data.repairnode[i])
                                DD.remove(data.repairnode[j])
                                DD.remove(data.repairnode[l])
                                if data.repairnode[0] in DD:
                                    DD.remove(data.repairnode[0])
                                if data.repairnode[1] in DD:
                                    DD.remove(data.repairnode[1])
                                if data.repairnode[2] in DD:
                                    DD.remove(data.repairnode[2])

                                if len(DD) > 0:
                                    model.addConstr(N[i, j, l] == 0)
                                else:
                                    model.addConstr(N[i, j, l] == 1)
                            else:
                                model.addConstr(N[i, j, l] == 0)

for i in range(data.repairnode_num):
    model.addConstr(N[0, 0, i] == 0)


ru = np.empty((data.repairnode_num, soledemand_num))
for i in range(2, data.repairnode_num):
    for j in range(soledemand_num):
        RR1, SS1 = dijkstra(data.traveltime_full, data.repairnode[i], add0demandnode[soledemand_index[j]])
        SET = set(RR1) & set(data.repairnode)
        SET.remove(data.repairnode[i])
        if data.repairnode[0] in SET:
            SET.remove(data.repairnode[0])
        if data.repairnode[1] in SET:
            SET.remove(data.repairnode[1])
        if data.repairnode[2] in SET:
            SET.remove(data.repairnode[2])
        if len(SET) == 0:
            ru[i][j] = 1

        else:
            ru[i][j] = 0

for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        for k in range(data.repairnode_num):
            for f in range(data.crew_num):
                if i != j and j != k and i != k:
                    model.addConstr(F[j, f] >= F[k, f] - (1 - FU[i, j, k, f]) * data.M)


"""Based on the results obtained in step 1, input the departure times of each maintenance team in step 2."""

model.addConstr(F[0, 0] == data.current_starttime0 )
model.addConstr(F[1, 1] == data.current_starttime1 )
model.addConstr(F[2,2] == data.current_starttime2)

for i in range(data.repairnode_num):
    for j in range(3,data.repairnode_num):
        for k in range(3,data.repairnode_num):
            for f in range(data.crew_num):
                if i!=j and j!=k and i!=k:
                    model.addConstr(F[j,f] >= F[k,f] - (1-FU[i,j,k,f]) * data.M)


for k in range(3,data.repairnode_num):
    model.addConstr(quicksum(R[i, k, f] for i in range(data.repairnode_num) for f in range(data.crew_num) if i != k) == 1,"(1)")


for f in range(data.crew_num):
    if f==0:
        start_node=0
    elif f==1:
        start_node=1
    elif f==2:
        start_node=2


    model.addConstr(quicksum(R[start_node, j, f] for j in range(data.repairnode_num) if j != start_node ) == 1)

    model.addConstr(quicksum(R[i, start_node, f] for i in range(data.repairnode_num) if i != start_node) == 1)


    for k in range(3,data.repairnode_num):
        model.addConstr(
            quicksum(R[i, k, f] for i in range(data.repairnode_num) if i != k) ==
            quicksum(R[k, j, f] for j in range(data.repairnode_num) if j != k)
       )

for f in range(data.crew_num):
    for i in range(data.repairnode_num):
        for j in range(data.repairnode_num):
            if (f == 0) and (i==1 or j==1 or i==2 or j==2):
                model.addConstr(R[i, j, f] == 0 )



for f in range(data.crew_num):
    for i in range(data.repairnode_num):
        for j in range(data.repairnode_num):
            if (f==1) and (i==0 or j==0 or i==2 or j==2):
                model.addConstr(R[i,j,f]==0)


for f in range(data.crew_num):
    for i in range(data.repairnode_num):
        for j in range(data.repairnode_num):
            if (f==2) and (i==0 or j==0 or i==1 or j==1):
                model.addConstr(R[i,j,f]==0)



for f in range(data.crew_num):
    for i in range(data.repairnode_num):
        for j in range(3,data.repairnode_num):
            if i != j:
                model.addConstr(u[i, f] - u[j, f] + data.repairnode_num * R[i, j, f] <= data.repairnode_num - 1)

for i in range(3,data.repairnode_num):
    model.addConstr(quicksum(R[i, j, f] for j in range(data.repairnode_num) for f in range(data.crew_num)) <= 1)




"""........."""
"""........."""
"""........."""

model.Params.OutputFlag = 1
model.Params.LogFile = "log_file.txt"
model.optimize()


model.printAttr('X')
print("objective value:", model.getAttr("ObjVal"))


