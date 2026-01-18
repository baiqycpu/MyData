from __future__ import print_function
from gurobipy import *
import pandas as pd
import numpy as np
class Data:

    repairnode_num = 0
    repair_time_LB = []
    repair_time_UB = []
    repair_time_average = []
    repair_time_MAD = []
    demandnode = []
    repairnode = []
    traveltime_full = [[]]
    traveltime_part = [[]]
    traveltime_repair = [[]]
    repairtime_average_LB = []
    repairtime_average_UB = []
    M = 0
    crew_num = 0

#Read data
def readData(data):

    """Read data of each instance CX"""

    data.demandnode = pd.read_excel(r'cede_location\demandnode_CX.xlsx', header=None)

    data.repairnode = pd.read_excel(r'cede_location\repairnode_CX.xlsx', header=None)
    data.repairnode_num = "the number of repairnode"
    data.traveltime_full = pd.read_excel(r'cede_location\traveltime_full.xlsx', header=None)
    data.traveltime_part = pd.read_excel(r'cede_location\traveltime_part_CX.xlsx',
                                         header=None)

    repairtime_average = pd.read_excel(r'cede_location\repairtime_average_CX.xlsx', header=None)
    data.repair_time_average = np.insert(repairtime_average, 0, [0])
    repairtime_LB = pd.read_excel(r'cede_location\repairtime_LB_CX.xlsx', header=None)
    data.repair_time_LB = np.insert(repairtime_LB, 0, [0])
    repairtime_UB = pd.read_excel(r'cede_location\repairtime_UB_CX.xlsx', header=None)
    data.repair_time_UB = np.insert(repairtime_UB, 0, [0])
    repairtime_MAD = pd.read_excel(r'cede_location\repairtime_MAD_CX.xlsx', header=None)
    data.repair_time_MAD = np.insert(repairtime_MAD, 0, [0])
    data.M =100000
    data.crew_num = "repair crew number"

    repairtime_average_LB = pd.read_excel(r'cede_location\repairtime_average_LB_CX.xlsx', header=None)
    data.repairtime_average_LB = np.insert(repairtime_average_LB, 0, [0])
    repairtime_average_UB = pd.read_excel(r'cede_location\repairtime_average_UB_CX.xlsx', header=None)
    data.repairtime_average_UB = np.insert(repairtime_average_UB, 0, [0])



def dijkstra(adj_matrix, start, end):

    num_vertices = len(adj_matrix)


    distances = [float('infinity')] * num_vertices
    distances[start] = 0


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


    return  shortest_path,distances[end]



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

        for v in range(num_nodes):
            if visited[v] or np.isinf(df.iloc[u, v]) or (v in blocked_nodes and v != start and v != end):
                continue

            new_dist = current_dist + df.iloc[u, v]

            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(heap, (new_dist, v))

    path = []
    current = end
    if distances[end] != np.inf:
        while current != -1:
            path.append(current)
            current = predecessors[current]
        path.reverse()

    return  path,distances[end]


data = Data()
readData(data)

add0demandnode = np.hstack([0, data.demandnode])
a=[]
b=[]
soledemand_index = []
buso = []
for i in range(1,len(add0demandnode)):
     _,aa= dijkstra(data.traveltime_part,0,add0demandnode[i])
     a.append(aa)
     if aa == float("inf"):
         soledemand_index.append(i)


soledemand_num = len(soledemand_index)



distance = np.empty((data.repairnode_num,data.repairnode_num))

for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        _,distance[i][j] = dijkstra2(data.traveltime_full, data.repairnode[i], data.repairnode[j],data.repairnode[1:])

for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        if distance[i][j] == float("inf"):
            distance[i][j] = data.M / 10


model = Model("MP-P1")
u = model.addVars(data.repairnode_num,data.crew_num, vtype=GRB.CONTINUOUS, name='u')

h = model.addVars(soledemand_num, vtype=GRB.CONTINUOUS, name = "h")
hf = model.addVars(soledemand_num, vtype=GRB.CONTINUOUS, name = "hf")
F = model.addVars(data.repairnode_num,data.crew_num,vtype=GRB.CONTINUOUS,name="F")
R = model.addVars(data.repairnode_num, data.repairnode_num, data.crew_num,vtype = GRB.BINARY, name = "R")
N = model.addVars(data.repairnode_num, data.repairnode_num, data.repairnode_num, vtype= GRB.BINARY, name = "N")


dual_lamda1 = model.addVar(vtype=GRB.CONTINUOUS, name="dual_lamda1")
dual_kesei1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei1")
dual_kesei2 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei2")
dual_fai11 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_fai11")
dual_fai12 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_fai12")
dual_bigseita1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_bigseita1")

dual_lamda1_1 = model.addVar(vtype=GRB.CONTINUOUS, name="dual_lamda1_1")
dual_kesei1_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei1_1")
dual_kesei2_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei2_1")
dual_fai11_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_fai11_1")
dual_fai12_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_fai12_1")
dual_bigseita1_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_bigseita1_1")



z = model.addVars(data.repairnode_num,data.crew_num, vtype=GRB.BINARY, name= "z")

FU = model.addVars(data.repairnode_num,data.repairnode_num,data.repairnode_num,data.crew_num,vtype=GRB.BINARY, name= "FU")


model.update()

model.setObjective(quicksum(h[i] for i in range(soledemand_num)) )



for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        current_traveltime_part = data.traveltime_part.copy()
        if i!=j:

            current_traveltime_part.iloc[data.repairnode[i],:] = data.traveltime_full.iloc[data.repairnode[i],:]
            current_traveltime_part.iloc[:,data.repairnode[i]] = data.traveltime_full.iloc[:,data.repairnode[i]]
            current_traveltime_part.iloc[data.repairnode[j], :] = data.traveltime_full.iloc[data.repairnode[j], :]
            current_traveltime_part.iloc[:, data.repairnode[j]] = data.traveltime_full.iloc[:, data.repairnode[j]]

            ctoute, AA = dijkstra(current_traveltime_part, data.repairnode[i], data.repairnode[j])
            if AA!=float("inf"):
                BB = set(data.repairnode) & set(ctoute)
                BB = list(BB)
                BB.remove(data.repairnode[i])
                BB.remove(data.repairnode[j])
                if data.repairnode[0] in BB:
                    BB.remove(data.repairnode[0])

                if len(BB)>0:
                    model.addConstr(N[i,j,data.repairnode.index(BB[0])] ==1)
                else:
                    for k in range(data.repairnode_num):
                        if i!=j and j!=k:
                            model.addConstr(N[i,j,k]==0)
            else:
                for l in range(data.repairnode_num):
                    if i!=j and j!=l:
                        current_traveltime_part1 = current_traveltime_part.copy()
                        current_traveltime_part1.iloc[data.repairnode[l], :] = data.traveltime_full.iloc[ data.repairnode[l], :]
                        current_traveltime_part1.iloc[:, data.repairnode[l]] = data.traveltime_full.iloc[:, data.repairnode[l]]
                        dtoute, CC = dijkstra(current_traveltime_part1, data.repairnode[i], data.repairnode[j])
                        if CC != float("inf"):
                            DD = set(data.repairnode) & set(dtoute)
                            DD = list(DD)
                            DD.remove(data.repairnode[i])
                            DD.remove(data.repairnode[j])
                            DD.remove(data.repairnode[l])
                            if data.repairnode[0] in DD:
                                DD.remove(data.repairnode[0])

                            if len(DD)>0:
                                model.addConstr(N[i,j,l]==0)
                            else:
                                model.addConstr(N[i,j,l]==1)
                        else:
                            model.addConstr(N[i,j,l]==0)
for i in range(data.repairnode_num):
    model.addConstr(N[0, 0, i]==0)


ru= np.empty((data.repairnode_num,soledemand_num))
for i in range(1,data.repairnode_num):
    for j in range(soledemand_num):
        RR1,SS1 = dijkstra(data.traveltime_full,data.repairnode[i],add0demandnode[soledemand_index[j]])
        SET = set(RR1) & set(data.repairnode)
        SET = list(SET)
        SET.remove(data.repairnode[i])
        if 0 in SET:
            SET.remove(0)
        if len(SET)==0:
            ru[i][j] = 1

        else:
            ru[i][j] = 0



for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        for k in range(data.repairnode_num):
            for f in range(data.crew_num):
                if i!=j and j!=k and i!=k:
                    model.addConstr(F[j,f] >= F[k,f] - (1-FU[i,j,k,f]) * data.M, "constraint")


for f in range(data.crew_num):
    model.addConstr(F[0,f] ==0, "constraint")


for k in range(1,data.repairnode_num):
    for f in range(data.crew_num):
        model.addConstr(quicksum(R[i, k, f] for i in range(data.repairnode_num) if i!= k) == quicksum( R[k, j, f] for j in range(data.repairnode_num) if k!=j))
        model.addConstr(quicksum(R[i, k, f] for i in range(data.repairnode_num) if i!=k) == z[k,f])


for i in range(1,data.repairnode_num):
    for f in range(data.crew_num):
        model.addConstr(F[i,f] <= data.M * z[i,f])

for i in range(1,data.repairnode_num):
   model.addConstr(quicksum(R[i,j,f] for j in range(data.repairnode_num) for f in range(data.crew_num) if i!=j) ==1)


for f in range(data.crew_num):
    model.addConstr(quicksum(R[0,j,f] for j in range(1,data.repairnode_num)) == 1)
    model.addConstr(quicksum(R[i,0,f] for i in range(1,data.repairnode_num)) == 1)


for i in range(soledemand_num):
    for j in range(data.repairnode_num):
        if j!=0:
            if ru[j][i]!=0:
                model.addConstr(
                    h[i] - quicksum(F[j, f] for f in range(data.crew_num)) + (1 - ru[j][i]) * data.M >= dual_lamda1 +
                    dual_kesei1[j] * data.repairtime_average_UB[j] - dual_kesei2[j] * data.repairtime_average_LB[j]
                    + dual_fai11[j] * data.repair_time_MAD[j] + dual_fai12[j] * data.repair_time_MAD[j]
                    )

for i in range(data.repairnode_num):
    model.addConstr(dual_lamda1 - dual_fai11[i] * data.repair_time_average[i] + dual_fai12[i] * data.repair_time_average[i] >= dual_bigseita1[i])
    model.addConstr(dual_bigseita1[i] >= (1 - dual_kesei1[i] + dual_kesei2[i] - dual_fai11[i] + dual_fai12[i]) * data.repair_time_LB[i])
    model.addConstr(dual_bigseita1[i] >= (1 - dual_kesei1[i] + dual_kesei2[i] - dual_fai11[i] + dual_fai12[i]) * data.repair_time_UB[i])


for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        for f in range(data.crew_num):
            if i != j and j != 0 :
                model.addConstr(F[j,f] - F[i,f] -
                                distance[i][j] + (1 - R[i, j, f]) * data.M >= dual_lamda1_1 + dual_kesei1_1[i] * data.repairtime_average_UB[i] - dual_kesei2_1[i] *
                            data.repairtime_average_LB[i]
                            + dual_fai11_1[i] * data.repair_time_MAD[i] + dual_fai12_1[i] * data.repair_time_MAD[i])

for i in range(data.repairnode_num):
    model.addConstr(dual_lamda1_1 - dual_fai11_1[i] * data.repair_time_average[i] + dual_fai12_1[i] * data.repair_time_average[i] >= dual_bigseita1_1[i])
    model.addConstr(dual_bigseita1_1[i] >= (1 - dual_kesei1_1[i] + dual_kesei2_1[i] - dual_fai11_1[i] + dual_fai12_1[i]) * data.repair_time_LB[i])
    model.addConstr(dual_bigseita1_1[i] >= (1 - dual_kesei1_1[i] + dual_kesei2_1[i] - dual_fai11_1[i] + dual_fai12_1[i]) * data.repair_time_UB[i])



for i in range(1,data.repairnode_num):
    for f in range(data.crew_num):
        model.addConstr(R[0, i, f] + R[i, 0, f] <= 1)

for i in range(data.repairnode_num):
    for f in range(data.crew_num):
        model.addConstr(R[i,i,f]==0)


for i in range(1,data.repairnode_num):
    for j in range(1,data.repairnode_num):
        for f in range(data.crew_num):
            if i!= j:
                model.addConstr(u[i,f] - u[j,f] + data.repairnode_num * R[i, j, f] <= data.repairnode_num - 1)

for f in range(data.crew_num):
    model.addConstr(u[0, f] == 1)
for f in range(data.crew_num):
    for i in range(1, data.repairnode_num):
        model.addConstr(u[i,f] >= 2)


for i in range(data.repairnode_num):
    for j in range(data.repairnode_num):
        for l in range(1,data.repairnode_num):
            for f in range(data.crew_num):
                model.addConstr(FU[i,j,l,f] <= N[i,j,l])
                model.addConstr(FU[i,j,l,f] <= R[i,j,f])
                model.addConstr(FU[i,j,l,f] >= N[i,j,l] + R[i,j,f] -1)



model.Params.OutputFlag = 1
model.Params.LogFile = "log_file.txt"
model.optimize()


model.printAttr('X')
print("objective value:", model.getAttr("ObjVal"))
for i in range(data.repairnode_num):
    for f in range(data.crew_num):
        if F[i,f].x !=0:
            print("the", i, "accessible time:",
                  F[i, f].x  + dual_lamda1.x + dual_kesei1[i].x * data.repairtime_average_UB[i] - dual_kesei2[i].x * data.repairtime_average_LB[i]
                + dual_fai11[i].x * data.repair_time_MAD[i] + dual_fai12[i].x * data.repair_time_MAD[i])

