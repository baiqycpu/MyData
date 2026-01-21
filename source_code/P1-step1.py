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

    """..........."""

#Read data
def readData(data):

    """Read data of each instance CX"""

    data.demandnode = pd.read_excel(r'cede_location\demandnode_CX.xlsx', header=None)

    data.repairnode = pd.read_excel(r'cede_location\repairnode_CX.xlsx', header=None)
    data.repairnode_num = "the number of repairnode"
    data.traveltime_full = pd.read_excel(r'cede_location\traveltime_full.xlsx', header=None)
    data.traveltime_part = pd.read_excel(r'cede_location\traveltime_part_CX.xlsx',
                                         header=None)
    """......"""



def dijkstra(adj_matrix, start, end):
    
    """Define dijksatra algorothm to obtain shortest route between two nodes"""

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
    """Define dijksatra algorothm that consider damaged nodes to obtain shortest route between two nodes"""

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


"""Define variables based on the model, and write the objective function and constraints."""

model = Model("MP-P1")
u = model.addVars(data.repairnode_num,data.crew_num, vtype=GRB.CONTINUOUS, name='u')

h = model.addVars(soledemand_num, vtype=GRB.CONTINUOUS, name = "h")
hf = model.addVars(soledemand_num, vtype=GRB.CONTINUOUS, name = "hf")
F = model.addVars(data.repairnode_num,data.crew_num,vtype=GRB.CONTINUOUS,name="F")
R = model.addVars(data.repairnode_num, data.repairnode_num, data.crew_num,vtype = GRB.BINARY, name = "R")

"""........"""

"""dual variables"""
dual_lamda1 = model.addVar(vtype=GRB.CONTINUOUS, name="dual_lamda1")
dual_kesei1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei1")
dual_kesei2 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei2")
dual_lamda1_1 = model.addVar(vtype=GRB.CONTINUOUS, name="dual_lamda1_1")
dual_kesei1_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei1_1")
dual_kesei2_1 = model.addVars(data.repairnode_num, vtype=GRB.CONTINUOUS, name="dual_kesei2_1")

"""..........."""

z = model.addVars(data.repairnode_num,data.crew_num, vtype=GRB.BINARY, name= "z")

FU = model.addVars(data.repairnode_num,data.repairnode_num,data.repairnode_num,data.crew_num,vtype=GRB.BINARY, name= "FU")


"""In accordance with the journal's data and code sharing policy, we provide the core implementation of the proposed algorithm. 
Due to proprietary technology limitations, some parts have been omitted."""

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
                    model.addConstr(F[j,f] >= F[k,f] - (1-FU[i,j,k,f]) * data.M)


for f in range(data.crew_num):
    model.addConstr(F[0,f] ==0)


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


"""........"""
"""".........""""
"""........"""

model.Params.OutputFlag = 1
model.Params.LogFile = "log_file.txt"
model.optimize()



model.printAttr('X')
print("objective value:", model.getAttr("ObjVal"))


