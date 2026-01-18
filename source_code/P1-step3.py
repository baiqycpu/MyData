import numpy as np
import pandas as pd
import copy
import random as rd

import heapq

import threading


def dijkstra(adj_matrix, start, end, constraint_nodes, constraint_times, start_time=0):
    time_constraints = dict(zip(
        constraint_nodes.flatten().astype(int).tolist(),
        constraint_times.flatten().astype(float).tolist()
    ))

    num_nodes = adj_matrix.shape[0]


    times = np.full(num_nodes, np.inf)
    visited = np.zeros(num_nodes, dtype=bool)

    times[start] = start_time
    priority_queue = [(0, start_time, start)]

    while priority_queue:
        current_distance, current_time, current_node = heapq.heappop(priority_queue)

        if current_node == end:
            return current_time

        if visited[current_node]:
            continue
        visited[current_node] = True

        for neighbor in range(num_nodes):
            travel_time = adj_matrix[current_node][neighbor]
            if np.isinf(travel_time):
                continue

            arrival_time = current_time + travel_time

            if neighbor in time_constraints:
                arrival_time = max(arrival_time, time_constraints[neighbor])

            if arrival_time < times[neighbor]:
                times[neighbor] = arrival_time
                heapq.heappush(priority_queue, (arrival_time, arrival_time, neighbor))

    return np.inf



def  greedy_initialsolution(vehiclecapacity,add0demandnode,demand_UB,consumcapacity,traveltime_full,vehicle_shuliang):

    initialadd0demandnode = copy.deepcopy(add0demandnode)

    add0demandnode = np.delete(add0demandnode,0)

    vehicle_demand =  [[] for _ in range(vehicle_shuliang)]
    vehicle_time = np.zeros((1,vehicle_shuliang))
    vehicle_load = np.zeros((1,vehicle_shuliang))
    gap_ini = np.zeros((1,vehicle_shuliang))

    for i in range(len(add0demandnode)):
        addnodeaftertime = np.zeros((1,vehicle_shuliang))
        for j in range(len(vehicle_demand)):
            currentload = calculateload(add0demandnode[i], demand_UB, consumcapacity,
                                              initialadd0demandnode)
            if vehicle_load[0][j] + currentload <= vehiclecapacity:
                if len(vehicle_demand[j])==0:
                    addnodeaftertime[0][j] = dijkstra(traveltime_full, 0, add0demandnode[i], repairnode,
                                                   repairnode_canreachtime, vehicle_time[0][j])
                else:
                    addnodeaftertime[0][j] =  dijkstra(traveltime_full, vehicle_demand[j][-1], add0demandnode[i], repairnode,
                                                   repairnode_canreachtime, vehicle_time[0][j])
            gap_ini[0][j] = vehicle_time[0][j]+addnodeaftertime[0][j] - max(vehicle_time[0])

        index = np.argmin(gap_ini)
        vehicle_demand[index].append(add0demandnode[i])
        vehicle_time[0][index] = vehicle_time[0][index] + addnodeaftertime[0][j]
        vehicle_load[0][index] = vehicle_load[0][index] + calculateload(add0demandnode[i], demand_UB, consumcapacity,
                                              initialadd0demandnode)

    initional_solution = two_chang_onedimensional(vehicle_demand)
    return initional_solution

def calculateload(index_demand, demand_UB, consumcapacity,add0demandnode):
    load = 0
    demandtype = demand_UB.shape[1]
    index = np.where(add0demandnode == index_demand)
    for i in range(demandtype):
        load = load + demand_UB[index, i] * consumcapacity[i]
    return load


def selectanduse_destoryoperator(destory_weight, current_solution,traveltime_full, w,repairnode,repairnode_canreachtime):
    destory_operator = -1
    sol = copy.deepcopy(current_solution)
    destory_roulette = np.array(destory_weight).cumsum()
    r = rd.uniform(0, max(destory_roulette))
    for i in range(len(destory_roulette)):
        if destory_roulette[i] >= r:
            if i == 0:
                destory_operator = i
                removed_node = random_destory(sol, w)
                destory_times[i] += 1
                break
            elif i == 1:
                destory_operator = i
                removed_node = largest_savingtime_destory(sol, traveltime_full, w,repairnode,repairnode_canreachtime)
                destory_times[i] += 1
                break
    return sol, removed_node, destory_operator

def selectanduse_repairoperator(repair_weight, destory_solution, removelist, vehiclecapacity, demand_UB, traveltime_full,
                                 consumcapacity, vehicle_shuliang,repairnode, repairnode_canreachtime,add0demandnode):
    repair_operator = -1
    repair_roulette = np.array(repair_weight).cumsum()
    r = rd.uniform(0, max(repair_roulette))
    for i in range(len(repair_roulette)):
        if repair_roulette[i] >= r:
            if i == 0:
                repair_operator = i
                repair_solution = random_insert(destory_solution, removelist, vehiclecapacity, demand_UB,
                                                consumcapacity, vehicle_shuliang,add0demandnode)
                repair_times[i] += 1
                break
            elif i == 1:
                repair_operator = i
                repair_solution = greedy_insert(destory_solution, removelist, traveltime_full, demand_UB,
                                                vehiclecapacity, consumcapacity, vehicle_shuliang,repairnode, repairnode_canreachtime,add0demandnode)
                repair_times[i] += 1
                break
            elif i == 2:
                repair_operator = i
                repair_solution = regret_insert(destory_solution, removelist, traveltime_full, demand_UB,
                                                vehiclecapacity, vehicle_shuliang,repairnode, repairnode_canreachtime,add0demandnode)
                repair_times[i] += 1
                break
    return repair_solution, repair_operator


def random_destory(sol, w):
    solution_new = copy.deepcopy(sol)
    removed = []
    non_remove_index = [i for i in range(len(sol)) if sol[i] != 500]
    remove_index = rd.sample(non_remove_index, w)
    for i in range(len(remove_index)):
        removed.append(solution_new[remove_index[i]])
        sol.remove(solution_new[remove_index[i]])
    return removed


def largest_savingtime_destory(sol, traveltime_full, w, repairnode, repairnode_canreachtime):
    removed = []
    vehicel_for_demandnode, usevehicle_num = split_solution(sol)

    timebeforeremoved= time_calculate(vehicel_for_demandnode,usevehicle_num,traveltime_full,repairnode, repairnode_canreachtime)

    timegap_afterremoved = []

    for i in range(len(vehicel_for_demandnode)):
        time_gap_vehicle = []
        currentvehicel_for_demandnode = copy.deepcopy(vehicel_for_demandnode)
        if len(currentvehicel_for_demandnode[i]) > 1:
            current_time = copy.deepcopy(timebeforeremoved)
            BB= copy.deepcopy(currentvehicel_for_demandnode[i])
            BB.remove(currentvehicel_for_demandnode[i][0])
            current_time[i]=time_calculate(BB,0,traveltime_full,repairnode, repairnode_canreachtime)
            time_gap_vehicle.append(max(timebeforeremoved) - max(current_time))
            for j in range(1, len(currentvehicel_for_demandnode[i]) - 1):
                current_time = copy.deepcopy(timebeforeremoved)
                BB =copy.deepcopy(currentvehicel_for_demandnode[i])
                BB.remove(currentvehicel_for_demandnode[i][j])
                current_time[i]= time_calculate(BB, 0, traveltime_full, repairnode, repairnode_canreachtime)

                time_gap_vehicle.append(max(timebeforeremoved) - max(current_time))
            current_time = copy.deepcopy(timebeforeremoved)
            BB = copy.deepcopy(currentvehicel_for_demandnode[i])
            BB.remove(currentvehicel_for_demandnode[i][-1])
            current_time[i] = time_calculate(BB, 0, traveltime_full,  repairnode, repairnode_canreachtime)

            time_gap_vehicle.append(max(timebeforeremoved) - max(current_time))
        else:
            current_time = copy.deepcopy(timebeforeremoved)
            current_time[i] = 0
            time_gap_vehicle.append(max(timebeforeremoved) - max(current_time))
        timegap_afterremoved.append(time_gap_vehicle)


    one_dimensional_timegap = [i for arr in timegap_afterremoved for i in arr]
    sorted_timegap = sorted(one_dimensional_timegap, reverse=True)
    top_w = sorted_timegap[:w]
    current_timegap_afterremoved = copy.deepcopy(timegap_afterremoved)


    for k in range(w):
        position = [(a, b) for a, sublist in enumerate(current_timegap_afterremoved) for b, num in enumerate(sublist) if
                    num == top_w[k]]

        removed.append(vehicel_for_demandnode[position[0][0]][position[0][1]])
        sol.remove(vehicel_for_demandnode[position[0][0]][position[0][1]])
        current_timegap_afterremoved[position[0][0]][position[0][1]] = float("-inf")

    return removed


def split_solution(sol):
    usevehicle_num = sol.count(500)
    vehicel_index = [index for index, value in enumerate(sol) if value == 500]
    vehicel_for_demandnode = []
    vehicel_for_demandnode.append(sol[0:vehicel_index[0]])
    for i in range(usevehicle_num - 1):
        vehicel_for_demandnode.append(sol[vehicel_index[i] + 1:vehicel_index[i + 1]])
    vehicel_for_demandnode.append(sol[vehicel_index[len(vehicel_index) - 1] + 1:len(sol)])
    return vehicel_for_demandnode, usevehicle_num

def time_calculate(vehicel_for_demandnode,usevehicle_num,traveltime_full,repairnode, repairnode_canreachtime):
    timeofvehicle = []

    if usevehicle_num > 0:
        for i in range(usevehicle_num + 1):
            currenttraveltime_full = copy.deepcopy(traveltime_full)
            if len(vehicel_for_demandnode[i]) > 1:

                expr = dijkstra(currenttraveltime_full, 0, vehicel_for_demandnode[i][0], repairnode,
                                repairnode_canreachtime, 0)

                for j in range(len(vehicel_for_demandnode[i]) - 1):
                    expr = dijkstra(currenttraveltime_full, vehicel_for_demandnode[i][j],
                                    vehicel_for_demandnode[i][j + 1], repairnode, repairnode_canreachtime, expr)

                expr = dijkstra(currenttraveltime_full, vehicel_for_demandnode[i][-1], 0, repairnode,
                                repairnode_canreachtime, expr)
                timeofvehicle.append(expr)
            elif len(vehicel_for_demandnode[i]) == 1:
                expr = dijkstra(currenttraveltime_full, 0, vehicel_for_demandnode[i][0], repairnode,
                                repairnode_canreachtime, 0)
                expr = dijkstra(currenttraveltime_full, vehicel_for_demandnode[i][0], 0, repairnode,
                                repairnode_canreachtime, expr)
                timeofvehicle.append(expr)
    elif usevehicle_num == 0:
        currenttraveltime_full = copy.deepcopy(traveltime_full)
        if len(vehicel_for_demandnode) > 1:

            expr = dijkstra(currenttraveltime_full, 0, vehicel_for_demandnode[0], repairnode,
                            repairnode_canreachtime, 0)

            for j in range(len(vehicel_for_demandnode) - 1):
                expr = dijkstra(currenttraveltime_full, vehicel_for_demandnode[j],
                                vehicel_for_demandnode[j + 1], repairnode, repairnode_canreachtime, expr)

            expr = dijkstra(currenttraveltime_full, vehicel_for_demandnode[-1], 0, repairnode,
                            repairnode_canreachtime, expr)
            timeofvehicle.append(expr)
        elif len(vehicel_for_demandnode) == 1:
            expr = dijkstra(currenttraveltime_full, 0, vehicel_for_demandnode[0], repairnode,
                            repairnode_canreachtime, 0)
            expr = dijkstra(currenttraveltime_full, vehicel_for_demandnode[0], 0, repairnode,
                            repairnode_canreachtime, expr)
            timeofvehicle.append(expr)
    return timeofvehicle

def obj_calculate(sol, traveltime_full, repairnode, repairnode_canreachtime):
    vehicle_demand, usevehiclenum = split_solution(sol)
    allvehicle_time= time_calculate(vehicle_demand, usevehiclenum, traveltime_full,repairnode, repairnode_canreachtime)
    obj_maxtimeofallvehicle = max(allvehicle_time)
    return obj_maxtimeofallvehicle

def obj_calculate1(sol, traveltime_full, repairnode, repairnode_canreachtime):
    vehicle_demand, usevehiclenum = split_solution(sol)
    allvehicle_time= time_calculate(vehicle_demand, usevehiclenum, traveltime_full,repairnode, repairnode_canreachtime)
    obj_maxtimeofallvehicle = max(allvehicle_time)
    return allvehicle_time

def two_chang_onedimensional(vehicle_demand):
    onedimensional_sol = []
    for i in range(len(vehicle_demand)):
        if len(vehicle_demand[i]) != 0:
            onedimensional_sol.append(vehicle_demand[i])
            onedimensional_sol.append([500])
    one_dimensional_sol = [i for arr in onedimensional_sol for i in arr]
    one_dimensional_end_sol = one_dimensional_sol[:-1]
    return one_dimensional_end_sol


def fixsolution(sol):
    currentsol = copy.deepcopy(sol)

    if currentsol[0] == 500:
        currentsol = currentsol[1:]
    if currentsol[-1] == 500:
        currentsol = currentsol[:-1]

    newsol = [currentsol[0]]

    for i in range(1, len(currentsol)):
        if currentsol[i] != currentsol[i - 1]:
            newsol.append(currentsol[i])
    return newsol


def random_insert(sol, removelist, vehiclecapacity, demand_UB, consumcapacity,
                  vehicle_shuliang,add0demandnode):
    vehicle_demand, vehicle_num = split_solution(sol)

    for i in range(vehicle_shuliang - len(vehicle_demand)):
        vehicle_demand.append([])

    vehicleload = []
    for i in range(len(vehicle_demand)):
        current_vehicleload = 0
        for j in range(len(vehicle_demand[i])):
            currentnodeload = calculateload(vehicle_demand[i][j], demand_UB, consumcapacity,add0demandnode)
            current_vehicleload = current_vehicleload + currentnodeload
        vehicleload.append(current_vehicleload)

    for i in range(vehicle_shuliang - len(vehicleload)):
        vehicleload.append(0)

    for i in range(len(removelist)):
        available_insertindex = []
        for j in range(len(vehicleload)):
            removednodeload = calculateload(removelist[i], demand_UB, consumcapacity,add0demandnode)
            if vehicleload[j] + removednodeload <= vehiclecapacity:
                available_insertindex.append(list(range(0, len(vehicle_demand[j]) + 1)))
            else:
                available_insertindex.append([])

        non_empty_rows = [index for index, row in enumerate(available_insertindex) if row]
        if non_empty_rows:
            selected_row = rd.choice(non_empty_rows)
            selected_column = rd.randint(0, len(available_insertindex[selected_row]) - 1)

            vehicleload[selected_row] = vehicleload[selected_row] + calculateload(removelist[i], demand_UB,
                                                                                  consumcapacity,add0demandnode)
            vehicle_demand[selected_row].insert(selected_column, removelist[i])
    repairsol = two_chang_onedimensional(vehicle_demand)
    return repairsol


def greedy_insert(sol, removelist, traveltime_full, demand_UB, vehiclecapacity, consumcapacity, vehicle_shuliang,repairnode, repairnode_canreachtime,add0demandnode):
    vehicle_demand, vehicle_num = split_solution(sol)

    for i in range(vehicle_shuliang - len(vehicle_demand)):
        vehicle_demand.append([])

    vehicleload = []
    for i in range(len(vehicle_demand)):
        if len(vehicle_demand[i]) != 0:
            current_vehicleload = 0
            for j in range(len(vehicle_demand[i])):

                currentnodeload = calculateload(vehicle_demand[i][j], demand_UB, consumcapacity,add0demandnode)
                current_vehicleload = current_vehicleload + currentnodeload
            vehicleload.append(current_vehicleload)

    for i in range(vehicle_shuliang - len(vehicleload)):
        vehicleload.append(0)

    time_for_vehicle= time_calculate(vehicle_demand, vehicle_num, traveltime_full,repairnode, repairnode_canreachtime)

    for i in range(vehicle_shuliang - len(time_for_vehicle)):
        time_for_vehicle.append(0)

    for i in range(len(removelist)):
        timeofvehicle = []
        timeofvehicle_index = []
        removednodeload = calculateload(removelist[i], demand_UB, consumcapacity,add0demandnode)
        for jj in range(vehicle_shuliang):
            time_afterinsert = []
            if len(vehicle_demand[jj]) > 0:
                current_vehicle_demand = copy.deepcopy(vehicle_demand)
                if vehicleload[jj] + removednodeload <= vehiclecapacity:

                    BB = copy.deepcopy(current_vehicle_demand[jj])
                    BB.insert(0,removelist[i])
                    AA = time_calculate(BB, 0,traveltime_full,repairnode, repairnode_canreachtime)
                    time_afterinsert.append(AA)
                    for k in range(1, len(vehicle_demand[jj])):
                        BB = copy.deepcopy(current_vehicle_demand[jj])
                        BB.insert(k,removelist[i])
                        AA = time_calculate(BB,0,traveltime_full,repairnode, repairnode_canreachtime)
                        time_afterinsert.append(AA)
                    BB = copy.deepcopy(current_vehicle_demand[jj])
                    BB.append(removelist[i])
                    AA = time_calculate(BB, 0, traveltime_full,repairnode, repairnode_canreachtime)
                    time_afterinsert.append(AA)

                    timeofvehicle.append(min(time_afterinsert))
                    timeofvehicle_index.append(np.argmin(time_afterinsert))
                else:
                    timeofvehicle.append(0)
                    timeofvehicle_index.append("inf")
            else:  # 若是空车
                AA = time_calculate([removelist[i]], 0, traveltime_full,repairnode, repairnode_canreachtime)
                timeofvehicle.append(AA)
                timeofvehicle_index.append(0)

        gap = []

        for k in range(len(timeofvehicle)):
            if timeofvehicle[k] != 0:
                gap.append(max(time_for_vehicle) - timeofvehicle[k])
            else:
                gap.append(float("-inf"))

        for ii in range(len(gap)):
            if isinstance(gap[ii],np.ndarray):
                gap[ii] = gap[ii].mean()
            else:
                gap[ii] = gap[ii]

        vehicle_index = np.argmax(gap)

        thevehicle_index = timeofvehicle_index[vehicle_index]


        vehicle_demand[vehicle_index].insert(thevehicle_index, removelist[i])
        vehicleload[vehicle_index] = vehicleload[vehicle_index] + removednodeload

    repairsol = two_chang_onedimensional(vehicle_demand)
    return repairsol


def regret_insert(sol, removelist, traveltime_full, demand_UB, vehiclecapacity,
                  vehicle_shuliang,repairnode, repairnode_canreachtime,add0demandnode):
    vehicle_demand, vehicle_num = split_solution(sol)

    for i in range(vehicle_shuliang - len(vehicle_demand)):
        vehicle_demand.append([])

    vehicleload = []
    for i in range(len(vehicle_demand)):
        if len(vehicle_demand[i]) != 0:
            current_vehicleload = 0
            for j in range(len(vehicle_demand[i])):
                currentnodeload = calculateload(vehicle_demand[i][j], demand_UB, consumcapacity,add0demandnode)
                current_vehicleload = current_vehicleload + currentnodeload
            vehicleload.append(current_vehicleload)

    for i in range(vehicle_shuliang - len(vehicleload)):
        vehicleload.append(0)

    time_for_vehicle = time_calculate(vehicle_demand, vehicle_num, traveltime_full ,repairnode, repairnode_canreachtime)

    for i in range(vehicle_shuliang - len(time_for_vehicle)):
        time_for_vehicle.append(0)

    for i in range(len(removelist)):
        timeofvehicle_one = []
        timeofvehicle_two = []
        timeofvehicle_index_one = []
        timeofvehicle_index_two = []
        removenodeload = calculateload(removelist[i], demand_UB, consumcapacity,add0demandnode)
        for j in range(vehicle_shuliang):
            time_afterinsert = []
            if len(vehicle_demand[j]) > 0:
                current_vehicle_demand = copy.deepcopy(vehicle_demand)
                if vehicleload[j] + removenodeload <= vehiclecapacity:
                    BB = copy.deepcopy(current_vehicle_demand[j])
                    BB.insert(0, removelist[i])
                    AA= time_calculate(BB, 0,
                                              traveltime_full,  repairnode,
                                              repairnode_canreachtime)
                    time_afterinsert.append(AA)

                    for k in range(1, len(vehicle_demand[j])):
                        BB = copy.deepcopy(current_vehicle_demand[j])
                        BB.insert(k, removelist[i])
                        AA = time_calculate(BB, 0,
                                                  traveltime_full,  repairnode,
                                                  repairnode_canreachtime)
                        time_afterinsert.append(AA)

                    BB = copy.deepcopy(current_vehicle_demand[j])
                    BB.append(removelist[i])
                    AA = time_calculate(BB, 0,
                                              traveltime_full,  repairnode,
                                              repairnode_canreachtime)
                    time_afterinsert.append(AA)

                    timeofvehicle_one.append(min(time_afterinsert))
                    timeofvehicle_index_one.append(np.argmin(time_afterinsert))

                    time_afterinsert[np.argmin(time_afterinsert)] = float("inf")

                    for ii in range(len(time_afterinsert)):
                        if isinstance(time_afterinsert[ii],list):
                            time_afterinsert[ii] = time_afterinsert[ii][0]
                        else:
                            time_afterinsert[ii] = time_afterinsert[ii]

                    timeofvehicle_two.append(min(time_afterinsert))
                    timeofvehicle_index_two.append(np.argmin(time_afterinsert))
                else:
                    timeofvehicle_one.append(0)
                    timeofvehicle_index_one.append("inf")

                    timeofvehicle_two.append(0)
                    timeofvehicle_index_two.append("inf")
            else:

                AA = time_calculate([removelist[i]], 0, traveltime_full,  repairnode, repairnode_canreachtime)

                timeofvehicle_one.append(AA)
                timeofvehicle_index_one.append(0)

                timeofvehicle_two.append(AA)
                timeofvehicle_index_two.append(0)

        gap1 = []
        gap2 = []

        for k in range(len(timeofvehicle_one)):
            if timeofvehicle_one[k] != 0:
                gap1.append(max(time_for_vehicle) - timeofvehicle_one[k])
            else:
                gap1.append(float("-inf"))

        for k in range(len(timeofvehicle_two)):
            if timeofvehicle_two[k] != 0:
                gap2.append(max(time_for_vehicle) - timeofvehicle_two[k])
            else:
                gap2.append(float("-inf"))

        one_two = [gap1, gap2]
        combination_oneandtwo = [l for arr in one_two for l in arr]
        top = [x for x in combination_oneandtwo if x < max(combination_oneandtwo)]

        if max(top) != float("-inf"):
            top_two = max(top)
        else:
            top_two = max(combination_oneandtwo)
        position = [(l1, l2) for l1, sublist in enumerate(one_two) for l2, num in enumerate(sublist) if num == top_two]
        if position[0][0] == 0:
            vehicle_demand[position[0][1]].insert(timeofvehicle_index_one[position[0][1]], removelist[i])
        else:
            vehicle_demand[position[0][1]].insert(timeofvehicle_index_two[position[0][1]], removelist[i])

        vehicleload[position[0][1]] = vehicleload[position[0][1]] + removenodeload  # 更新车辆负载

    repairsol = two_chang_onedimensional(vehicle_demand)
    return repairsol

"""read data of each instance"""

demandnode = pd.read_excel(r'codelocation\demandnode_CX.xlsx',header=None)
np.save("demandnode_CX",demandnode)
demandnode =np.load(r'codelocation\demandnode_CX.npy')
repairnode = pd.read_excel(r'codelocation\repairnode_CX.xlsx',header=None)
np.save("repairnode_CX",repairnode)
repairnode =np.load(r'codelocation\repairnode_CX.npy')
traveltime_full = pd.read_excel(r'codelocation\traveltime_full_CX.xlsx',header=None)
np.save("traveltime_full_CX",traveltime_full)
traveltime_full =np.load(r'codelocation\traveltime_full_CX.npy')
demand_UB1 = pd.read_excel(r'codelocation\demand_UB_CX.xlsx')
np.save("demand_UB_CX",demand_UB1)
demand_UB1=np.load(r'codelocation\demand_UB_CX.npy')
repairnode_canreachtime = pd.read_excel(r'codelocation\repairnode_canreachtime_CX_DMP.xlsx',header=None)
np.save("repairnode_canreachtime_CX",repairnode_canreachtime)
repairnode_canreachtime =np.load(r'codelocation\repairnode_canreachtime_CX_DMP.npy')





depot = [[0, 0, 0, 0]]
demand_UB = np.r_[depot, demand_UB1]
demandnode_num = len(demandnode) + 1
destory_weight = [1 for i in range(2)]
repair_weight = [1 for i in range(3)]
destory_times = [0 for i in range(2)]
repair_times = [0 for i in range(3)]
destory_score = [1 for i in range(2)]
repair_score = [1 for i in range(3)]
iterx, iterxMax = 0, 100
vehiclecapacity = 5000
consumcapacity = [1, 2, 1, 3]
a = 0.97
b = 0.5
T = 100
w = 1
vehicle_shuliang = 8

add0demandnode = np.insert(demandnode,0,[0])
repairnode = np.insert(repairnode,0,[0])

solution= greedy_initialsolution(vehiclecapacity, add0demandnode, demand_UB, consumcapacity,traveltime_full,vehicle_shuliang)
best_solution = copy.deepcopy(solution)

Jbestobj = []
Jbestobj.append( obj_calculate(best_solution, traveltime_full,repairnode,repairnode_canreachtime))
Jbestobj_change = 0

while iterx < iterxMax:
    print(iterx)

    Jbestobj.append(obj_calculate(best_solution, traveltime_full, repairnode, repairnode_canreachtime))
    if Jbestobj[iterx + 1] == Jbestobj[iterx]:
         Jbestobj_change = Jbestobj_change + 1

    if Jbestobj_change >= 10:
        break

    while T > 10:

        destoryedsolution1, remove, destoryoperatorindex = selectanduse_destoryoperator(destory_weight, solution,traveltime_full,w,repairnode, repairnode_canreachtime)
        destoryedsolution = fixsolution(destoryedsolution1)
        newsolution, repairoperatorindex = selectanduse_repairoperator(repair_weight, destoryedsolution, remove,
                                                                       vehiclecapacity, demand_UB, traveltime_full,
                                                                       consumcapacity, vehicle_shuliang,repairnode, repairnode_canreachtime,add0demandnode)
        if obj_calculate(newsolution,traveltime_full,repairnode, repairnode_canreachtime) <= obj_calculate(solution, traveltime_full,repairnode,repairnode_canreachtime):
            solution = newsolution
            if obj_calculate(newsolution, traveltime_full,repairnode,repairnode_canreachtime) <= obj_calculate(best_solution, traveltime_full,repairnode,repairnode_canreachtime):
                best_solution = newsolution
                destory_score[destoryoperatorindex] += 1.5
                repair_score[repairoperatorindex] += 1.5
            else:
                destory_score[destoryoperatorindex] += 1.2
                repair_score[repairoperatorindex] += 1.2
        else:
            if rd.random() < np.exp(-obj_calculate(newsolution,traveltime_full,repairnode,repairnode_canreachtime) / T):
                solution = newsolution
                destory_score[destoryoperatorindex] += 0.8
                repair_score[repairoperatorindex] += 0.8
            else:
                destory_score[destoryoperatorindex] += 0.6
                repair_score[repairoperatorindex] += 0.6

        destory_weight[destoryoperatorindex] = destory_weight[destoryoperatorindex] * b + (1 - b) * \
                                               (destory_score[destoryoperatorindex] / destory_times[
                                                   destoryoperatorindex])
        repair_weight[repairoperatorindex] = repair_weight[repairoperatorindex] * b + (1 - b) * \
                                             (repair_score[repairoperatorindex] / repair_times[repairoperatorindex])

        T = a * T
    iterx += 1
    T = 100

print(best_solution)
print(obj_calculate(best_solution, traveltime_full,repairnode,repairnode_canreachtime))
print(obj_calculate1(best_solution, traveltime_full,repairnode,repairnode_canreachtime))
