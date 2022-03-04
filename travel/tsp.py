from math import sqrt
import random
from re import L
from typing import Optional
from numpy.lib.function_base import average, insert
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
from convex import convex_hull
from time import sleep
import itertools  # 用于排列组合

def dist_calculate(x ,y):
    print("x: ",x ,"y: ", y, "x - y: ", x - y, "norm: ", np.linalg.norm(x - y) )
    return np.linalg.norm(x - y) 

def indexofMin(arr):
    minindex = 0
    currentindex = 1
    while currentindex < len(arr):
        if arr[currentindex] < arr[minindex]:
            minindex = currentindex
        currentindex += 1
    return minindex

def reduce_cost(dist, city_sequence):
    res = []
    delta = 0
    links = [(city_sequence[0,i],city_sequence[0,i+1], i)  for i in range(len(city_sequence.T) - 1 )]
    for combination in itertools.combinations(links,2):
        if(combination[0][1] == combination[1][0]  or combination[0][0] == combination[1][1]):
            continue
        else:
            newlink_1 = (combination[0][0], combination[1][0])
            newlink_2 = (combination[0][1], combination[1][1])
            temp_delta = dist[newlink_1] + dist[newlink_2] - dist[combination[0][0:2]] - dist[combination[1][0:2]]
            if(temp_delta < delta):
                delta = temp_delta
                res =  [combination[0][2] + 1, combination[1][2], delta]
    return res
    

if __name__ == "__main__":
    
    ## 获取参数
    opts, _ = getopt.getopt(
        sys.argv[1:],
        "N:S:O:",
        [
            "city_num=",
            "spare=",
            "option="
        ],
    )
    city_num = 20
    spare_val = 0
    optional_flag = 0

    for opt, val in opts:
        if opt in ("--city_num", "-N"):
            print("city num is specified as :", val)
            city_num = int(val)
        if opt in ("-S", "--spare"):
            print("spare para is specified as : ", val)
            spare_val = int(val)
        if opt in ("-O", "--option"):
            print("optional flag is specified as : ", val)
            optional_flag = int(val)
    
    #生成问题
    city_loc = np.array([  [random.randint(1,300) if j < 2 else i for i in range(city_num)  ] for j in range(3)])
    np.savetxt('city_location.txt', city_loc, fmt="%d",delimiter=',') #保存为整数
    # 距离矩阵计算
    dist = np.array([[ dist_calculate(city_loc[0:2,i] ,city_loc[0:2,j]) for j in range(city_num)] for i in range(city_num)])
    #生成凸包 (初始解)
    convex_para = [(city_loc[0][i], city_loc[1][i], city_loc[2][i]) for i in range(city_num)]
    convex_res  = convex_hull(convex_para)
    
    init_res = np.array([[convex_res[i][j]  for i in range(len(convex_res)) ] for j in range(3)])
    init_res = np.column_stack((init_res, init_res[:,0]))
    
    all_index = set(range(city_num))
    inside_index = set(init_res[2,:])
    outside_index = all_index - inside_index

    # 绘制初始解
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for item in city_loc.T:
        plt.text(item[0], item[1], item[2])
    ax.set_aspect(1)
    ax.scatter(city_loc[0],city_loc[1],color="red")
    ax.plot(init_res[0,:],init_res[1,:])
    plt.savefig("init_res.svg")
    
    
    iteration_id = 1
    target_val = []
    
    while (len(outside_index) > 0): 
        print("iteration :",iteration_id )
        iteration_id = iteration_id + 1
        chosen_one = outside_index.pop()
        
        links = [( init_res[2][i], init_res[2][i+1]) if i < len(inside_index) - 1  else ( init_res[2][i], init_res[2][0]) for i in range(len(inside_index))]
        print("emmmm",links)
        # emm_val  指的是插入代价
        links_val = [dist[(i[0], chosen_one)]+ dist[i[1], chosen_one] for i in links]
        target_val.append( sum([dist[link[0]][link[1]]   for link in links  ]))
        min_index = indexofMin(links_val)
        print("insert ", chosen_one , "between ", links[min_index][0], "and",links[min_index][1])
        init_res = np.insert(init_res, min_index + 1, city_loc[:,chosen_one],1 )
    print("final res is:", init_res)
    
    # 绘制 中间解
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.scatter(city_loc[0],city_loc[1])
    ax.plot(init_res[0,:],init_res[1,:])
    for item in city_loc.T:
        plt.text(item[0], item[1], item[2])
    
    plt.savefig("mid_res.svg")
    cross_index =  reduce_cost(dist,init_res[2:3,:])
    
    while(len(cross_index) > 0):
        print(cross_index[2])
        target_val.append(sum([ dist[init_res[2,i], init_res[2,i+1]] for i in range(city_num)]))
        temp = np.copy(init_res[0:3,cross_index[1]]) 
        init_res[0:3,cross_index[1]] = init_res[:,cross_index[0]]
        init_res[0:3,cross_index[0]] = temp
        cross_index = reduce_cost(dist, init_res[2:3,:])
        
    # 绘制最终解
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.scatter(city_loc[0],city_loc[1])
    ax.plot(init_res[0,:],init_res[1,:])
    for item in city_loc.T:
        plt.text(item[0], item[1], item[2])
    
    plt.savefig("fin_res.svg")
    #绘制目标值 历史曲线
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.scatter(iteration_id-1, target_val[iteration_id-1], color='red')
    ax.plot(target_val)
    ax.set_title("target value history")
    ax.set_xlabel("iteration index")
    ax.set_ylabel("target value")
    ax.grid(True)
    plt.savefig("target_val.svg")
    plt.show()