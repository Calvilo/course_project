import numpy as np
import time
import random
T1=time.time()
#city_loc = np.array([[ 67, 146, 282,  97,  18, 172,  51, 160,  80,  91, 233, 173, 268, 300, 297, 125,  78, 217,
#  172, 243],[275 ,135, 123, 207, 141, 298, 211,  71,  57,  81, 255,  52,  58, 223, 151, 201, 173, 219,
#   69, 124]])

def dist_calculate(x ,y):
    #print("x: ",x ,"y: ", y, "x - y: ", x - y, "norm: ", np.linalg.norm(x - y) )
    return np.linalg.norm(x - y) 


city_num=100
city_loc = np.array([  [random.randint(1,300) if j < 2 else i for i in range(city_num)  ] for j in range(3)])

distmat = np.array([[ dist_calculate(city_loc[0:2,i] ,city_loc[0:2,j]) for j in range(city_num)] for i in range(city_num)])

numant = 12 
numpoint = distmat.shape[0] 
alpha = 1   #信息素重要程度因子
beta = 5    #启发函数重要程度因子
rho = 0.1   #信息素的挥发速度
Q = 1

iter = 0
itermax = 150

etatable = 1.0/(distmat+np.diag([1e10]*numpoint)) #启发函数矩阵
        
pheromonetable  = np.ones((numpoint,numpoint)) # 信息素矩阵
pathtable = np.zeros((numant,numpoint)).astype(int) #路径记录表

lengthaver = np.zeros(itermax) 
lengthbest = np.zeros(itermax) 
pathbest = np.zeros((itermax,numpoint)) 

while iter < itermax:
    # 随机产生各个蚂蚁的起点城市
    if numant <= numpoint:
        pathtable[:,0] = np.random.permutation(range(0,numpoint))[:numant] 
    else: #蚂蚁数比城市数多，需要补足
        pathtable[:numpoint,0] = np.random.permutation(range(0,numpoint))[:]
        pathtable[numpoint:,0] = np.random.permutation(range(0,numpoint))[:numant-numpoint]

    length = np.zeros(numant) 

    for i in range(numant): 
        visiting = pathtable[i,0] 
        visited = set() #已访问过的城市，防止重复
        visited.add(visiting) 
        unvisited = set(range(numpoint))
        unvisited.remove(visiting) 

        for j in range(1,numpoint):
            #每次用轮盘法选择下一个要访问的城市
            listunvisited = list(unvisited)
            probtrans = np.zeros(len(listunvisited))

            for k in range(len(listunvisited)): 
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]],alpha)\
                *np.power(etatable[visiting][listunvisited[k]],alpha)

            cumsumprobtrans = (probtrans/sum(probtrans)).cumsum()  #累加数组，归一化           
            cumsumprobtrans -= np.random.rand() 
            
            for m in range(len(cumsumprobtrans)):  #选择下一个要访问的城市
                if cumsumprobtrans[m]>0:           
                    k=listunvisited[m]
                    break
            
            pathtable[i,j] = k
            unvisited.remove(k)
            visited.add(k)

            length[i] += distmat[visiting][k]
            visiting = k

        length[i] += distmat[visiting][pathtable[i,0]] 
    #print('the %d iteration,all the ant''s distance are:'%iter)
    #print(length)
    # 输出本次迭代，所有蚂蚁的路径距离

    lengthaver[iter] = length.mean() 

    if iter == 0: 
        lengthbest[iter] = length.min()
        pathbest[iter] = pathtable[length.argmin()].copy()     
    else: #以后的迭代
        if length.min() > lengthbest[iter-1]: #比上一次最优迭代差
            lengthbest[iter] = lengthbest[iter-1]
            pathbest[iter] = pathbest[iter-1].copy()
        else:
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()  

    # 更新信息素
    changepheromonetable = np.zeros((numpoint,numpoint))
    for i in range(numant):
        for j in range(numpoint-1):
            changepheromonetable[pathtable[i,j]][pathtable[i,j+1]] += \
            Q/distmat[pathtable[i,j]][pathtable[i,j+1]]

        changepheromonetable[pathtable[i,j+1]][pathtable[i,0]] += \
        Q/distmat[pathtable[i,j+1]][pathtable[i,0]]
    pheromonetable = (1-rho)*pheromonetable + changepheromonetable

    iter += 1 #迭代次数指示器+1

bestpath = pathbest[-1]
bestlength = lengthbest[-1]
T2=time.time()
print('最佳路径')
print(bestpath)
print('最短距离：%d'%bestlength)
print("time cost: %s ms"%((T2-T1)*1000) )
