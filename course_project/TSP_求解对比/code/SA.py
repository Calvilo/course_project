#模拟退火算法
import numpy as np
import time
import random



T1 = time.time() 
city_num=100
city_loc = np.array([  [random.randint(1,300) if j < 2 else i for i in range(city_num)  ] for j in range(3)])
#city_loc = np.array([[ 67, 146, 282,  97,  18, 172,  51, 160,  80,  91, 233, 173, 268, 300, 297, 125,  78, 217,
#  172, 243],[275 ,135, 123, 207, 141, 298, 211,  71,  57,  81, 255,  52,  58, 223, 151, 201, 173, 219,
#   69, 124]])
def dist_calculate(x ,y):
    #print("x: ",x ,"y: ", y, "x - y: ", x - y, "norm: ", np.linalg.norm(x - y) )
    return np.linalg.norm(x - y) 

   

distmat = np.array([[ dist_calculate(city_loc[0:2,i] ,city_loc[0:2,j]) for j in range(city_num)] for i in range(city_num)])

def initpara(): #参数初始化
    alpha = 0.99
    t = (1,10) 
    markovlen = 10000
    return alpha,t,markovlen
alpha,t2,markovlen = initpara()

num = distmat.shape[0] 

solutionnew = np.arange(num)  #创建数组，返回一个array对象，需要引入numpy
valuenew = 100000 

solutioncurrent = solutionnew.copy()
valuecurrent = 100000

solutionbest = solutionnew.copy()
valuebest = 100000

t = t2[1] 
result = [] #记录迭代过程中的最优解

while t > t2[0]:
    for i in np.arange(markovlen): 

        #下面的两交换和三角换是两种扰动方式，用于产生新解
        if np.random.rand() > 0.5:# 两交换            
            while True:
                loc1 = np.int(np.ceil(np.random.rand()*(num-1))) 
                loc2 = np.int(np.ceil(np.random.rand()*(num-1)))
                if loc1 != loc2:
                    break  #只有当得到两不同的随机数时才跳出循环
            solutionnew[loc1],solutionnew[loc2] = solutionnew[loc2],solutionnew[loc1] 
        else: #三交换
            while True:
                loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                loc2 = np.int(np.ceil(np.random.rand()*(num-1))) 
                loc3 = np.int(np.ceil(np.random.rand()*(num-1)))

                if((loc1 != loc2)&(loc2 != loc3)&(loc1 != loc3)):
                    break

            # 下面的三个判断语句使得loc1<loc2<loc3
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1   
            if loc2 > loc3: 
                loc2,loc3 = loc3,loc2
            if loc1 > loc2:
                loc1,loc2 = loc2,loc1

            #将[loc1,loc2)之间的数据移到loc3之后，3之后的代码移到1之前
            tmplist = solutionnew[loc1:loc2].copy()       
            solutionnew[loc1:loc3-loc2+1+loc1] = solutionnew[loc2:loc3+1].copy()
            solutionnew[loc3-loc2+1+loc1:loc3+1] = tmplist.copy()  

        valuenew = 0 
        for i in range(num-1):
            valuenew += distmat[solutionnew[i]][solutionnew[i+1]]
        valuenew += distmat[solutionnew[0]][solutionnew[11]]  

        if valuenew < valuecurrent: 
            #接受该解，更新solutioncurrent 和solutionbest
            valuecurrent = valuenew
            solutioncurrent = solutionnew.copy()

            if valuenew < valuebest:
                valuebest = valuenew
                solutionbest = solutionnew.copy()
        else:#按一定的概率接受该解
            if np.random.rand() < np.exp(-(valuenew-valuecurrent)/t):
                valuecurrent = valuenew
                solutioncurrent = solutionnew.copy()
            else:
                solutionnew = solutioncurrent.copy()

    t = alpha*t #逐步减小接受错误的概率
    result.append(valuebest)
    print (t) #程序运行时间较长，通过t来监视程序进展速度

T2=time.time()
print('最优路径：')
print(solutionbest)
print('最短距离：')
print(valuebest)
print("time cost: %s ms"%((T2-T1)*1000) )
