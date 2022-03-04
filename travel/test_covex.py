from convex import convex_hull
if __name__ == '__main__':
    pass
    test_data = [(220, -100, 1), (0,0,2), (-40, -170,3), (240, 50,4), (-160, 150,5), (-210, -150,6)]
    print(test_data)

    result = convex_hull(test_data)
    print(result)
    t=0

import matplotlib.pyplot as plt
x1=[]
y1=[]
for i in range(len(test_data)):
    ri=test_data[i]
    #print(ri)
    x1.append(ri[0])
    y1.append(ri[1])

plt.plot(x1,y1,linestyle=' ',marker='.')


xx=[]
yy=[]
for i in range(len(result)):
    ri=result[i]
    #print(ri)
    xx.append(ri[0])
    yy.append(ri[1])
ri = result[0]
xx.append(ri[0])
yy.append(ri[1])

plt.plot(xx,yy,linestyle='-',marker='*')
plt.show()