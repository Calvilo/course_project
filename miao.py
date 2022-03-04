import pulp as p

pro = p.LpProblem("berth", p.LpMaximize)
index = range(3)
x = p.LpVariable.dict("x", index, 0, None, cat=p.LpInteger)
cost=[1/4 ,3/8 ,1/2]
constraint = [25, 35, 45]
pro += sum([cost[i]*x[i] for i in index ])
pro += sum([constraint[i]*x[i] for i in index ]) <= 100
pro.solve()
for v in pro.variables():
    print(v.name, "=", v.varValue)