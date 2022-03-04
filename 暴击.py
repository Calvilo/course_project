p = []
check = 0
for i in range(11):
    p.append(min(1, 1.05 - i * 0.1) * min(1, 0.05 + 0.1 * i))
    for j in range(i):
        p[i] *= min(1, 1 - j * 0.1 + 0.05)
    check += p[i]
    p[i] *= i + 1
print("sum of probability: ", check)
print("expectation of n:", sum(p))
print("result: ", 1 / sum(p))

