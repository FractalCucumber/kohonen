import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

N = 200
input = []
data = pd.read_csv('Mall_Customers.csv')

max_age = int(data["Age"].max())
mean_age = int(data["Age"].mean())

max_inc = int(data["Annual Income (k$)"].max())
mean_inc = int(data["Annual Income (k$)"].mean())

max_score = int(data["Spending Score (1-100)"].max())
mean_score = int(data["Spending Score (1-100)"].mean())


for i, row in data.iterrows():
    if row[1] == 'Male': gen = -1
    else: gen = 1

    age = (row[2]-mean_age)/(max_age-mean_age)
    inc = (row[3]-mean_inc)/(max_inc-mean_inc)
    score = (row[4]-mean_score)/(max_score-mean_score)

    input += [[gen, age, inc, score]]


k = 5
output = [[0 for j in range(3)] for i in range(3)]

for i in range(3):
    for j in range(3):
        output[i][j] = [random.random()*0.2-0.1, random.random()*0.2-0.1, random.random()*0.2-0.1, random.random()*0.2-0.1]

def eta(n, N):
    return 1-n/N

def G(n, i1, j1, i2, j2):
    if n<0: return 1
    else:
        if i1==i2 and j1==j2: return 1
        else: return 0


clst = [[0, 0] for i in range(N)]

T = 500
SSE = [0 for i in range(T)]
for t in range(T):
    n = random.randint(0, N-1)
    xi = input[n]
    min_r = 10**6
    winner = []
    for i in range(3):
        for j in range(3):
            w = output[i][j]
            r = sum([(w[t]-xi[t])**2 for t in range(4)])
            if r < min_r:
                min_r = r
                winner = [i, j]
    clst[n] = winner
    for i in range(3):
        for j in range(3):
            w = output[i][j]
            output[i][j] = [w[t] + eta(n+1, T)*G(n, *winner, i, j)*(xi[t] - w[t]) for t in range(4)]
    SSE[t] = sum([sum([(output[clst[k][0]][clst[k][1]][t]-input[k][t])**2 for t in range(4)]) for k in range(N)])


fig, ax = plt.subplots()

ax.plot(np.array(SSE))
ax.set_xlabel('итерации')
ax.set_ylabel('сумма квадратов ошибок')
plt.show()


cluster = [output[clst[i][0]][clst[i][1]] for i in range(N)]

SSE1 = [0 for i in range(9)]

SSE1[0] = SSE[T-1]

weights = []
for i in range(3):
    for j in range(3):
        weights += [output[i][j]]

for t in range(1, 9):
        min_r = 10**6
        u1 = 0
        u2 = 0
        n = len(weights)
        answer = [0 for i in range(n-1)]
        for i in range(n):
            for j in range(i+1, n):
                r = sum([(weights[i][m] - weights[j][m])**2 for m in range(4)])
                if r < min_r:
                    min_r = r
                    u1 = weights[i]
                    u2 = weights[j]
        mean = [(u1[m]+u2[m])/2 for m in range(4)]
        for m in range(N):
            if np.array_equal(cluster[m], u1) or np.array_equal(cluster[m], u2):
                cluster[m] = mean
        weights.remove(u1)
        weights.remove(u2)
        weights += [mean]

        for i in range(n-1):
            if weights[i][0] > 0:
                gen = "Female"
            else:
                gen = "Male"

            age = int(weights[i][1] * (max_age - mean_age) + mean_age)
            inc = int(weights[i][2] * (max_inc - mean_inc) + mean_inc)
            score = int(weights[i][3] * (max_score - mean_score) + mean_score)

            answer[i] = [gen, age, inc, score]

        print(n-1, ' clusters:')
        for i in range(n-1):
            print(answer[i])


        SSE1[t] = sum([sum([(cluster[k][m]-input[k][m])**2 for m in range(4)]) for k in range(N)])

fig, ax = plt.subplots()

ax.plot(np.array([i for i in range(9, 0, -1)]), np.array(SSE1))
ax.set_xlabel('число кластеров')
ax.set_ylabel('сумма квадратов ошибок')
plt.show()


