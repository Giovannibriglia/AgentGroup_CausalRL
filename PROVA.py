import matplotlib.pyplot as plt

decay = 0.995
start = 1
n_ep = 10000
vet = []

res = start
for e in range(n_ep):
    res = res * decay
    vet.append(res)


plt.plot(vet)
plt.show()