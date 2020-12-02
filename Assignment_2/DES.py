import numpy as np
import simpy
import matplotlib.pyplot as plt
import copy
import math

def queue(la, mu, n, time):
    rho = la / (n*mu)
    SS = 0
    Na = 0
    Nd = 0
    n = 0
    T0 = 10
    tA = copy.copy(T0)
    tD = float('inf')



    for i in range(time):
        np.random.exponential(1/la)


def mean_wait(la, mu, n):
    # calculates the mean waiting time for an M/M/n queue
    rho = la / (mu)
    if n == 1:
        return rho / ((1 - rho) * mu)
    
    else: 
        sum_term = (1-rho) * np.sum([(n*rho)**m / math.factorial(m) for m in range(n)])
        return (n*rho)**n / math.factorial(n) * (sum_term + (n*rho)**n / math.factorial(n))**(-1) * 1 / (1-rho) * 1 / (n*mu)

la = 1
mu = 4/3
n_vals = np.arange(1, 10, 1)

system_1 = np.array([mean_wait(la, mu, 1) for n in n_vals])
system_n = np.array([mean_wait(la, mu, n) for n in n_vals])

plt.plot(n_vals, system_1, '.-',  color='red', label='$M/M/1$')
plt.plot(n_vals, system_n, '.-', color='blue', label='$M/M/n$')
plt.legend()
# plt.xlim(min(n_vals), max(n_vals))
plt.xlabel('$n$')
plt.ylabel('$E(W)$')
plt.savefig('mean_waiting_times.pdf')
plt.show()