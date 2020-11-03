import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_area(s, i):
    inset = 0
    outset = 0
    total_area = 3.5*2

    for j in range(s):
        x0 = np.random.uniform(-2.5, 1)
        y0 = np.random.uniform(-1, 1)
        x = 0
        y = 0
        iteration = 0
        while (x*x + y*y) <= 2*2 and iteration < i:
            z = x*x - y*y + x0
            y = 2*x*y + y0
            x = z
            iteration += 1

        if iteration == i:
            inset += 1
        else:
            outset += 1

    return inset/s * total_area


i = np.logspace(1, 4, 10, dtype=int)
j = np.array([int(0.9*I) for I in i])
s = 1000

n_runs = 30
mean_diff = []
std_diff = []
for I, J in zip(i, j):
    difference = []
    for k in range(n_runs):
        difference += [mandelbrot_area(s, J) - mandelbrot_area(s, I)]
    mean_diff += [np.mean(difference)]
    std_diff += [np.std(difference)]

mean_diff = np.array(mean_diff)
std_diff = np.array(std_diff)

plt.plot(j, mean_diff, color='blue')
plt.fill_between(j, mean_diff - std_diff, mean_diff + std_diff, color='blue',
                 alpha=0.5)
plt.xlim(min(j), max(j))
plt.xlabel('j')
plt.ylabel(r'$A_{j,s} - A_{i, s}$')

plt.show()
