import matplotlib.pyplot as plt
import matplotlib.cm as clr
import csv
import numpy as np

"""
This file makes a plot of the Mandelbrot, for pregenerated data. Shown in
figure 1 in the report.

Guido Vaessen (12488860)
Joos Akkerman (11304723)
Stochastic Simulation - Assignment 1
"""

size_x = 2000
size_y = 2000

# read csv of mandelbrot set
x_pixels = np.linspace(-2.5, 1, size_x)
y_pixels = np.linspace(-1, 1, size_y)


with open('mandelbrot_2000.csv', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # y_coord = 1
    all_pixels = []
    for i,row in enumerate(reader):
        row_pixels = []
        for j,pixel in enumerate(row):
            if pixel != '':
                row_pixels += [np.log(int(pixel))]
        all_pixels += [np.array(row_pixels)]

all_pixels = np.array(all_pixels)
plt.figure(figsize=(17.5,10))
plt.imshow(all_pixels, aspect='auto')
plt.axis('off')
plt.tight_layout()
plt.savefig('mandelbrot_set_2000x2000.pdf')
