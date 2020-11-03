import numpy as np
import matplotlib.pyplot as plt


def print_to_csv(x_pixels, values):

    string = ''
    for i in range(len(x_pixels)):
        string += str(values[i])+','

    f = open("mandelbrot_2000.csv", "a")
    f.write(string + '\n')
    f.close()


def mandelbrot_set(size_x, size_y, max_iter):
    """
    Computes the mandelbrot set and plots
    """

    # for each pixel, perform iteration
    x_pixels = np.linspace(-2.5, 1, size_x)
    y_pixels = np.linspace(-1, 1, size_y)

    for i in y_pixels:
        values = []
        for j in x_pixels:
            y0 = i
            x0 = j
            x = 0
            y = 0
            iteration = 0
            while (x*x + y*y) <= 2*2 and iteration < max_iter:
                z = x*x - y*y + x0
                y = 2*x*y + y0
                x = z
                iteration += 1

            values += [iteration]

        print_to_csv(x_pixels, values)

if __name__ == '__main__':
    size_x   = 2000
    size_y   = 2000
    max_iter = 1000

    mandelbrot_set(size_x, size_y, max_iter)
