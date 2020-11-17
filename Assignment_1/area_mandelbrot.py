import numpy as np
import matplotlib.pyplot as plt
import math
import csv


def print_to_csv(type, areas):

    string = ''
    for i in range(len(areas)):
        string += str(areas[i])+','

    f = open(str(type)+'.csv', "a")
    f.write(string + '\n')
    f.close()


def sample(type, s, row_count, col_count, subsamps):
    """
    Samples random points fully random, using Latin Hypercube or using
    Orthogonal Sampling, given the type of sampling.
    """

    coordinates = []

    # perform simple random sampling. If not random sampling, move further
    if type == "Random":

        for i in range(s):
            x = np.random.uniform(-2.5, 1)
            y = np.random.uniform(-1, 1)
            coordinates += [(x, y)]

        return coordinates

    if type == "Antithetic":

        # for i in range(int(s/2)):
        while len(coordinates)<s:
            x = np.random.uniform(-2.5, 1)
            x_mir = -0.75 - (x+0.75)
            y = np.random.uniform(-1, 1)
            y_mir = -y
            # print((x, y), (x_mir, y_mir))
            coordinates += [(x, y), (x_mir, y_mir)]

        return coordinates


    # Latin Hypercube sampling, with extension if Orthogonal sampling
    # divide sample space in rows and columns
    subspaces = []
    rows = list(np.linspace(-2.5, 1, row_count))
    cols = list(np.linspace(-1, 1, col_count))

    # divide rows and colums up in sub-sample spaces if orthogonal
    if type == "Orthogonal":

        # for orthogonal sampling, n has to be a multiple of the amount of
        # subsamples. So, round n up.
        s = subsamps * round(s/subsamps)
        print(s)
        n = int(len(rows)/np.sqrt(subsamps))

        for i in range(0, len(rows), n):
            for j in range(0, len(cols), n):
                subspaces += [(rows[i:i+n], cols[j:j+n])]

    # if not orthogonal, all rows and columns are part of sample space
    else:
        subspaces += [(rows, cols)]

    # iterate over sample space(s), randomly pick combination of row and column,
    # pick location within
    for subspace in subspaces:
        subrows = subspace[0]
        subcols = subspace[1]

        # determine width of rows and columns, all rows have the same width, as
        # do colums, so only computing the distance for the first row and column
        # suffices
        row_width = subrows[1] - subrows[0]
        col_width = subcols[1] - subcols[0]

        for pick in range(int(s/len(subspaces))):
            row = np.random.choice(subrows)
            col = np.random.choice(subcols)
            subrows.remove(row)
            subcols.remove(col)

            x = np.random.uniform(row, row+row_width)
            y = np.random.uniform(col, col+col_width)
            coordinates += [(x, y)]

    return coordinates


def mandelbrot_area(type, s, i, row_count, col_count, subsamps):
    """
    Estimates the area of the Mandelbrot Set
    """
    inset = 0
    total_area = 3.5*2

    # print(s)
    coordinates = sample(type, s, row_count, col_count, subsamps)
    s = subsamps * round(s/subsamps)
    for j in range(s):
        # print(len(coordinates))
        x0 = coordinates[j][0]
        y0 = coordinates[j][1]
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

    return inset/s * total_area, coordinates


def plot_diff(i, j, s, n_runs=30, type='Random', row_count=1, col_count=1,
              subsamp=4, ax=None, color='blue'):

    if ax is None:
        ax = plt.gca()

    mean_diff = []
    std_diff = []
    for J in j:
        difference = []
        for k in range(n_runs):
            difference += [mandelbrot_area(J, s, type, row_count, col_count,
                           subsamp) - mandelbrot_area(i, s, type, row_count,
                           col_count, subsamp)]
        mean_diff += [np.mean(difference)]
        std_diff += [np.std(difference)]

    mean_diff = np.array(mean_diff)
    std_diff = np.array(std_diff)

    ax.plot(j, mean_diff, color=color)
    ax.fill_between(j, mean_diff - std_diff, mean_diff+std_diff, color=color,
                    alpha=0.5)
    ax.set_xlim(min(j), max(j))
    ax.set_xscale('log')
    ax.set_xlabel('j')
    ax.set_ylabel(r'$A_{j,s} - A_{i, s}$')


def s_experiment(type, s, i, row_count, col_count, subsamps):
    """
    Experiment for different values of sample size
    """

    N = 60
    # S = np.logspace(1, np.log(5000), 10)
    S = [9]
    S += [i**2  for i in range(5,80,5)]
    i = 1000

    print(f's experiment for {type}')

    for s in S:
        results_for_s = []
        for n in range(N):
            row_count = int(s)
            col_count = int(s)
            subsamps  = int(s)
            area, coordinates = mandelbrot_area(type, int(s), i, row_count, col_count, subsamps)
            results_for_s += [area]

        print_to_csv('exp_s_'+str(type), results_for_s)


def control_experiment(type, s, i, row_count, col_count, subsamps):
    """
    Experiment for control variates
    """

    N = 60
    S = [9]
    S += [i**2  for i in range(5,80,5)]
    i = 1000

    print(f'control variets experiment for {type}')

    r1  = 0.25
    cx1 = -1
    cy1 = 0

    r2  = 0.6
    cx2 = -0.15
    cy2 = 0

    area_circles = np.pi*r1**2 + np.pi*r2**2

    for s in S:
        results_for_s = []
        results_Y = []
        results_X = []
        for n in range(N):
            row_count = int(s)
            col_count = int(s)
            subsamps  = int(s)

            area, coordinates = mandelbrot_area(type, int(s), i, row_count, col_count, subsamps)
            results_X += [area]

            count_in = 0
            for coord in coordinates:
                d1 = r1**2 - ((cx1-coord[0])**2 + (cy1-coord[1])**2)
                d2 = r2**2 - ((cx2-coord[0])**2 + (cy2-coord[1])**2)

                if d1 > 0 or d2 > 0:
                    count_in += 1

            area_cont = 7*(count_in/len(coordinates))
            results_Y += [area_cont]

        c = -np.cov(results_X, results_Y)[0][1]/np.var(results_Y)

        results_for_s += [results_X[i] + c*(results_Y[i] - area_circles) for i in range(60)]

        print_to_csv('exp_cont_'+str(type), results_for_s)


if __name__ == '__main__':

    i = 1000
    s = 1000
    row_count = s
    col_count = s
    subsamps = s

    # type = 'Random'
    # area = mandelbrot_area(type, s, i, row_count, col_count, subsamps)
    # print(f'Area with random sampling: {area}')
    #
    # type = 'LatinHypercube'
    # area = mandelbrot_area(type, s, i, row_count, col_count, subsamps)
    # print(f'Area with Latin Hypercube sampling: {area}')
    #
    # type = 'Orthogonal'
    # area = mandelbrot_area(type, s, i, row_count, col_count, subsamps)
    # print(f'Area with Orthogonal sampling: {area}')

    # type = 'Random'
    # s_experiment(type, s, i, row_count, col_count, subsamps)

    # type = "LatinHypercube"
    # s_experiment(type, s, i, row_count, col_count, subsamps)
    #
    # type = "Orthogonal"
    # s_experiment(type, s, i, row_count, col_count, subsamps)

    # type = "Antithetic"
    # s_experiment(type, s, i, row_count, col_count, subsamps)

    # type = "Control"
    # s_experiment(type, s, i, row_count, col_count, subsamps)

    # i = np.logspace(1, 4, 10, dtype=int)
    # j = np.array([int(0.9*I) for I in i])
    # plot_diff(i,j, s)
    # plt.show()

    # type = 'Random'
    # control_experiment(type, s, i, row_count, col_count, subsamps)

    type = 'LatinHypercube'
    control_experiment(type, s, i, row_count, col_count, subsamps)

    type = 'Orthogonal'
    control_experiment(type, s, i, row_count, col_count, subsamps)
