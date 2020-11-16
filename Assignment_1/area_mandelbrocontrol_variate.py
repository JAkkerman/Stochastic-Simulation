import numpy as np
import matplotlib.pyplot as plt
import math


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

    count = 0

    # perform simple random sampling. If not random sampling, move further
    if type == "Random":

        for i in range(s):
            x = np.random.uniform(-2.5, 1)
            y = np.random.uniform(-1, 1)
            coordinates += [(x, y)]
            count += 1

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

    # print(subspaces)

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
            count += 1
            coordinates += [(x, y)]

    return coordinates


def mandelbrot_area(type, s, i, row_count, col_count, subsamps):
    """
    Estimates the area of the Mandelbrot Set
    """
    inset = 0
    total_area = 3.5*2

    coordinates = sample(type, s, row_count, col_count, subsamp)

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

    return inset/s * total_area


def plot_diff(i, j, s, n_runs=30, type='Random', row_count=1, col_count=1,
              subsamp=4, ax=None):

    if ax is None:
        ax = plt.gca()

    mean_diff = []
    std_diff = []
    for J in j:
        difference = []
        for k in range(n_runs):
            difference += [mandelbrot_area(i, s, type, row_count, col_count,
                           subsamp) - mandelbrot_area(J, s, type, row_count,
                           col_count, subsamp)]
        mean_diff += [np.mean(difference)]
        std_diff += [np.std(difference)]

    mean_diff = np.array(mean_diff)
    std_diff = np.array(std_diff)

    ax.plot(j, mean_diff, color='blue')
    ax.fill_between(j, mean_diff - std_diff, mean_diff+std_diff, color='blue',
                    alpha=0.5)
    ax.xlim(min(j), max(j))
    ax.xscale('log')
    ax.xlabel('j')
    ax.ylabel(r'$A_{j,s} - A_{i, s}$')

    return ax


def control_variate(X, Y, mu_Y):
    cov_matrix = np.cov([X, Y])
    c = -cov_matrix[0, 1] / cov_matrix[1, 1]
    return np.array(X) + c * (np.mean(Y)-mu_Y)


def ES_area(bm, max_m):
    '''
    Calculation of the area of M using method from Ewing and Schroder
    '''
    m = np.sum(np.arange(1, max_m) * abs(bm)**2)
    return np.pi * (1 - m)


if __name__ == '__main__':

    i = 1000
    S = np.logspace(1, 4, 10, dtype=int)

    all_area = []
    all_variates = []

    for s in S:
        row_count = 2*s
        col_count = 2*s
        type = 'Random'
        subsamp = 4
        n_runs = 30

        all_area += [[mandelbrot_area(i, s, type, row_count, col_count, subsamp)
                     for j in range(n_runs)]]
        Y = np.random.uniform(0, 100, n_runs)
        all_variates += [control_variate(all_area[-1], Y, 50)]

    all_area = np.array(all_area)
    all_variates = np.array(all_variates)

    # normal method
    plt.plot(S, np.mean(all_area, axis=1), color='blue')
    plt.fill_between(S, np.mean(all_area, axis=1)-np.std(all_area),
                     np.mean(all_area, axis=1) + np.std(all_area), color='blue'
                     , alpha=0.5)

    # method with contrl variates
    plt.plot(S, np.mean(all_variates, axis=1), color='red')
    plt.fill_between(S, np.mean(all_variates, axis=1)-np.std(all_variates),
                     np.mean(all_variates, axis=1) + np.std(all_variates),
                     color='red', alpha=0.5)

    plt.xlim(min(S), max(S))
    plt.xlabel('s')
    plt.ylabel('area')
    plt.show()


def s_experiment(type, s, i, row_count, col_count, subsamps):

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
            area = mandelbrot_area(type, int(s), i, row_count, col_count, subsamps)
            results_for_s += [area]

        print_to_csv('exp_s_'+str(type), results_for_s)


if __name__ == '__main__':

    i = 1000
    s = 1000
    row_count = s
    col_count = s
    subsamps  = s

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

    type = 'Random'
    s_experiment(type, s, i, row_count, col_count, subsamps)

    type = "LatinHypercube"
    s_experiment(type, s, i, row_count, col_count, subsamps)

    type = "Orthogonal"
    s_experiment(type, s, i, row_count, col_count, subsamps)
>>>>>>> master

    # type = 'LatinHypercube'
    # area = [mandelbrot_area(i, s, type, row_count, col_count, subsamp)
    #         for j in range(n_runs)]
    # print(f'Area with Latin Hypercube sampling: {np.mean(area)}')
    #
    # type = 'Orthogonal'
    # area = [mandelbrot_area(i, s, type, row_count, col_count, subsamp)
    #         for j in range(n_runs)]
    # print(f'Area with Orthogonal sampling: {np.mean(area)}')

    # j = np.logspace(1, 3, 10, dtype=int)
    # plot_diff(i, j, s)
    # plt.show()
