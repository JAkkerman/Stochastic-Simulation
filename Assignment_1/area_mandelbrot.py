import numpy as np
import matplotlib.pyplot as plt


def sample(type, s, row_count, col_count, subsamp):
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
        n = int(len(rows)/np.sqrt(subsamp))
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
            count += 1
            coordinates += [(x, y)]

    return coordinates


def mandelbrot_area(coordinates, i):
    """
    Estimates the area of the Mandelbrot Set
    """
    inset = 0
    total_area = 3.5*2

    for j in range(s):
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


def plot_diff():

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


if __name__ == '__main__':

    i = 1000
    s = 1000
    row_count = 2*s
    col_count = 2*s
    type      = 'Random'
    subsamp   = 4

    coordinates = sample(type, s, row_count, col_count, subsamp)
    area        = mandelbrot_area(coordinates, i)
    print(f'Area with random sampling: {area}')

    type = 'LatinHypercube'
    coordinates = sample(type, s, row_count, col_count, subsamp)
    area        = mandelbrot_area(coordinates, i)
    print(f'Area with Latin Hypercube sampling: {area}')

    type = 'Orthogonal'
    coordinates = sample(type, s, row_count, col_count, subsamp)
    area        = mandelbrot_area(coordinates, i)
    print(f'Area with Orthogonal sampling: {area}')

    # i = np.logspace(1, 4, 10, dtype=int)
    # j = np.array([int(0.9*I) for I in i])
    # plot_diff(i,j)
