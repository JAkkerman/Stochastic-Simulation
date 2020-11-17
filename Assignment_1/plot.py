import matplotlib.pyplot as plt
import numpy as np
import csv
import tabulate as tabul
import pandas as pd
import scipy.stats as stats


def plot_s_experiment():
    """
    Plots confidence interval
    """

    S = [9]
    S += [i**2  for i in range(5,80,5)]

    for type in ['Random', 'LatinHypercube', 'Orthogonal']:
        with open('exp_s_'+type+'.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            areas = []
            for i,row in enumerate(reader):
                row = [np.single(i) for i in row if i != '']
                areas += [(np.average(row), np.std(row))]

        avg = np.array([i[0] for i in areas])
        std = np.array([i[1] for i in areas])
        if type == 'Random':
            plt.semilogx(S, avg - (1.96*std)/np.sqrt(S), label=type, color='blue', linestyle='dotted')
            plt.semilogx(S, avg + (1.96*std)/np.sqrt(S), color='blue', linestyle='dotted')

        elif type == 'LatinHypercube':
            plt.semilogx(S, avg - (1.96*std)/np.sqrt(S), label=type, color='red', linestyle='--')
            plt.semilogx(S, avg + (1.96*std)/np.sqrt(S), color='red', linestyle='--')
        else:
            plt.semilogx(S, avg - (1.96*std)/np.sqrt(S), label=type, color='green', linestyle='-.')
            plt.semilogx(S, avg + (1.96*std)/np.sqrt(S), color='green', linestyle='-.')

    plt.xlabel('sample size (s)',fontsize=12)
    plt.ylabel('Confidence interval for $a_i$',fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend()
    plt.savefig('s_experiment_std1.pdf', bbox_inches = "tight")


def plot_error():
    """
    Plots standard deviations
    """

    S = [9]
    S += [i**2  for i in range(5,80,5)]

    for type in ['Random', 'LatinHypercube', 'Orthogonal']:
        with open('exp_s_'+type+'.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            areas = []
            for i,row in enumerate(reader):
                row = [np.single(i) for i in row if i != '']
                areas += [(np.average(row), np.var(row))]

        std = np.array([i[1] for i in areas])

        if type == 'Random':
            plt.plot(S, std, label=type, color='blue')
            plt.tick_params(labelsize=14)
        elif type == 'LatinHypercube':
            plt.plot(S, std, label=type, color='red')
            plt.tick_params(labelsize=14)
        else:
            plt.plot(S, std, label=type, color='green')
            plt.tick_params(labelsize=14)


    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$\hat{\sigma}^2_i$', fontsize=12)
    plt.xlabel('sample size (s)',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig('s_experiment_std2.pdf', bbox_inches = "tight")


def create_table():
    """
    Generates table with results
    """

    data = {}
    for type in ['Random', 'LatinHypercube', 'Orthogonal']:
        with open('exp_s_'+type+'.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data[type] = [i for i in reader]

    for type in ['Random', 'LatinHypercube', 'Orthogonal']:
        with open('exp_cont_'+type+'.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data[type+'_cont'] = [i for i in reader]

    def avg(list):
        list = [[np.single(i) for i in j if i != ''] for j in list]
        list = [round(np.average(j), 3) for  j in list]
        return list

    def std(list):
        list = [[np.single(i) for i in j if i != ''] for j in list]
        list = [round(np.var(j), 3) for  j in list]
        return list

    def f_test(list1, list2):
        list1 = [[np.single(i) for i in j if i != ''] for j in list1]
        list2 = [[np.single(i) for i in j if i != ''] for j in list2]
        f_res = []
        pvals = []
        for i,c in enumerate(list1):
            res = stats.levene(c, list2[i])
            f_res += [res[0]]
            pvals += [res[1]]

        all_res = []
        # print(f_res)
        for i,c in enumerate(f_res):
            stars = ''
            if pvals[i]/2 <= 0.01:
                stars = '^{***}'
            elif pvals[i]/2 <= 0.05:
                stars = '^{**}'
            elif pvals[i]/2 <= 0.1:
                stars = '^*'
            all_res += [f'${round(c,3)}'+stars+'$']

        # print(all_res)

        return all_res


    table = {}
    table['$s$'] = [9]+[i**2  for i in range(5,80,5)]

    # table['$\hat{a_R}$'] = [round(j, 3) for j in [avg(i) for i in data['Random']]]
    table['$\hat{A}_R$'] = avg(data['Random'])
    table['$\hat{\sigma}^2_R$'] = std(data['Random'])

    table['$\hat{A}_{LH}$'] = avg(data['LatinHypercube'])
    table['$\hat{\sigma}^2_{LH}$'] = std(data['LatinHypercube'])

    table['$\hat{A}_O$'] = avg(data['Orthogonal'])
    table['$\hat{\sigma}^2_O$'] = std(data['Orthogonal'])

    table['$\mathcal{W}_{R>LH}$'] = f_test(data['Random'], data['LatinHypercube'])
    table['$\mathcal{W}_{R>O}$'] = f_test(data['Random'], data['Orthogonal'])
    table['$\mathcal{W}_{LH>O}$'] = f_test(data['LatinHypercube'], data['Orthogonal'])

    table['$\mathcal{W}_{R>R_C}$'] = f_test(data['Random'], data['Random_cont'])
    table['$\mathcal{W}_{LH>LH_C}$'] = f_test(data['LatinHypercube'], data['LatinHypercube_cont'])
    table['$\mathcal{W}_{O>O_C}$'] = f_test(data['Orthogonal'], data['Orthogonal_cont'])

    df = pd.DataFrame(table)
    print(df.to_latex(index=False, escape=False, multicolumn_format='r'))


def control_variets_fit():
    """
    Plots the fit for the control variant
    """

    img = plt.imread('mandelbrot_set_2000x2000.png')
    fig, ax = plt.subplots(figsize=[14,8])
    ax.imshow(img, extent=[-2.5, 1, -1, 1])

    circle1 = plt.Circle((-1,0), 0.25, color='red', fill=False, linestyle='--', linewidth=5)
    circle2 = plt.Circle((-0.1,0), 0.65, color='red', fill=False, linestyle='--', linewidth=5)

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('control_fit.pdf')


def control_variets_var():
    """
    Plots the difference in variance for the control variates
    """
    S = [9]
    S += [i**2  for i in range(5,80,5)]

    # for type in ['Random', 'LatinHypercube', 'Orthogonal']:
    for type in ['LatinHypercube']:
        fig, ax = plt.subplots(1,2, figsize=[8,3.5])
        for f in ['s_', 'cont_']:
            with open('exp_'+f+type+'.csv', newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                areas = []
                for i,row in enumerate(reader):
                    row = [np.single(i) for i in row if i != '']
                    areas += [(np.average(row), np.var(row))]

            avg = np.array([i[0] for i in areas])
            std = np.array([i[1] for i in areas])

            # determine color
            color = 'blue'
            if type == 'LatinHypercube':
                color = 'red'
            elif type == 'Orthogonal':
                color = 'green'

            if f == 's_':
                ax[1].plot(S, std, label='Pure '+type, color=color)

                ax[0].semilogx(S, avg - (1.96*std)/np.sqrt(S), label='Pure '+type, color=color)
                ax[0].semilogx(S, avg + (1.96*std)/np.sqrt(S), color=color)
                # ax1.tick_params(labelsize=14)
            elif f == 'cont_':
                ax[1].plot(S, std, label='Control variates', color=color, linestyle='--')

                ax[0].semilogx(S, avg - (1.96*std)/np.sqrt(S), label='Control variates', color=color, linestyle='dotted')
                ax[0].semilogx(S, avg + (1.96*std)/np.sqrt(S), color=color, linestyle='dotted')

        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[0].set_xscale('log')

        ax[0].set_ylabel('Confidence interval $\hat{A}$', fontsize=12)
        ax[0].set_xlabel('sample size (s)',fontsize=12)

        ax[1].set_ylabel('$\hat{\sigma}^2_i$', fontsize=12)
        ax[1].set_xlabel('sample size (s)',fontsize=12)

        ax[0].legend()
        ax[1].legend()

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'cont_experiment_{type}.pdf', bbox_inches = "tight")


def plot_antithetic():
    S = [9]
    S += [i**2  for i in range(5,80,5)]

    for type in ['Random1', 'Orthogonal', 'Antithetic']:
        with open('exp_s_'+type+'.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            areas = []
            for i,row in enumerate(reader):
                row = [np.single(i) for i in row if i != '']
                areas += [(np.average(row), np.var(row))]

        std = np.array([i[1] for i in areas])

        if type == 'Random1':
            plt.plot(S, std, label=type, color='blue')
            plt.tick_params(labelsize=14)
        elif type == 'Orthogonal':
            plt.plot(S, std, label=type, color='green')
            plt.tick_params(labelsize=14)
        elif type == 'Antithetic':
            plt.plot(S, std, label=type, color='orange')
            plt.tick_params(labelsize=14)
        # elif type == 'Control':

    control_X = control_variets()
    plt.plot(S, [np.var(i) for i in control_X], label='Control', color='purple')
    plt.tick_params(labelsize=14)


    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$\hat{\sigma}^2_i$', fontsize=12)
    plt.xlabel('sample size (s)',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.show()
    # plt.savefig('s_experiment_anti.pdf', bbox_inches = "tight")


if __name__ == '__main__':
    # plot_s_experiment()

    # plot_error()

    create_table()

    # plot_antithetic()

    # control_variets_fit()

    # control_variets_var()
