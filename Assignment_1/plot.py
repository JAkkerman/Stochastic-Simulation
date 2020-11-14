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
        print(f_res)
        for i,c in enumerate(f_res):
            stars = ''
            if pvals[i] <= 0.01:
                stars = '^{***}'
            elif pvals[i] <= 0.05:
                stars = '^{**}'
            elif pvals[i] <= 0.1:
                stars = '^*'
            all_res += [f'{round(c,3)}{stars}']

        print(all_res)

        return all_res


    table = {}
    table['sample size (s)'] = [9]+[i**2  for i in range(5,80,5)]

    # table['$\hat{a_R}$'] = [round(j, 3) for j in [avg(i) for i in data['Random']]]
    table['$\hat{a}_R$'] = avg(data['Random'])
    table['$\hat{\sigma}^2_R$'] = std(data['Random'])

    table['$\hat{a}_{LH}$'] = avg(data['LatinHypercube'])
    table['$\hat{\sigma}^2_{LH}$'] = std(data['LatinHypercube'])

    table['$\hat{a}_O$'] = avg(data['Orthogonal'])
    table['$\hat{\sigma}^2_O$'] = std(data['Orthogonal'])

    table['W_{R=LH}'] = f_test(data['Random'], data['LatinHypercube'])
    table['W_{R=O}'] = f_test(data['Random'], data['Orthogonal'])
    table['W_{LH=O}'] = f_test(data['LatinHypercube'], data['Orthogonal'])

    df = pd.DataFrame(table)
    print(df.to_latex(index=False, escape=False, multicolumn_format='r'))


if __name__ == '__main__':
    # plot_s_experiment()

    plot_error()

    # create_table()
