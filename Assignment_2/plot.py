import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import scipy.stats as stats

def plot_nservers():
    """
    Plots histogram for 1, 2 and 4 servers
    """

    fig = plt.subplots(1, 1, figsize=[10,4])

    n_servers = 1
    n_clients = 1000
    ser_type  = 'exp'
    n_exp     = 1000

    # open data
    with open('Data/'+str(n_clients)+'_cl_'+str(n_exp)+'_exper_'+ser_type+'_ser.csv', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for i,row in enumerate(reader):
            row = [np.single(i) for i in row if i != '']

            c = 'blue'
            lab = '1 server'
            if i==1:
                c = 'red'
                lab = '2 servers'
            elif i==2:
                c = 'green'
                lab = '4 servers'

            # plot average data in a histogram
            plt.hist(row, label=lab, alpha=0.5, bins=50, color=c)

    plt.xlabel('average waiting time', fontsize=14)
    plt.xlim(0,7)
    plt.ylabel('number of experiments', fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig('hist_servers_together.pdf')


def plot_conf_n_servers():

    fig = plt.subplots(1, 1, figsize=[8,4])

    all_n_exp = np.array([int(i) for i in np.logspace(1,3,8)])

    avgs = [[] for i in range(1,4)]
    stds = [[] for i in range(1,4)]

    n_servers = 1
    n_clients = 1000
    ser_type  = 'exp'

    for n_exp in all_n_exp:
        with open('Data/'+str(n_clients)+'_cl_'+str(n_exp)+'_exper_'+ser_type+'_ser.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i,row in enumerate(reader):
                row = [np.single(i) for i in row if i != '' if '[' not in i and ']' not in i]
                print(row)

                avgs[i] += [np.average(row)]
                stds[i] += [np.std(row)]

    avgs = [np.array(avg) for avg in avgs]
    stds = [np.array(std) for std in stds]

    for i in range(0,3):
        avg = avgs[i]
        std = stds[i]
        c = 'blue'
        lab = '1 server'
        if i==1:
            c = 'red'
            lab = '2 servers'
        elif i==2:
            c = 'green'
            lab = '4 servers'

        plt.plot(all_n_exp, avg - (1.96*std)/np.sqrt(all_n_exp),
                label=lab, color=c)
        plt.plot(all_n_exp, avg + (1.96*std)/np.sqrt(all_n_exp),
                color=c)

        plt.fill_between(all_n_exp, avg - (1.96*std)/np.sqrt(all_n_exp),
                        avg + (1.96*std)/np.sqrt(all_n_exp), color=c, 
                        alpha=0.4)
    
    plt.xscale('log')
    plt.xlabel('number of experiments', fontsize=14)
    plt.ylabel('confidence interval', fontsize=14)
    plt.legend(fontsize=12, loc='right')
    plt.tight_layout()
    plt.savefig('CI_nservers.pdf')


def plot_conf_prior():
    """
    Plots figure for performance with priority
    """

    fig = plt.subplots(1, 1, figsize=[8,4])

    all_n_exp = np.array([int(i) for i in np.logspace(1,3,8)])

    avgs = [[] for i in range(2)]
    stds = [[] for i in range(2)]

    n_clients = 1000
    ser_type  = 'exp'

    for j,prior in enumerate(['', '_prior']):
        for n_exp in all_n_exp:
            filename = 'Data/'+str(n_clients)+'_cl_'+str(n_exp)+'_exper_'+ser_type+'_ser'+prior+'.csv'
            print(filename)
            with open(filename, newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                for i,row in enumerate(reader):

                    # print(row)
                    if prior == '':
                        if i==0:
                            row = [np.single(k) for k in row if k != '' if '[' not in k and ']' not in k]

                            avgs[j] += [np.average(row)]
                            stds[j] += [np.std(row)]
                    else:
 
                        row = [np.single(k) for k in row if k != '' if '[' not in k and ']' not in k]

                        avgs[j] += [np.average(row)]
                        stds[j] += [np.std(row)]

    avgs = [np.array(avg) for avg in avgs]
    stds = [np.array(std) for std in stds]

    print(avgs)
    print(stds)

    for i in range(2):
        avg = avgs[i]
        std = stds[i]
        c = 'blue'
        lab = 'no priority'
        if i==1:
            c = 'orange'
            lab = 'priority'

        plt.plot(all_n_exp, avg - (1.96*std)/np.sqrt(all_n_exp),
                label=lab, color=c)
        plt.plot(all_n_exp, avg + (1.96*std)/np.sqrt(all_n_exp),
                color=c)

        plt.fill_between(all_n_exp, avg - (1.96*std)/np.sqrt(all_n_exp),
                        avg + (1.96*std)/np.sqrt(all_n_exp), color=c, 
                        alpha=0.4)
    
    plt.xscale('log')
    plt.xlabel('number of experiments', fontsize=14)
    plt.ylabel('confidence interval', fontsize=14)
    plt.legend(fontsize=12, loc='right')
    plt.tight_layout()
    plt.show()
    # plt.savefig('CI_prior.pdf')


def table1():
    """
    Creates table 1, showing averages, standard deviations and t-tests
    """
    all_n_exp = np.array([int(i) for i in np.logspace(1,4,10)])
    data = {'1':[], '2':[], '4':[]}

    for n_exp in all_n_exp:
        with open('res_n_servers_'+str(n_exp)+'_1_.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i,row in enumerate(reader):
                # print(row)
                data[row[0]] += [row[1:]]

    print(data)

    def avg(list1):
        list1 = [[np.single(i) for i in j if i != ' ' and i != ''] for j in list1]
        list1 = ['%.3f' % np.average(j) for  j in list1]
        return list1

    def std(list1):
        list1 = [[np.single(i) for i in j if i != ' ' and i != ''] for j in list1]
        list1 = ['%.3f' % np.std(j) for  j in list1]
        return list1

    def ttest(list1, list2):
        list1 = [[np.single(i) for i in j if i != ' ' and i != ''] for j in list1]
        list2 = [[np.single(i) for i in j if i != ' ' and i != ''] for j in list2]

        print(list1[0])
        print(list2[0])

        results = []
        for j,l in enumerate(list1):
            res = stats.ttest_ind(l, list2[j], equal_var=False)
            print(res)
            tval = res[0]
            pval = res[1]
            results += [[tval,pval]]

        print('yeet')
        print(results)

        ttests = []
        for i,c in enumerate(results):
            stars = ''
            if c[1]/2 <= 0.01:
                stars = '^{***}'
            elif c[1]/2 <= 0.05:
                stars = '^{**}'
            elif c[1]/2 <= 0.1:
                stars = '^*'
            ttests += [f'${round(c[1],3)}'+stars+'$']
        
        return ttests


    table = {}
    table['$n_{exp}$'] = list(all_n_exp)


    for w in ['1','2','4']:
        table['$\overline{W}_'+w+'$'] = avg(data[w])
        table['${S}_{\overline{W}_'+w+'}$'] = std(data[w])

    for c in [('1','2'), ('1','4'), ('2','4')]:
        table['$t_{'+c[0]+'>'+c[1]+'}$'] = ttest(data[c[0]], data[c[1]])


    df = pd.DataFrame(table)
    print(df.to_latex(index=False, escape=False, multicolumn_format='r'))

    

if __name__ == '__main__':
    # plot_nservers()

    # plot_conf_n_servers()

    plot_conf_prior()

    # table1()