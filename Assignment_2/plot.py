import matplotlib.pyplot as plt
import csv
import numpy as np

def plot_nservers():
    """
    Plots histogram for 1, 2 and 4 servers
    """

    fig = plt.subplots(1, 1, figsize=[10,4])

    # open data
    with open('res_n_servers.csv', newline='\n') as csvfile:
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
    plt.xlim(0,60)
    plt.ylabel('number of experiments', fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig('hist_servers_together.pdf')


def plot_conf_n_servers():

    for type in [0,1]:

        fig = plt.subplots(1, 1, figsize=[8,4])

        all_n_experiments = np.array([int(i) for i in np.logspace(1,4,10)])

        avgs = [[] for i in range(1,4)]
        stds = [[] for i in range(1,4)]

        for n_experiments in all_n_experiments:
            with open('res_n_servers_'+str(n_experiments)+'_'+str(type)+'_.csv', newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                for i,row in enumerate(reader):
                    row = [np.single(i) for i in row if i != '' if '[' not in i and ']' not in i]

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

            plt.plot(all_n_experiments, avg - (1.96*std)/np.sqrt(all_n_experiments),
                    label=lab, color=c)
            plt.plot(all_n_experiments, avg + (1.96*std)/np.sqrt(all_n_experiments),
                    color=c)

            plt.fill_between(all_n_experiments, avg - (1.96*std)/np.sqrt(all_n_experiments),
                            avg + (1.96*std)/np.sqrt(all_n_experiments), color=c, 
                            alpha=0.4)
        
        plt.xscale('log')
        plt.xlabel('number of experiments', fontsize=14)
        plt.ylabel('confidence interval', fontsize=14)
        plt.legend(fontsize=12, loc='right')
        # plt.show()
        plt.tight_layout()
        plt.savefig('CI_nservers_all_'+str(type)+'.pdf')


def plot_conf_prior():

    # for type in [0,1]:

    type = 1

    fig = plt.subplots(1, 1, figsize=[8,4])

    all_n_experiments = np.array([int(i) for i in np.logspace(1,4,10)])

    avgs = [[] for i in range(2)]
    stds = [[] for i in range(2)]

    for j,prior in enumerate(['', 'prior']):
        for n_experiments in all_n_experiments:
            with open('res_n_servers_'+str(n_experiments)+'_'+str(type)+'_'+prior+'.csv', newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                for i,row in enumerate(reader):
                    if prior == '':
                        if i==0:
                            row = [np.single(i) for i in row if i != '' if '[' not in i and ']' not in i]

                            avgs[j] += [np.average(row)]
                            stds[j] += [np.std(row)]
                    else:
                        if i!=0:
                            row = [np.single(i) for i in row if i != '' if '[' not in i and ']' not in i]

                            avgs[j] += [np.average(row)]
                            stds[j] += [np.std(row)]

    avgs = [np.array(avg) for avg in avgs]
    stds = [np.array(std) for std in stds]

    print(avgs)

    for i in range(2):
        avg = avgs[i]
        std = stds[i]
        c = 'blue'
        lab = 'no priority'
        if i==1:
            c = 'orange'
            lab = 'priority'

        print(len(avg))
        print(len(std))

        plt.plot(all_n_experiments, avg - (1.96*std)/np.sqrt(all_n_experiments),
                label=lab, color=c)
        plt.plot(all_n_experiments, avg + (1.96*std)/np.sqrt(all_n_experiments),
                color=c)

        plt.fill_between(all_n_experiments, avg - (1.96*std)/np.sqrt(all_n_experiments),
                        avg + (1.96*std)/np.sqrt(all_n_experiments), color=c, 
                        alpha=0.4)
    
    plt.xscale('log')
    plt.xlabel('number of experiments', fontsize=14)
    plt.ylabel('confidence interval', fontsize=14)
    plt.legend(fontsize=12, loc='right')
    plt.tight_layout()
    # plt.show()
    plt.savefig('CI_prior.pdf')


    

if __name__ == '__main__':
    # plot_nservers()

    plot_conf_n_servers()

    # plot_conf_prior()
