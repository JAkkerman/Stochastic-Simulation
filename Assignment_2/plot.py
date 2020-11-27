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

if __name__ == '__main__':
    plot_nservers()
