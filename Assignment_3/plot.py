# import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

from main import error 
from main import integrate
from main import open_data


def plot_best(all_overal_min):
    """
    Plots best results
    """

    t,x,y = open_data()

    for j, error_method in enumerate(['absolute', 'mean squared']):

        fig, ax = plt.subplots(1,2, figsize=[10,4])

        overall_min = all_overal_min[j]
        # print(overall_min[1])
        # param  = ['0.8765293781135163', '0.45235692134210853', '1.9732604243852723', '1.150132947492257']
        print(error_method, overall_min[1])
        # x_val, y_val = integrate(np.single(np.array(overall_min[1])),t,x,y,x_keys=np.array([False]), y_keys=np.array([False]))

        # ax[0].plot(t,x,label='real', alpha=0.6, linewidth=1.5, c='red')
        # ax[0].plot(t,x_val,label='best fitted',alpha=0.6, linewidth=1.5, c='blue')
        # ax[0].set_ylabel('predators', fontsize=12)
        # ax[0].set_xlabel('t', fontsize=12)
        # ax[0].legend()
        

        # ax[1].plot(t,y,label='real', alpha=0.6, linewidth=1.5, c='red')
        # ax[1].plot(t,y_val,label='best fitted', alpha=0.6, linewidth=1.5, c='blue')
        # ax[1].set_ylabel('prey', fontsize=12)
        # ax[1].set_xlabel('t', fontsize=12)
        # ax[1].legend()

        # fig.suptitle(error_method+' error = '+str(round(overall_min[0],2)), fontsize=14)
        # fig.subplots_adjust(top=0.88)
        # fig.tight_layout()
        # plt.show()
        # plt.savefig(error_method+'_bestres.pdf')


def boxplots():
    """
    Makes boxplots of found errors
    """

    fig, ax = plt.subplots(1,2, figsize=[8,3])
    all_overal_min = []

    for j,error_method in enumerate(['absolute','mean squared']):

        all_best = []
        all_last = []

        cooling = 'linear'
        filename = 'SA_'+cooling+'_'+error_method+'.csv'
        with open(filename) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                overal_min = [10,[]]

                for i,row in enumerate(reader):
                    errors = np.single(np.array(row[8:]))
                    # print(errors)
                    all_best += [np.single(min(errors))]
                    all_last += [np.single(errors[-1])]

                    # print(row[:8])
                    # print(np.single(min(errors)),'\n')

                    if np.single(min(errors)) < overal_min[0]:
                        overal_min = [np.single(min(errors)), row[4:8]]

        # print(overal_min)
        # print(all_best)
        # print(all_last)

        ax[j].boxplot([all_last, all_best], labels=['last', 'best'], widths=0.6)
        ax[j].set_title(error_method+' errors', fontsize=12)
        ax[j].set_ylabel(error_method+' error', fontsize=12)

        all_overal_min += [overal_min]

        # plt.show()
        # plt.savefig(error_method+'_boxplot.pdf')

    # print(all_overal_min)

    plot_best(all_overal_min)


def boxplot_randomreduce():

    for q in ['x', 'y']:

        cooling = 'linear'
        error_method = 'mean squared'
        filename = 'SA_'+cooling+'_'+error_method+'_reducerand_'+q+'.csv'
        with open(filename) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                errors = []
                for i,row in enumerate(reader):
                    # print(row[4:8])

                    t,x,y = open_data()

                    x_val, y_val = integrate(np.single(np.array(row[4:8])),t,x[0],y[0], x_keys=np.linspace(0,99,100), y_keys=np.linspace(0,99,100))
                    errors += [error(x, y, x_val, y_val, np.linspace(0,99,100), np.linspace(0,99,100), error_method='mean squared')]

        # print([errors[10*i:10*i+10] for i in range(10)])
        plt.boxplot([errors[10*i:10*i+10] for i in range(5)], labels=['$80\%$', '$60\%$', '$40\%$' ,'$20\%$', '$0\%$'], showfliers=False)
        plt.show()






if __name__ == '__main__':
    
    # boxplots()

    boxplot_randomreduce()