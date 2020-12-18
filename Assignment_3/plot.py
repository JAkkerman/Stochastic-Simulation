# import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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


def boxplots_all_MSE():
    """
    Makes boxplots of found errors, converges best of MAE to MSE
    """

    
    # all_overal_min = []

    best_MAE = []
    last_MAE = []
    best_MSE = []
    last_MSE = []

    for j,error_method in enumerate(['absolute','mean squared']):

        all_best = []
        all_last = []

        

        t,x,y = open_data()

        cooling = 'linear'
        filename = 'SA_'+cooling+'_'+error_method+'.csv'
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            overal_min = [10,[]]

            if error_method == 'absolute':

                best_coeff = []
                last_coeff = []

                for i,row in enumerate(reader):
                    # errors = np.single(np.array(row[8:]))
                    # print(errors)
                    best_coeff += [list(np.single(np.array(row[4:8])))]
                    last_coeff += [list(np.single(np.array(row[0:4])))]

                # print(best_coeff)


                best_xyval = [integrate(i,t,x[0],y[0], x_keys=np.linspace(0,99,100), y_keys=np.linspace(0,99,100)) for i in best_coeff]
                # print(best_xyval)

                best_MAE = [error(x, y, p[0], p[1], np.linspace(0,99,100), np.linspace(0,99,100), error_method='mean squared')  for p in best_xyval]

                last_xyval = [integrate(i,t,x[0],y[0], x_keys=np.linspace(0,99,100), y_keys=np.linspace(0,99,100)) for i in last_coeff]
                last_MAE = [error(x, y, p[0], p[1], np.linspace(0,99,100), np.linspace(0,99,100), error_method='mean squared')  for p in last_xyval]
            
            else:
                for i,row in enumerate(reader):
                    errors = np.single(np.array(row[8:]))
                    best_MSE += [np.single(min(errors))]
                    last_MSE += [np.single(errors[-1])]
                    


        # print(all_best)

                # print(row[:8])
                # print(np.single(min(errors)),'\n')

    # print(overal_min)
    # print(all_best)
    # print(all_last)

    fig, ax = plt.subplots(1,2, figsize=[8,3])
    ax[0].boxplot([last_MAE, last_MSE], labels=['MAE last', 'MSE last'], widths=0.6)
    ax[0].set_title('last results', fontsize=12)
    ax[0].set_ylabel('mean squared error', fontsize=12)

    ax[1].boxplot([best_MAE, best_MSE], labels=['MAE best', 'MSE best'], widths=0.6)
    ax[1].set_title('best results', fontsize=12)
    ax[1].set_ylabel('mean squared error', fontsize=12)

    # all_overal_min += [overal_min]

    # plt.show()
    plt.savefig('boxplot_error_compare.pdf')


def boxplot_randomreduce():
    """
    Plots boxplots for random reduction
    """

    for q in ['x', 'y']:

        fig = plt.subplots(figsize=(7,3))

        cooling = 'linear'
        error_method = 'mean squared'
        filename = 'SA_'+cooling+'_'+error_method+'_removed_data_'+q+'.csv'
        with open(filename) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                errors = []
                for i,row in enumerate(reader):

                    t,x,y = open_data()

                    x_val, y_val = integrate(np.single(np.array(row[0:4])),t,x[0],y[0], x_keys=np.linspace(0,99,100), y_keys=np.linspace(0,99,100))
                    errors += [error(x, y, x_val, y_val, np.linspace(0,99,100), np.linspace(0,99,100), error_method='mean squared')]
        plt.boxplot([errors[30*i:30*i+30] for i in range(5)], labels=['$80\%$', '$60\%$', '$40\%$' ,'$20\%$', '$0\%$'], showfliers=False)
        title = 'predators for different amounts of random reduction'
        if q == 'y':
            title = 'prey for different amounts of random reduction'
        plt.title(title)
        plt.xlabel('reduced percentage')
        plt.ylabel('mean squared error')
        plt.tight_layout()
        plt.show()



def boxplot_reduceboth():
    """
    plots boxplots for random reduction of both time series
    """

    fig = plt.subplots(figsize=(7,3))

    cooling = 'linear'
    error_method = 'mean squared'
    filename = 'SA_'+cooling+'_'+error_method+'_removed_data_xy.csv'

    with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            errors = []
            for i,row in enumerate(reader):
                t,x,y = open_data()

                x_val, y_val = integrate(np.single(np.array(row[4:8])),t,x[0],y[0], x_keys=np.linspace(0,99,100), y_keys=np.linspace(0,99,100))
                errors += [error(x, y, x_val, y_val, np.linspace(0,99,100), np.linspace(0,99,100), error_method='mean squared')]

    plt.boxplot([errors[30*i:30*i+30] for i in range(4)], labels=['$80\%$', '$60\%$', '$40\%$' ,'$20\%$'], showfliers=True)
    plt.title('reduction of both x and y')
    plt.xlabel('reduced percentage')
    plt.ylabel('mean squared error')
    plt.tight_layout()
    plt.savefig('boxplot_randred_xy.pdf')


def boxplot_HC_SA():
    HC_MSE = []
    SA_MSE = []

    with open('hillclimber_res.csv') as csvfile:
        for i,row in enumerate(csvfile):
            HC_MSE += [np.single(row)]

    with open('SA_linear_mean squared.csv') as csvfile:

        reader = csv.reader(csvfile, delimiter=',')

        for i,row in enumerate(reader):
            errors = np.single(np.array(row))
            print(errors[0:10])
            print(errors[8:10])
            errors = errors[8:]
            SA_MSE += [np.single(min(errors))]

            plt.plot(errors, range(len(errors)))

    plt.yscale('log')
    # plt.show()

    # print(SA_MSE)

    # print(np.average(SA_MSE))
    # print(np.average(HC_MSE))

    # fig, ax = plt.subplots(figsize=[5,4])

    # plt.boxplot([HC_MSE, SA_MSE], labels=['HC', 'SA'], showfliers=True, widths=0.6)
    # plt.title('performance of HC and SA')
    # # plt.xlabel('reduced percentage')
    # plt.ylabel('mean squared error')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('boxplot_HC_SA.pdf')

    print(scipy.stats.ttest_ind(HC_MSE, SA_MSE))

    SA_MSE = np.array(SA_MSE)
    SA_MSE = SA_MSE[abs(SA_MSE - np.mean(SA_MSE)) < 1.96 * np.std(SA_MSE)]

    print(scipy.stats.ttest_ind(HC_MSE, SA_MSE))



if __name__ == '__main__':
    
    # boxplots()

    # boxplot_randomreduce()

    # boxplots_all_MSE()

    # boxplot_reduceboth()

    boxplot_HC_SA()