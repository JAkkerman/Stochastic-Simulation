import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import csv
import pandas as pd



def open_data():
    """
    Opens data from csv
    """
    with open('predator-prey-data.csv', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        t = []
        y = []
        x = []

        for i,row in enumerate(reader):
            if i != 0:
                t += [np.double(row[0])]
                x += [np.double(row[1])]
                y += [np.double(row[2])]

    return t,x,y


def integrate(param,t,x,y):
    a = param[0]
    b = param[1]
    g = param[2]
    d = param[3]
    time = np.linspace(0, 20, 100)

    def eq(begin_pop,t,a,b,g,d):
        x,y = begin_pop
        dxdt = a*x - b*x*y
        dydt = d*x*y - g*y
        # dxdt = d*x*y - g*x
        # dydt = a*y - b*x*y
        return dxdt, dydt

    begin_pop = x[0],y[0]
    numint = odeint(eq, begin_pop, time, args=(a,b,g,d))

    x_val, y_val = numint.T

    return x_val, y_val


def error(x, y, x_val, y_val, method='mean squared'):
    '''
    Computes the loss between data and predictions
    x, y : data
    x_val, y_val: predictions
    method: which loss function to use 'mean squared' or 'absolute'
    '''
    if method == 'mean squared':
        error_x = np.average([(np.array(x)-np.array(x_val))**2])
        error_y = np.average([(np.array(y)-np.array(y_val))**2])

    elif method == 'absolute':
        error_x = np.mean(np.abs([np.array(x) - np.array(x_val)]))
        error_y = np.mean(np.abs([np.array(y) - np.array(y_val)]))

    return error_x + error_y


def save_to_csv(res):
    """
    Saves results to csv
    """
    print(res)
    data = {}
    lastsols = [list(exp[0][0]) for exp in res]
    bestsols = [list(exp[1][0]) for exp in res]
    errors   = [list(exp[2]) for exp in res]

    data = {'lastsols': lastsols, 'bestsols': bestsols, 'errors': errors}

    # for i,exp in enumerate(res):
    #     data = {'last': last_sol[0], 'best': best_sol[0], 'errors': errors}
    #     data[i] = []
    
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf='SA_exp_fulldataset.csv')



def hillclimber(t,x,y, method='mean squared', plot_error=False, plot_fit=False, steps=1000, 
                n_runs=4):
    '''
    t, x, y: time, predator, prey data
    error: 'absolute'or 'mean squared', defines which loss function to use
    plot_error: boolean, whether or not to plot the evolution of the error
    plot_fit: boolean, whether or not to plot the final fit
    steps: int, how often the algorithm is executed
    n_runs: int, how often the algorithm starts again from a random point
    ''' 

    # dictionary with key, value = run, (errors, params)
    output = {}

    for run in range(n_runs):

        param = np.random.uniform(0.1, 3, size=4).tolist()
        x_est, y_est = integrate(param,t,x,y)

        error_xy = error(x,y,x_est,y_est, method=method)

        all_error_xy = []

        for i in range(steps):
            change = np.random.choice(param)
            new_param = param
            new_param[param.index(change)] = np.abs(change + np.random.normal(0,0.1))
            
            x_est, y_est = integrate(param,t,x,y)
            new_error_xy = error(x,y,x_est,y_est, method=method)


            if new_error_xy < error_xy:
                param = new_param
                error_xy = new_error_xy

        all_error_xy += [error_xy]
        output[run] = [error_xy, new_param]

    # select best fit
    min_error = min([val[0] for key, val in output.items()])
    best_param = [val[1] for key, val in output.items() if val[0] == min_error][0]

    if plot_error ==  True:
        plt.plot(range(steps), all_error_xy)
        plt.show()

    # select best fit
    min_error = min([val[0] for key, val in output.items()])
    best_param = [val[1] for key, val in output.items() if val[0] == min_error][0]
    x_est, y_est = integrate(best_param,t,x,y)

    if plot_fit == True:
        fig , ax = plt.subplots(1,2, figsize=(7, 4))
        ax[0].plot(t, x_est, label='est')
        ax[0].plot(t, x, label='real')
        ax[0].set_title('predator')
        ax[0].legend()
        # for i,tval in enumerate(t):
        #     plt.arrow(tval, min(x[i], x_val[i]), 0, max(x[i],x_val[i]))
        ax[1].plot(t, y_est, label='est')
        ax[1].plot(t,y, label='real')
        ax[1].set_title('prey')
        ax[1].legend()

        plt.show()

    return best_param

def SA(t, x, y, method='mean squared', cooling='linear'):
    '''
    Simulated annealing algorithm
    t, x, y: time, predator, prey data
    p0: array_like, initial parameters
    error: 'absolute'or 'mean squared', defines which loss function to use
    '''

    all_errors = []

    # initializations
    dT = 10e-5   # step size
    Tf = 10e-5   # final temperature
    if cooling != 'linear':
        Tf = 0.065
    Tc = 1   # current temperature

    count = 1

    # param = list(np.random.uniform(0, 1, size=2)) + list(np.random.uniform(2, 4, size=2))
    param = np.random.uniform(0, 2, size=4)
    x_current, y_current = integrate(param, t, x, y)
    error_current = error(x, y, x_current, y_current, method=method)
    all_errors += [error_current]
    best_sol = [param, error_current]

    while Tc > Tf:
        # choose new parameters
        noise = np.array([np.random.normal(0, 0.1), 0, 0, 0])
        np.random.shuffle(noise)
        # print(noise)
        param_neighbour = np.abs(np.array(param) + noise)
        x_neighbour, y_neighbour = integrate(param_neighbour, t, x, y)
        error_neighbour = error(x, y, x_neighbour, y_neighbour, method=method)

        diff = error_current - error_neighbour

        if error_neighbour < error_current:
            error_current = error_neighbour
            param = param_neighbour

        else:
            if np.random.uniform(0, 1) < np.exp(diff / Tc):
                error_current = error_neighbour
                param = param_neighbour

        # print('prob',np.exp(diff / Tc))

        all_errors += [error_current]

        if error_current < best_sol[1]:
            best_sol = [param, error_current]

        if cooling =='linear':
            Tc -= dT
        else:
            Tc = 0.7/np.log(1+count)

        # print(Tc)
        count += 1

    print('final parameters: \na: ',param[0],'\nb: ',param[1],'\ng: ',param[2],'\nd: ',param[3])
    print('best parameters: \na: ',best_sol[0][0],'\nb: ',best_sol[0][1],'\ng: ',best_sol[0][2],'\nd: ',best_sol[0][3])
    print('final ratio a/b: ', round(param[0]/param[1],2), ',final ratio g/d:', round(param[2]/param[3],2))
    print('best ratio a/b: ', round(best_sol[0][0]/best_sol[0][1],2), ',best ratio g/d:', round(best_sol[0][2]/best_sol[0][3],2))

    x_current, y_current = integrate(param, t, x, y)
    x_bestsol, y_bestsol = integrate(best_sol[0], t, x, y)

    fig , ax = plt.subplots(1,2, figsize=(7, 4))
    ax[0].plot(t, x_current, label='last est')
    ax[0].plot(t, x_bestsol, label='best est')
    ax[0].plot(t, x, label='real')
    ax[0].set_title('predator')
    ax[0].legend()
    ax[1].plot(t, y_current, label='last est')
    ax[1].plot(t, y_bestsol, label='best est')
    ax[1].plot(t,y, label='real')
    ax[1].set_title('prey')
    ax[1].legend()

    plt.show()

    plt.plot(np.arange(0, len(all_errors), 1), all_errors)
    plt.show()

    return [param, error_current], best_sol, all_errors
    


if __name__ == "__main__":

    error_type = 'means_sq'

    t,x,y = open_data()

    n_experiments = 1
    # param=[0.3,0.3,0.3,0.3]
    # x_val, y_val = integrate(param,t,x,y)

    # if error_type == 'means_sq':
    #     error_x, error_y = means_sq(x,y,x_val,y_val)
    #     print(error_x, error_y)

    # params = hillclimber(t,x,y, plot_fit=True, n_runs=4, steps=2000)
    # res = []
    for i in range(n_experiments):
        last_sol, best_sol, errors = SA(t, x, y, method='absolute', cooling='logarithmic')
        # res += [[last_sol, best_sol, errors]]
    # save_to_csv(res)